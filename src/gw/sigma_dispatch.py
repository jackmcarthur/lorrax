"""Mode-orthogonal Σ_xc dispatch.

A single entry point :func:`compute_sigma_xc` that the QSGW iteration
map calls regardless of compute mode (X_ONLY, COHSEX, GN_PPM, HL_PPM).
The dispatch decides which Σ kernel runs internally; the iteration map
sees one signature and one result type.

The dispatch is EXHAUSTIVE over ``gw_config.ComputeMode``: a mode with
no kernel here is refused by name, not absorbed by the last branch.
MPA has a branch below and is no longer gated at entry: its
``gw_config.UNIMPLEMENTED_MODES`` row was deleted once the metal pipeline
ran end to end.  The site-level refusals are the safety now — a metal plan
without an ``OccupationState`` is still refused by name.

Returned :class:`SigmaResult` always contains ``v_h_kij_ry``,
``sigma_x_kij_ry``, and a single ``sigma_xc_kij_ry`` representing the
total exchange-correlation contribution to ``H_QP = kin_ion + V_H +
Σ_xc``.  Dynamic-mode-only diagnostics (full ω-grid Σ_c, on-shell diagonals,
head decomposition) live as optional fields and are populated only when
the mode produces them.

Every band-indexed field comes back in the basis of the ``wfns`` bundle
this module was handed; the four field tuples beside the dataclass say
which of them a consumer sees in which basis.

The Σ kernels live under ``cohsex_sigma`` (static channels),
``ppm_pipeline`` (two-point PPM Σ_c), ``gw.mpa`` (multipole Σ_c) and
``qsgw_utils`` (the QSGW Hermitisation); this module orchestrates them
and owns the shared dynamic-Σ finalization seam
(:func:`finalize_dynamic_sigma`: head injection, at-DFT interpolation,
file write, QSGW build).
"""

from __future__ import annotations

import functools as _functools
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.units import RYD_TO_EV
from .gw_config import (
    BRACKET_SCHEME_DEFAULT, ComputeMode, SigmaChannel,
    band_extrapolation_is_consumable,
    mode_builds_channels, refuse_explicit_gij_under_low_mem_bands,
    refuse_unimplemented_compute_mode,
    sigma_stage_modes)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SigmaResult:
    """Outputs of one full Σ pipeline call.

    Always populated
    ----------------
    v_h_kij_ry           : (nk, nb, nb)   Total direct field
                                          ``V_H + H_T`` (replicated)
    v_h_scalar_kij_ry    : (nk, nb, nb)   Scalar charge Hartree only
    h_transverse_kij_ry  : (nk, nb, nb)   Transverse current Hartree, or
                                          None for charge-only runs
    sigma_x_kij_ry       : (nk, nb, nb)   Bare exchange (replicated)
    sigma_xc_kij_ry      : (nk, nb, nb)   Exchange-correlation total going
                                          into ``H_QP = kin_ion + V_H + Σ_xc``.
                                          Static modes: Σ_SX + Σ_COH (with
                                          head).  Dynamic modes: Σ_x + Σ_c^QSGW.

    Static-mode-only (None in PPM)
    ------------------------------
    sigma_sx_kij_ry      : (nk, nb, nb)   Σ_SX with head
    sigma_coh_kij_ry     : (nk, nb, nb)   Σ_COH with head

    Dynamic-mode-only (None in static)
    ----------------------------------
    sigma_c_omega_kij_ry      : (nω, nk, nb, nb), sharded P(None,None,'x','y')
                                Full ω-grid Σ_c (post-head); drives eqp1
                                Z-factor central difference.
    sigma_c_at_dft_diag_ev    : (nk, nb)  diag(Σ_c) at E_DFT (eV).
    omega_dft_rel_ev          : (nk, nb)  E_DFT − E_F (eV).
    e_eval_ev                 : (nk, nb)  the energies the QSGW ansatz
                                EVALUATED this Σ at — ``e_qp_ev``, i.e.
                                E_DFT for a one-shot call and the map's
                                input (converged, under self-consistency)
                                QP energies for an SC iteration.  eqp1's
                                linearization is centred here; see
                                ``eqp_bgw.assemble_eqp``.
    omega_grid_ev             : (nω,)     ω-grid in eV.
    omega_grid_ry             : (nω,)     ω-grid in Ry.
    head_sigma_diag_w_kn_ry   : (nω, nk, nb)  Dynamic q→0 head diagonal.
    sigma_omega_h5_path       : str       on-disk Σ_c(ω) HDF5 path.

    Basis
    -----
    Every band-indexed field is built in the basis of the ``wfns`` bundle
    ``compute_sigma_xc`` was handed.  Which basis that is, per field,
    after the object reaches a consumer: see the four tuples below.
    """

    v_h_kij_ry: jax.Array
    sigma_x_kij_ry: jax.Array
    sigma_xc_kij_ry: jax.Array
    v_h_scalar_kij_ry: jax.Array | None = None
    h_transverse_kij_ry: jax.Array | None = None
    sigma_sx_kij_ry: jax.Array | None = None
    sigma_coh_kij_ry: jax.Array | None = None
    #: The un-extrapolated (N₃, ordinary full-band) QSGW Σ_xc, populated only
    #: when ``use_band_extrapolation`` is driving this stage.  ``sigma_xc_
    #: kij_ry`` is then the EXTRAPOLATED one and is what enters H; this twin
    #: exists so the driver can diagonalize both once per iteration and print
    #: the correction at the eqp level.  None everywhere else, which is what
    #: keeps the default path's object graph unchanged.
    sigma_xc_kij_ry_unextrap: jax.Array | None = None
    sigma_c_omega_kij_ry: jax.Array | None = None
    sigma_c_at_dft_diag_ev: np.ndarray | None = None
    omega_dft_rel_ev: np.ndarray | None = None
    e_eval_ev: np.ndarray | None = None
    omega_grid_ev: np.ndarray | None = None
    omega_grid_ry: np.ndarray | None = None
    head_sigma_diag_w_kn_ry: np.ndarray | None = None
    sigma_omega_h5_path: str | None = None
    efermi_dft_ev: float | None = None
    #: Which convention ``efermi_dft_ev`` IS — "fixed-N mu" or "midgap".
    #: Carried beside the number so the end-of-SC writer stamps the same
    #: answer the finalize interpolated with (audit A2), instead of
    #: re-deriving it from a config key one layer further from the choice.
    omega_reference_provenance: str | None = None
    #: :class:`gw.dynamic_sigma.OmegaCoverage` for the at-DFT interpolation
    #: — WHICH (k, n) the ω grid actually sampled, and what was done with the
    #: rest.  Carried on the result rather than recomputed by each writer
    #: because ``sigma_c_at_dft_diag_ev`` cannot say by itself which of its
    #: cells are measurements and which are grid endpoints, and the Na
    #: semicore run shipped 41.3 % endpoints as if they were Σ.
    omega_coverage: object | None = None

    def __post_init__(self) -> None:
        if self.v_h_scalar_kij_ry is None:
            if self.h_transverse_kij_ry is not None:
                raise ValueError(
                    "SigmaResult: H_T requires an explicit scalar V_H component")
            # Compatibility for charge-only constructors: the historical
            # aggregate is exactly the scalar component in that sector.
            object.__setattr__(self, "v_h_scalar_kij_ry", self.v_h_kij_ry)


# ---------------------------------------------------------------------------
# Which basis each field is in
# ---------------------------------------------------------------------------
#
# ``compute_sigma_xc`` builds every band-indexed output in the basis of
# the ``wfns`` bundle it was handed: the current QP basis under
# self-consistency, the DFT basis for a one-shot call.  The SC driver
# owes the post-Σ seam a DFT-basis object, so its finalize
# (``sc_iteration.run_sc_driver``) rotates the matrices named in
# ``ROTATED_TO_DFT_FIELDS`` with ``O_DFT = U·O_QP·U†`` and leaves the
# rest untouched.  The returned object therefore holds TWO BASES AT
# ONCE, deliberately; these four tuples are the record of which field is
# in which, and they partition ``dataclasses.fields(SigmaResult)``
# exactly (``tests/test_sigma_result_basis.py`` fails when they do not).

#: Band-basis matrices, rotated together or not at all.  A Σ channel
#: added to the dataclass and NOT added here comes back from the SC
#: driver in the QP basis with the right shape, dtype and sharding: no
#: shape gate, finiteness gate or SC invariance gate can see it (the
#: ``max_iter=1`` invariance gate runs at U = identity, where the two
#: bases agree by construction).
ROTATED_TO_DFT_FIELDS = (
    "v_h_kij_ry",
    "v_h_scalar_kij_ry",
    "h_transverse_kij_ry",
    "sigma_x_kij_ry",
    "sigma_xc_kij_ry",
    # The un-extrapolated N₃ twin of ``sigma_xc_kij_ry``, present only when
    # the band extrapolation is driving.  It is here and not in
    # ``SIGMA_BASIS_FIELDS`` because it is the SAME KIND OF OBJECT as the
    # field above it -- a QSGW static Σ_xc matrix -- and the driver's whole
    # use for it is to build a second H in the DFT basis and diagonalize it
    # beside the first.  A twin that came back in the QP basis while its
    # partner came back rotated would report a "correction" that was mostly
    # the basis difference.
    "sigma_xc_kij_ry_unextrap",
    "sigma_sx_kij_ry",
    "sigma_coh_kij_ry",
)

#: Left in the Σ compute basis on purpose — do NOT rotate these.
#: ``sigma_c_omega_kij_ry`` is the operand of the QSGW ansatz
#: ``Σ_ij^QSGW = ½[Σ_ij(E_i) + Σ_ij(E_j)]ʰ`` (``qsgw_utils.py``:402),
#: whose band indices must label the states whose energies E_i, E_j it
#: is evaluated at, so the construction is only itself in that basis;
#: it is also the (nω, nk, nb, nb) sharded tensor and the contents of
#: sigma_mnk.h5.  The other three are already band DIAGONALS, on which a
#: basis rotation does not act element-wise.  ``e_eval_ev`` belongs here
#: for the strongest form of that reason: it is the list of energies
#: E_i whose band index MUST be the one the cube's band index is, since
#: the ansatz pairs them element-wise.
SIGMA_BASIS_FIELDS = (
    "sigma_c_omega_kij_ry",
    "sigma_c_at_dft_diag_ev",
    "head_sigma_diag_w_kn_ry",
    "e_eval_ev",
)

#: Band-indexed but read from the WFN file, hence DFT basis on every
#: path: ``omega_dft_rel_ev`` is E_DFT − E_F, built from
#: ``get_enk_bandrange`` in ``dynamic_sigma.eval_sigma_c_at_dft_energies``.
#: TRAP: under self-consistency it labels bands by the DFT index while
#: ``sigma_c_at_dft_diag_ev`` — the interpolation it drives, in that same
#: function — labels them by the QP index.
DFT_BASIS_FIELDS = (
    "omega_dft_rel_ev",
    # Its ``mask_kn`` is derived from ``omega_dft_rel_ev`` in the same
    # function, so it carries the same band labelling and the same trap.
    "omega_coverage",
)

#: No band index; basis-independent.
BASIS_FREE_FIELDS = (
    "omega_grid_ev",
    "omega_grid_ry",
    "sigma_omega_h5_path",
    "efermi_dft_ev",
    "omega_reference_provenance",
)


# ---------------------------------------------------------------------------
# H₀'s Hartree term: resolve the source once, cache the array
# ---------------------------------------------------------------------------

#: (kin_ion path, b0, b3, resolved source, mesh identity) →
#: (source, V_H (nk,nb,nb) Ry | None).  Mesh identity is the STABLE
#: shape/axis/device-id tuple from :func:`_mesh_cache_key`, not ``id()``.
#: The QSGW loop calls ``compute_sigma_xc`` once per SC iteration and the
#: exact V_H does not change with the band basis (it is a fixed operator in
#: the DFT basis, and ``rotate_wavefunctions`` handles the basis change on
#: H₀ as a whole), so re-reading — or worse, re-running the ``gspace``
#: build, every iteration would be pure waste.
#:
#: WHAT THE CACHE ASSUMES, AND WHEN IT MUST BE DROPPED.  The statement
#: above is exactly true for QSGW **at fixed density** — the only kind the
#: driver runs today: the SC loop rotates the band basis, it does not
#: rebuild ρ.  A density-updating QSGW breaks the assumption, because then
#: V_H is a function of the current occupied orbitals and *does* change
#: every iteration.  The kernel is ready for that
#: (``compute_hartree_matrix(..., psi_rotation=U[:, :, :nocc])`` builds ρ
#: from the rotated ψ) and the cost is affordable — see the QSGW readiness
#: note in the scorecard — but such a loop MUST call
#: :func:`invalidate_hartree_cache` at the top of each iteration, or it
#: will silently keep iteration 0's Hartree potential for ever.
_hartree_cache: dict = {}


def _mesh_cache_key(mesh_xy) -> tuple:
    """STABLE identity of a Mesh for cache keying: axis names + extents
    plus the flat device-id tuple.

    The previous key component was ``id(mesh_xy)``, which a new Mesh can
    REUSE after the old one is garbage-collected — a false hit would then
    hand back V_H arrays sharded on a dead mesh.  Conversely, two JAX
    ``Mesh`` objects over the same devices/axes compare equal and their
    ``NamedSharding``s compose, so a hit across equivalent Mesh OBJECTS
    (which ``id()`` needlessly missed) is correct.  (audit fix/zq
    2026-07-28)
    """
    return (tuple(mesh_xy.shape.items()),
            tuple(int(d.id) for d in mesh_xy.devices.flat))


def _place_band_rotation(U, mesh_xy, dtype):
    """``U`` as a global array at ``qsgw_density.band_rotation_spec``.

    A no-op ``device_put`` on the SC path (every producer already emits
    that layout).  The host branch is not dead: on a reduced k-set the
    k-star broadcast can leave a numpy U, and plain ``jax.device_put`` of
    a host array onto a multi-process sharding fires JAX's hidden replica
    ``assert_equal`` all-gather (common.collectives header).
    """
    from .qsgw_density import band_rotation_spec

    sh = NamedSharding(mesh_xy, band_rotation_spec())
    if isinstance(U, jax.Array):
        return jax.device_put(jnp.asarray(U, dtype=dtype), sh)
    return device_put_process_local(np.asarray(U, dtype=dtype), sh)


@_functools.partial(jax.jit, static_argnames=("mesh",))
def _rotate_v_h_to_qp(v_h_dft, U, *, mesh: Mesh):
    """``V_H^QP = U† · V_H^DFT · U`` with U kept at ``band_rotation_spec``.

    The RESULT is pinned replicated because ``SigmaResult.v_h_kij_ry``
    feeds the k-star select and the host readbacks downstream; U is not,
    because it is the (nk, nb, nb) object that reaches 9.2 GB/rank at
    nk=144/nb=2000.
    """
    from .qsgw_density import rotate_band_matrix

    out = rotate_band_matrix(v_h_dft, U, mesh=mesh, to_qp=True)
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, P(None, None, None)))


def invalidate_hartree_cache() -> None:
    """Drop the memoised V_H so the next call rebuilds it.

    Required by any self-consistency loop that updates the DENSITY (not
    just the band basis) — see the note on :data:`_hartree_cache`.
    """
    _hartree_cache.clear()


def resolve_external_hartree(config, meta, band_slices, mesh_xy, *,
                             wfn=None, sym=None, print_fn=print):
    """``(source, V_H | None, V_H_T | None)`` for this run.

    ``source`` is one of ``'stored' | 'folded' | 'isdf' | 'gspace'``.
    The scalar array is returned only for ``stored`` / ``gspace``; ``folded``
    means "V_H is inside kin_ion's values, add nothing", and ``isdf``
    means "keep the ISDF quadrature".  A bispinor run additionally receives
    the exact transverse direct matrix, independently of the scalar source:
    read from its separate kin_ion dataset except when ``gspace`` builds both
    fields in one occupied-WFN transaction.
    """
    from file_io.kin_ion import (
        load_hartree_submatrix, load_transverse_hartree_submatrix,
        resolve_hartree_source,
    )

    path = config.paths.kin_ion_file
    requested = getattr(config, "hartree_source", "auto")
    source = resolve_hartree_source(path, requested, print_fn=print_fn)
    require_transverse = bool(config.bispinor)
    if require_transverse and source == "folded":
        raise ValueError(
            "kinetic-balance bispinor GW refuses a legacy folded kin_ion "
            "artifact; regenerate separate authenticated scalar and "
            "transverse direct Hartree datasets.")
    key = (os.path.abspath(path), int(band_slices.b0), int(band_slices.b3),
           source, require_transverse, _mesh_cache_key(mesh_xy))
    if key in _hartree_cache:
        return _hartree_cache[key]

    v_h = None
    v_h_t = None
    if require_transverse and source != "gspace":
        v_h_t = load_transverse_hartree_submatrix(
            path, band_slices.b0, band_slices.b3, mesh=mesh_xy)
        print_fn(
            "  V_H^T: exact periodic G-space Dirac-current matrix read "
            "from kin_ion.h5; G=0 is zero and no exchange head enters.")
    if source == "stored":
        v_h = load_hartree_submatrix(
            path, band_slices.b0, band_slices.b3,
            mesh=mesh_xy)
        print_fn("  V_H: exact FFT-grid matrix read from kin_ion.h5's "
                 "'v_hartree' dataset; the ISDF quadrature is not used.")
    elif source == "gspace":
        if wfn is None or sym is None:
            raise ValueError(
                "hartree_source=gspace needs the WFN loader and SymMaps.")
        print_fn("  V_H: rebuilding the exact FFT-grid matrix on the fly "
                 "(hartree_source=gspace) — DISTRIBUTED over the run's own "
                 "mesh (ρ: one psum; Poisson: replicated; ⟨mk|V_H|nk⟩: "
                 "k-partitioned + one gather).")
        # Lazy: pulls in the psp stack, which the ISDF path does not need.
        # ``replicate_to_mesh`` is generic k-partition plumbing and comes
        # from the SERVICE; only ``compute_hartree_matrix`` (real physics,
        # and the reason for the lazy import) comes from the driver.  Both
        # used to be imported from ``kin_ion_io``, which made a library
        # module depend on a CLI for a collective helper.
        from common.collectives import replicate_to_mesh
        from gw.kin_ion_io import compute_hartree_matrix
        exact_hartree = compute_hartree_matrix(
            wfn, sym, meta,
            truncation_2d=(int(config.sys_dim) == 2),
            nb=int(band_slices.b3), mesh=mesh_xy,
            band_chunk_size=int(config.memory.band_chunk_size),
            include_transverse=require_transverse,
            print_fn=print_fn)
        v_h_np = (exact_hartree.charge if require_transverse
                  else exact_hartree)
        # ``compute_hartree_matrix`` hands every rank the same host array;
        # publish it as a genuinely REPLICATED global array so it composes
        # with the (global) ``sig_h`` it replaces.  ``jnp.asarray`` here
        # would have produced a single-device array — fine at P=1, an
        # operand-sharding mismatch at P>1.
        v_h = replicate_to_mesh(
            np.ascontiguousarray(
                v_h_np[:, band_slices.b0:band_slices.b3,
                       band_slices.b0:band_slices.b3]),
            mesh_xy)
        if require_transverse:
            v_h_t = replicate_to_mesh(
                np.ascontiguousarray(
                    exact_hartree.transverse[
                        :, band_slices.b0:band_slices.b3,
                        band_slices.b0:band_slices.b3]),
                mesh_xy)
    elif source == "folded":
        print_fn("  V_H: LEGACY folded kin_ion.h5 — V_H is inside its values; "
                 "the ISDF sig_h is suppressed to avoid double counting.")
    else:
        print_fn("  V_H: ISDF V_q[0] quadrature (hartree_source=isdf); H0 "
                 "therefore depends on the centroid count.")

    _hartree_cache[key] = (source, v_h, v_h_t)
    return source, v_h, v_h_t


# ---------------------------------------------------------------------------
# Dynamic-Sigma finalization (shared by every frequency ansatz)
# ---------------------------------------------------------------------------

def finalize_dynamic_sigma(
    sigma_c_body_omega: jax.Array,
    head_sigma_diag_w_kn_ry: np.ndarray | None,
    *,
    sig_x: jax.Array,
    sig_h: jax.Array,
    v_h_scalar: jax.Array | None = None,
    h_transverse: jax.Array | None = None,
    e_qp_ev: np.ndarray,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    band_slices,
    input_dir: str,
    write_sigma_omega_h5: bool = True,
    band_extrapolation: dict | None = None,
    sigma_c_body_omega_unextrap: jax.Array | None = None,
    print_fn: Callable = print,
    efermi_ry=None,
    efermi_provenance=None,
) -> SigmaResult:
    """Finalize one dynamic Sigma ansatz without knowing its pole model.

    The ansatz supplies only its body cube and band-diagonal q->0 head.
    This seam performs the common head injection, interpolation at DFT
    energies, canonical file write and QSGW build, then returns the uniform
    :class:`SigmaResult`.  The fixed-point solve and optional scissor remain
    in :func:`gw.qsgw_utils.solve_qp`, which consumes the retained omega cube;
    there is still one owner of that energy-update policy.

    ``e_qp_ev`` — THE ENERGIES THIS SPECTRUM IS EVALUATED AT — is both the
    QSGW build's evaluation point below and, carried out on the result as
    ``e_eval_ev``, the point eqp1's linearization is centred at.  It is
    E_DFT for a one-shot call (so the eqp writers are unchanged there,
    bit-for-bit) and the map's input QP energies under self-consistency,
    which at convergence is where the QP poles are.  The at-DFT
    interpolation beside it stays at E_DFT: it is what eqp0 and the
    ``sig_c(Edft)`` diagnostics mean.
    """
    import common.timing as timing
    from .dynamic_sigma import (
        add_head_sigma_diag,
        eval_sigma_c_at_dft_energies,
        sigma_omega_output_path,
        write_sigma_omega,
    )
    from .qsgw_utils import build_qsgw_sigma_xc

    if v_h_scalar is None:
        if h_transverse is not None:
            raise ValueError(
                "finalize_dynamic_sigma: H_T requires scalar V_H")
        v_h_scalar = sig_h

    with timing.section("gw_jax.dynamic_sigma_finalize"):
        sigma_c_omega = add_head_sigma_diag(
            sigma_c_body_omega, head_sigma_diag_w_kn_ry)

        (sigma_c_at_dft_ev,
         omega_dft_rel_ev,
         efermi_dft_ev,
         omega_reference_provenance,
         omega_coverage) = eval_sigma_c_at_dft_energies(
            sigma_c_omega,
            config=config,
            band_slices=band_slices, wfn=wfn, sym=sym, meta=meta,
            mesh_xy=mesh_xy, print_fn=print_fn,
            efermi_ry=efermi_ry,
            efermi_provenance=efermi_provenance,
        )

        # Static Sigma_x is added in the QSGW kernel.  E_F here is the SAME
        # reference the interpolation above used — one omega reference per
        # finalize, or the two reads sample different grid positions.
        # Formed BEFORE the write because the write stamps it: see below.
        omega_grid_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
        e_qp_rel_ev = (
            np.asarray(e_qp_ev, dtype=np.float64) - efermi_dft_ev)

        if write_sigma_omega_h5:
            sigma_omega_h5_path = write_sigma_omega(
                sigma_c_omega,
                sig_x=sig_x, sig_h=sig_h,
                v_h_scalar=v_h_scalar,
                h_transverse=h_transverse,
                config=config, input_dir=input_dir,
                meta=meta, mesh_xy=mesh_xy,
                omega_reference_ev=efermi_dft_ev,
                omega_reference_provenance=omega_reference_provenance,
                # THE ENERGIES THIS SPECTRUM WAS EVALUATED AT, stamped.
                # Until 2026-08-22 the file recorded the omega REFERENCE but
                # not the evaluation point, so a from-disk reassembly
                # (`eqp_bgw.make_eqp_bgw`) could not tell a one-shot cube
                # from a self-consistent one and silently reverted to the
                # at-E_DFT linearization -- which is right for one-shot and
                # a different, wrong calculation for SC.
                eval_energies_rel_ev=e_qp_rel_ev,
                # MEASURED, not assumed: the two candidates differ exactly
                # when the evaluation spectrum is not E_DFT, and that is an
                # array comparison this function can make and a downstream
                # reader cannot.
                eval_energies_provenance=(
                    "at_e_dft"
                    if np.array_equal(e_qp_rel_ev,
                                      np.asarray(omega_dft_rel_ev,
                                                 dtype=np.float64))
                    else "self_consistent_qp"),
                omega_coverage=omega_coverage,
                sym=sym, band_extrapolation=band_extrapolation,
                print_fn=print_fn,
            )
        else:
            sigma_omega_h5_path = sigma_omega_output_path(config, input_dir)
        sig_x_rep = device_put_process_local(
            sig_x, NamedSharding(mesh_xy, P(None, None, None)))
        sigma_xc_qsgw, qsgw_diag = build_qsgw_sigma_xc(
            sigma_c_omega, sig_x_rep,
            omega_grid_ev, e_qp_rel_ev, mesh_xy,
        )
        print_fn(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
                 f"({100*qsgw_diag['frac_clipped']:.1f}%)")

        # ── THE SECOND QSGW MATRIX: N₃, UN-EXTRAPOLATED ─────────────────
        # Built only when the band extrapolation is driving, and built the
        # SAME way as the one above so the pair differs in exactly one thing:
        # whether Σ_c carries the extrapolated band tail.  Same head (the
        # q->0 head is band-diagonal with no unoccupied sum, hence
        # bracket-independent), same ω grid, same E_qp, same clipping policy.
        # Its only consumer is the driver's side-by-side eqp report; it is
        # never mixed into the carry.
        sigma_xc_qsgw_unextrap = None
        if sigma_c_body_omega_unextrap is not None:
            sigma_c_omega_unextrap = add_head_sigma_diag(
                sigma_c_body_omega_unextrap, head_sigma_diag_w_kn_ry)
            sigma_xc_qsgw_unextrap, _ = build_qsgw_sigma_xc(
                sigma_c_omega_unextrap, sig_x_rep,
                omega_grid_ev, e_qp_rel_ev, mesh_xy,
            )

        # Only append when this call created the base file.  SC iterations
        # pass False and append once, in the cube's own basis, at convergence.
        if write_sigma_omega_h5:
            from .qsgw_utils import write_qsgw_sigma_cube
            write_qsgw_sigma_cube(
                sigma_omega_h5_path, sigma_xc_qsgw,
                config=config, print_fn=print_fn)

    return SigmaResult(
        v_h_kij_ry=sig_h,
        v_h_scalar_kij_ry=v_h_scalar,
        h_transverse_kij_ry=h_transverse,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_qsgw,
        sigma_xc_kij_ry_unextrap=sigma_xc_qsgw_unextrap,
        sigma_c_omega_kij_ry=sigma_c_omega,
        sigma_c_at_dft_diag_ev=sigma_c_at_dft_ev,
        omega_dft_rel_ev=omega_dft_rel_ev,
        # The energies THIS call's Σ spectrum was evaluated at, kept so the
        # eqp1 writer can centre its linearization where the QP pole
        # actually is.  Absolute eV (not ω-relative) on purpose: the
        # consumer forms the relative pair with the one ω reference it also
        # forms ``e_dft_rel_ev`` with, rather than round-tripping this
        # through a second subtraction and addition.
        e_eval_ev=np.asarray(e_qp_ev, dtype=np.float64),
        omega_grid_ev=config.omega_grid_ev,
        omega_grid_ry=config.omega_grid_ry,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
        sigma_omega_h5_path=sigma_omega_h5_path,
        efermi_dft_ev=efermi_dft_ev,
        omega_reference_provenance=omega_reference_provenance,
        omega_coverage=omega_coverage,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_sigma_xc(
    mode: ComputeMode,
    *,
    wfns,
    V_q: jax.Array,
    W_by_role: dict,
    e_qp_ev: np.ndarray | None,
    static_head_terms,
    head_resolver,
    quad,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    band_slices,
    input_dir: str,
    Gij: jax.Array | None = None,
    wfns_transverse=None,
    bispinor_v_q_path: str | None = None,
    photon_response=None,
    write_sigma_omega_h5: bool = True,
    hartree_basis_rotation: jax.Array | None = None,
    omit_v_h: bool = False,
    iteration_head=None,
    occupation_state=None,
    print_fn: Callable = print,
) -> SigmaResult:
    """One-line entry point: build the full Σ_xc + V_H given the current
    wfn bundle and screened W's.

    Parameters
    ----------
    mode
        Compute-mode pivot.  Determines which Σ kernel chain runs and
        which roles in ``W_by_role`` are consulted.
    wfns
        ``Wavefunctions`` bundle in the *current* QP basis (or DFT basis
        for the iter-0 / one-shot call).
    V_q
        Bare Coulomb in flat-q ISDF basis.
    W_by_role
        Screened-Coulomb dict produced by
        :func:`gw.screening.compute_screening`, keyed by symbolic role.
        Conventional roles consumed here:

        * ``"static"`` — W(ω = 0).  Used by COHSEX (Σ_SX, Σ_COH) and as
          the ω-zero anchor for the PPM two-point fit.
        * ``"probe"``  — W at the GN/HL probe frequency.  Used by PPM
          for the second fit point.
        * ``"mpa_fit"`` — on-disk path of the MPA screening-model fit
          store (``gw.screening.compute_screening_model`` for
          ``ComputeMode.MPA``); the MPA branch reads it instead of an
          in-memory W.

        ``X_ONLY`` ignores ``W_by_role`` entirely.  Adding a new mode
        means picking the role labels it needs in
        :func:`gw.screening.screening_requests_for`, giving it a row in
        ``gw_config.MODE_SIGMA_CHANNELS``, and reading the roles here —
        no plumbing changes elsewhere.  Until it has a branch here it is
        refused by name; it is never served by the PPM one.
    e_qp_ev
        Per-(k, n) QP energies (eV) used by the QSGW build to evaluate
        Σ_c(E_m, E_n).  Required for dynamic modes; ignored for static.
    static_head_terms, head_resolver
        q→0 head plumbing; ``static_head_terms`` is None when ``do_G0`` is
        false in the config.
    quad
        Static minimax quadrature for χ₀; produced by
        ``minimax_screening.build_static_quadrature`` once per W solve.
    config, meta, mesh_xy, sym, wfn, band_slices, input_dir
        Standard driver scaffolding.
    Gij
        Optional band-space occupation projector; ``None`` builds it
        inside the static kernels from ``occupation_state``.  Supplying
        both is refused (``cohsex_sigma._resolve_Gij``).
    occupation_state
        The iteration's :class:`gw.efermi.OccupationState`.  It reaches
        BOTH halves of Σ here: the MPA branch below (µ, stamps, the
        fractional contour) and — since this commit — the static
        channels, so Σ_X / Σ_SX / V_H and the PPM invalid-pole static
        term take the same ``diag(f)`` weights Σ_c does.  ``None`` is
        the insulating default and every static channel is then
        bit-for-bit the integer ``occ > 0.5`` projector.
    wfns_transverse, bispinor_v_q_path
        Bispinor Σ^B channel (transverse-centroid ψ bundle + V^{i,j}
        tile file).  Both-or-neither; the static kernels fold Σ^B into
        ``sig_x`` and, for COHSEX, the physical ``sig_sx`` component that
        forms ``sigma_xc``.  ``None`` for scalar runs.
    photon_response
        Packed static four-current response.  Used only by
        ``bispinor_gw=full_static_cohsex``; the default bare-transverse path
        neither inspects nor constructs it.
    print_fn
        Rank-0-only print.

    Returns
    -------
    :class:`SigmaResult` populated per the mode.
    """
    from .cohsex_sigma import compute_cohsex_sigma, compute_v_h_sigma_x
    from .ppm_pipeline import compute_ppm_sigma_pipeline

    # ── THE MODE IS CHECKED BEFORE ANY KERNEL RUNS ──────────────────────
    # ``gw_jax.main`` already refused a declared-but-unbuilt mode at
    # driver entry, and this is the same refusal at the seam that would
    # otherwise absorb it.  Both exist on purpose: the entry check is what
    # saves the operator's allocation, this one is what makes the SC loop,
    # the tests and any future caller safe without having to remember the
    # entry check.  It is a dict lookup on a resolved enum, so it costs
    # nothing on the Σ path it guards.
    refuse_unimplemented_compute_mode(mode, context="compute_sigma_xc")

    # ── THE ONE ENVELOPE ROW NO DECK KEY CAN EXPRESS ─────────────────────
    # low_mem_bands's other four unsupported combinations (head_correction,
    # qp_solver, mpa_material_class, bispinor) already refused at config
    # resolution, before this function -- or anything upstream of it --
    # ever ran.  An explicit Gij is a call-time Python parameter with no
    # deck key, so it is checked here instead: this is the only seam that
    # ever sees both a resolved low_mem_bands and a live Gij operand
    # together, and it still runs before any Gij-dependent allocation.
    refuse_explicit_gij_under_low_mem_bands(config, Gij)

    # ── PPM-ONLY IS A CORRECTNESS GUARD, NOT A WIRING GAP ───────────────
    # Two independent reasons, and the second is the load-bearing one.
    #
    # (1) Wiring.  ``sigma_band_extrapolation`` is read by the GN/HL
    #     two-point PPM Σ kernel and nothing else.  Reaching MPA / COHSEX /
    #     X_ONLY with it set would produce a perfectly ordinary run whose
    #     log simply lacks the extrapolation block — the exact failure mode
    #     measurement-discipline rule 1 names, where a green A/B measured
    #     nothing because one arm silently dropped the knob.
    #
    # (2) THE MATH ITSELF IS MODE-DEPENDENT.  The extrapolation's limit
    #     point is 1/N → 0, and that limit is WRONG for a static Coulomb
    #     hole.  MEASURED 2026-08-15 against BerkeleyGW's exact static CH
    #     (the closure sum — no band sum and no extrapolation in it), Si
    #     4×4×4 SOC, 192 (k, band) states, MAE in meV:
    #
    #         nband                     60      76     100     124
    #         static COHSEX, 1/N → 0  94.9    96.6   202.8   288.2   WORSE
    #         GN-PPM,        1/N → 0 171.3    97.4    55.1    32.8   better
    #
    #     The static arm ANTI-CONVERGES — more bands determine the line
    #     better and drive it more confidently ~340 meV past the right
    #     answer — because the static CH's high-energy tail is not
    #     suppressed by a pole denominator and keeps contributing past where
    #     the 1/N law was calibrated.  So routing this at a static mode
    #     would not merely fail to log; it would return a wrong number
    #     carrying a "consistent" verdict that gets worse the more you spend
    #     on it.  Report: sandbox
    #     reports/ch_converge_band_extrapolation_2026-08-15/.
    # ── RECONCILING THE GUARD WITH A DEFAULT-ON KEY ─────────────────────
    # Before 2026-08-16 the key defaulted OFF, so "set it on a non-PPM mode"
    # was always a deliberate act and refusing was the whole answer.  The key
    # now defaults ON, and a refusal that fires on the DEFAULT would make
    # every COHSEX / MPA / X_ONLY run in the tree unrunnable — two gates
    # fighting, with the operator caught in the middle.
    #
    # So the guard splits on PROVENANCE, which is the only thing that
    # distinguishes the two situations:
    #
    #   explicitly named + NO stage can consume it  ->  REFUSE.  The operator
    #       wrote the knob down and nothing in this run will read it; silently
    #       doing nothing with it is exactly how a green A/B comes to measure
    #       nothing (measurement-discipline rule 1).
    #   defaulted, or a LATER STAGE will consume it ->  DISABLE FOR THIS
    #       STAGE, and SAY SO.  The stage is not what the key is for, but the
    #       run may still be, and killing it would refuse a run that works.
    #
    # Both branches keep the physics guard intact: no static-mode Σ is ever
    # extrapolated either way.  What changes is who gets refused.
    #
    # ── THE REFUSAL IS ABOUT THE RUN, NOT ABOUT THIS STAGE ──────────────
    # Corrected 2026-08-16 against the REAL staged-SC interface
    # (``origin/feat/staged-sc-2026-08-15``, 98289d77), which the wiring
    # branch had concluded did not exist — from an ``--all`` search in a
    # single-branch checkout, where ``--all`` covers only fetched refs.
    # See ``gw_config.sigma_stage_modes`` for the full correction.  The
    # short form: ``run_staged_self_consistency`` rewrites ``compute_mode``
    # per stage, so a per-stage DISABLE written against ``compute_mode`` was
    # already right — but a per-stage REFUSAL is not, because it kills the
    # run before the stage that would have consumed the key.  Two shipped
    # configurations it would have killed:
    #
    #   sc_stage_1_type = cohsex, sc_stage_2_type = gnppm
    #       -> dies at stage 1, one stage short of the consumer.
    #   compute_mode = mpa  (the DEFAULT ladder is GN_PPM then MPA)
    #       -> dies at stage 2, after paying for a full GN-PPM stage.
    #
    # Asking the LADDER instead makes both runnable and still refuses the
    # case the guard was written for: an explicit key on a run in which no
    # stage is a plasmon-pole model.
    if bool(config.sigma.band_extrapolation) and mode.ppm_model is None:
        explicit_switch = bool(getattr(
            config.sigma, "band_extrapolation_explicit", False))
        explicit_scheme = bool(getattr(
            config.sigma, "band_extrapolation_bracket_scheme_explicit",
            False))
        explicit = explicit_switch or explicit_scheme
        run_modes = sigma_stage_modes(config, fallback=mode)
        consumable = band_extrapolation_is_consumable(run_modes)
        ladder = " -> ".join(getattr(m, "value", str(m)) for m in run_modes)
        bracket_scheme = getattr(
            config.sigma, "band_extrapolation_bracket_scheme",
            BRACKET_SCHEME_DEFAULT)
        if explicit and not consumable:
            raise NotImplementedError(
                f"Band extrapolation was explicitly configured "
                f"(use_band_extrapolation = true; "
                f"band_extrapolation_bracket_scheme = "
                f"{bracket_scheme}), but NO "
                f"stage of this run consumes it.  This run's Σ schemes are "
                f"[{ladder}]; the "
                f"stage refusing here is compute_mode = "
                f"{getattr(mode, 'value', mode)}.  The band-convergence "
                f"extrapolation is wired into the two-point plasmon-pole Σ_c "
                f"kernel only (gn_ppm / hl_ppm), and this is a CORRECTNESS "
                f"guard rather than a wiring gap: the 1/N -> 0 limit point is "
                f"mode-dependent, and on a static Coulomb hole it overshoots "
                f"the exact answer by ~340 meV and gets WORSE with more bands "
                f"(MEASURED against BerkeleyGW's exact static CH: 94.9 meV MAE "
                f"at nband 60 rising to 288.2 at nband 124, against 171.3 "
                f"falling to 32.8 for GN-PPM).  Use compute_mode = gn_ppm, add "
                f"a gnppm stage to the sc_stage_N_type ladder, or set "
                f"use_band_extrapolation = false.  (This deck NAMES the key; "
                f"had it been left at its default the feature would have "
                f"disabled itself here with a note instead of refusing.  A "
                f"ladder containing ANY gn_ppm / hl_ppm stage also does not "
                f"refuse — the non-PPM stages in it disable themselves and the "
                f"run continues.)")
        # AUTO-DISABLED, LOUDLY.  Printed at the Σ seam every iteration
        # rather than once at startup: a staged run changes mode between
        # stages, and the fact "this stage did not extrapolate" belongs
        # beside that stage's Σ, not in a banner scrolled past an hour ago.
        if explicit:
            why = (f"this deck NAMES the key and a PPM stage in this run's "
                   f"ladder [{ladder}] will consume it — this stage is not "
                   f"that one, so it is skipped here rather than refusing "
                   f"the run")
        else:
            why = ("no deck key named it; use_band_extrapolation defaults on")
        # The JUSTIFICATION differs by stage kind and must not be recited
        # wrongly.  A static mode gets the measured static-CH anti-convergence;
        # MPA is DYNAMIC, so that measurement is not about it, and claiming it
        # were would be inventing evidence.
        if getattr(mode, "is_dynamic", False):
            because = (
                "MPA is dynamic, so the static Coulomb-hole measurement below "
                "is NOT the reason here: the reason is that the 1/N -> 0 "
                "limit has never been measured for this ansatz and its Σ_c is "
                "not built by the bracketed two-point PPM kernel, so there is "
                "no bracket axis to fit.  Extrapolating it would be an "
                "unvalidated claim, not a correction")
        else:
            because = (
                "The 1/N -> 0 limit is MODE-DEPENDENT and is wrong for a "
                "static Coulomb hole: measured against BerkeleyGW's exact "
                "static CH it ANTI-converges, 94.9 -> 288.2 meV MAE as nband "
                "goes 60 -> 124, overshooting by ~340 meV, while GN-PPM "
                "improves 171.3 -> 32.8 over the same range")
        print_fn(
            f"  Σc band extrapolation: AUTO-DISABLED for compute_mode = "
            f"{getattr(mode, 'value', mode)} ({why}).  {because}.  "
            f"This stage's Σ is the ordinary full-band sum.")
        # NOTHING IS REBOUND HERE, deliberately.  ``config.sigma.
        # band_extrapolation`` is read in exactly one place — the GN/HL-PPM
        # pipeline's ``plan_band_brackets`` call — and this branch is the one
        # where that pipeline is NOT reached.  Rewriting the config to keep it
        # cosmetically truthful would mean a ``dataclasses.replace`` of the
        # whole frozen LorraxConfig (re-running its __post_init__) to change a
        # field with no remaining reader.  The log line above is the record.

    # Static channels: sig_h (V_H) and sig_x (bare exchange) are needed
    # by every mode; sig_sx / sig_coh use W(ω=0), and WHICH MODES BUILD
    # THEM IS THE CHANNEL TABLE'S ANSWER (``gw_config.
    # MODE_SIGMA_CHANNELS``), not this branch's opinion — that is the one
    # fact the QSGW appendix writer and this dispatch have to agree on,
    # and they now read it from the same row.  Route to a separate
    # top-level entry point for the V-only path so the modes that build no
    # static screened channels never invoke the W-touching kernels, and
    # the two paths each get their own jit-cached graph.
    W_static = W_by_role.get("static", V_q)
    builds_static_screened = mode_builds_channels(
        mode, SigmaChannel.SX, SigmaChannel.COH)
    from .gw_config import BispinorGWMode, coerce_bispinor_gw_mode
    bispinor_gw = coerce_bispinor_gw_mode(getattr(
        config, "bispinor_gw", BispinorGWMode.BARE_TRANSVERSE))
    if bispinor_gw is BispinorGWMode.FULL_STATIC_COHSEX:
        if not builds_static_screened or mode is not ComputeMode.COHSEX:
            raise ValueError(
                "full_static_cohsex reached Sigma outside compute_mode=cohsex; "
                "the config/driver envelope should have refused this before "
                "screening allocation.")
        if photon_response is None:
            raise RuntimeError(
                "bispinor_gw=full_static_cohsex reached Sigma without the "
                "packed static photon response.  Refusing instead of "
                "falling back to charge-only screened COHSEX.")
        if static_head_terms is not None:
            raise ValueError(
                "full_static_cohsex received scalar static_head_terms.  Its "
                "q->0 policy already lives in the packed four-current V/W; "
                "a scalar correction would double count the charge sector "
                "and omit coupled current wings.")

        # The V-only facade remains the scalar Hartree owner, but skips its
        # historical scalar+TT exchange contraction.  X, SX, and COH are all
        # REPLACED, not augmented, by one sixteen-block photon loop over the
        # same packed V/W and canonical Green/convolution/projector services.
        from .cohsex_sigma import _resolve_Gij
        photon_Gij = _resolve_Gij(Gij, meta, mesh_xy, occupation_state)
        cohsex = compute_v_h_sigma_x(
            wfns, V_q, meta, mesh_xy,
            Gij=photon_Gij,
            static_head_terms=None,
            occupation_state=None,
            compute_bare_x=False,
        )
        from .photon_sigma import compute_static_photon_sigma
        photon_x, photon_sx, photon_coh = compute_static_photon_sigma(
            wfns_charge=wfns,
            wfns_transverse=wfns_transverse,
            Gij=photon_Gij,
            V_packed=photon_response.V_packed,
            W_packed=photon_response.W_packed,
            photon_layout=photon_response.layout,
            meta=meta,
            mesh_xy=mesh_xy,
            print_fn=print_fn,
        )
        cohsex["sig_x"] = photon_x
        cohsex["sig_sx"] = photon_sx
        cohsex["sig_coh"] = photon_coh
    elif builds_static_screened:
        cohsex = compute_cohsex_sigma(
            wfns, V_q, W_static, meta, mesh_xy,
            Gij=Gij,
            do_screened=True,
            static_head_terms=static_head_terms,
            compute_bare_x=True,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
            occupation_state=occupation_state,
        )
    else:
        cohsex = compute_v_h_sigma_x(
            wfns, V_q, meta, mesh_xy,
            Gij=Gij,
            static_head_terms=static_head_terms,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
            occupation_state=occupation_state,
        )
    sig_h = cohsex["sig_h"]
    sig_x = cohsex["sig_x"]

    # ── The V_H-source seam (single point of truth) ──────────────────────
    # ``sig_h`` above is the scalar ISDF V_q[0] quadrature.  This is the ONE
    # place the scalar term enters ``SigmaResult``, so resolving its source
    # here makes every downstream consumer consistent by construction rather
    # than each remembering the rule: the eigh operand ``sigma_total =
    # Σ_xc + V_H``, the fixed-point h₀, the SC iteration map, eqp{0,1}.dat
    # and sigma_diag.dat's VH column all read what this decides.
    #   stored/gspace → replace it with the exact FFT-grid matrix
    #   folded        → zero it (V_H is inside kin_ion's values already)
    #   isdf          → keep it
    # The ISDF quadrature runs regardless; it is cheap next to Σ AT SCALE, and
    # running it unconditionally keeps the graph shape source-independent.
    # Sized 2026-08-11 (P=4, BFC@0.85): gw_jax.isdf is 24.4% of the driver wall
    # against Σ's 28.6% at Si 4x4x4, and 27.7% against 41.8% at Si 6x6x6 — so
    # "cheap next to Σ" holds where it matters and INVERTS on the gnppm_debug
    # fixture (23.7% against 5.2%), which is a bring-up-dominated gate deck.
    source, v_h_ext, h_transverse = resolve_external_hartree(
        config, meta, band_slices, mesh_xy, wfn=wfn, sym=sym, print_fn=print_fn)
    if omit_v_h:
        # DENSITY SELF-CONSISTENCY.  V_H is supplied by the caller in the
        # DFT basis and added straight to ``kin_ion_dft``, which is also
        # DFT — so it must NOT travel through here.  Everything this
        # function returns is in the QP basis and gets rotated back by
        # ``sc_iteration``; routing V_H through that round trip would
        # rotate it twice for no reason.  Zero it and let the caller own
        # it.  ``resolve_external_hartree`` still runs above so the graph
        # shape stays source-independent.
        source, v_h_ext = "caller_dft", None
        sig_h = jnp.zeros_like(sig_h)
    if source == "folded":
        sig_h = jnp.zeros_like(sig_h)
    elif v_h_ext is not None:
        v_h_ext = jnp.asarray(v_h_ext, dtype=sig_h.dtype)
        if hartree_basis_rotation is not None:
            # QSGW: ``wfns`` is in the CURRENT QP basis, so every Σ channel
            # this function returns is too, and ``sc_iteration`` rotates the
            # lot back with ``O_DFT = U·O_QP·U†``.  The stored/gspace V_H is
            # a fixed operator in the DFT basis, so it must be rotated INTO
            # the QP basis first (``O_QP = U†·O_DFT·U``) — substituting it
            # raw would make the rotate-back return ``U·V_H·U†`` and put a
            # basis error into a ~500 eV term with no other symptom.
            #
            # U STAYS AT ``qsgw_density.band_rotation_spec``, which is what
            # ``sc_iteration`` hands over, and the rotation is
            # ``rotate_band_matrix`` — the same primitive
            # ``sc_iteration._rotate_to_dft_basis`` and
            # ``qsgw_density.rotate_bands`` use.  Only the RESULT is pinned
            # replicated: a sharded ``SigmaResult.v_h_kij_ry`` propagates
            # into the k-star select and the host readbacks downstream.
            # (The previous form gathered U replicated first — 10.7 ms
            # against 5.3 ms for an already-replicated U at nk=16/nb=128 on
            # a 2×2 mesh, job 7889424 — which is the 9.2 GB/rank object at
            # nk=144/nb=2000.)  This branch is skipped entirely under
            # density-SC (``omit_v_h`` zeroes sig_h above).
            v_h_ext = _rotate_v_h_to_qp(
                jnp.asarray(v_h_ext, dtype=sig_h.dtype),
                _place_band_rotation(hartree_basis_rotation, mesh_xy,
                                     sig_h.dtype),
                mesh=mesh_xy)
        sig_h = v_h_ext
    v_h_scalar = sig_h
    if h_transverse is not None and hartree_basis_rotation is not None:
        h_transverse = _rotate_v_h_to_qp(
            jnp.asarray(h_transverse, dtype=sig_h.dtype),
            _place_band_rotation(
                hartree_basis_rotation, mesh_xy, sig_h.dtype),
            mesh=mesh_xy)
    if h_transverse is not None:
        # ``hartree_source`` and ``omit_v_h`` own only the scalar charge
        # potential.  The exact periodic G-space current artifact is a
        # separate operator, so append it independently after the scalar
        # replacement rather than letting hartree_source erase it.  It is the
        # SSOT for BOTH BARE_TRANSVERSE and FULL_STATIC_COHSEX; no centroid
        # direct contraction or exchange q->0 head reaches this seam.
        sig_h = sig_h + h_transverse
        sig_h.block_until_ready()
    sig_sx = cohsex["sig_sx"]                    # zero placeholders for V-only path
    sig_coh = cohsex["sig_coh"]

    if mode is ComputeMode.X_ONLY:
        # sigma_sx ← sig_x so the static sigma_diag.dat writer's sigSX
        # column reports Σ_X (incl. the bispinor Σ^B fold-in) instead of
        # zeros; sigTOT = sigSX + sigCOH stays consistent.
        return SigmaResult(
            v_h_kij_ry=sig_h,
            v_h_scalar_kij_ry=v_h_scalar,
            h_transverse_kij_ry=h_transverse,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sig_x,
            sigma_sx_kij_ry=sig_x,
            sigma_coh_kij_ry=jnp.zeros_like(sig_x),
        )
    if mode is ComputeMode.COHSEX:
        sigma_xc = sig_sx + sig_coh
        return SigmaResult(
            v_h_kij_ry=sig_h,
            v_h_scalar_kij_ry=v_h_scalar,
            h_transverse_kij_ry=h_transverse,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sigma_xc,
            sigma_sx_kij_ry=sig_sx,
            sigma_coh_kij_ry=sig_coh,
        )

    # Dynamic modes (MPA + the PPM pair) all evaluate the QSGW Σ_c at QP
    # energies — one check above both branches, one message.
    if e_qp_ev is None:
        raise ValueError(
            f"compute_sigma_xc: dynamic mode {mode!r} requires e_qp_ev "
            "(QP energies for the QSGW Σ_c evaluation).")

    if mode is ComputeMode.MPA:
        from file_io import mpa_store
        from .head_correction import compute_complex_pole_head_sigma_diag
        from .mpa.sigma import compute_sigma_c_mpa_omega_grid
        from .mpa.sigma_windows import CROSSING_NODE_FLOOR
        from .efermi import resolve_sigma_efermi_ry
        from .ppm_windows import sigma_regularization_for_config

        try:
            fit_path = W_by_role["mpa_fit"]
        except KeyError as exc:
            raise KeyError("MPA Sigma requires W_by_role['mpa_fit']") from exc
        # ── DECK KEYS THIS BRANCH HONORS, NAMED ─────────────────────────
        # Both keys below are parsed and validated by gw_config and were
        # then IGNORED here: MPA hard-coded ``wfn.efermi`` and always
        # emitted the sharded cube, while the PPM branch honored both.  A
        # parsed-but-ignored key is a defect (TASTE 13), and it became a
        # live one the moment UNIMPLEMENTED_MODES stopped holding MPA back.
        #
        # sigma_omega_layout: the MPA executor's accumulator is born
        # P(None,None,'x','y') and there is no replicated plan for it --
        # which is what the metal-only refusal in
        # gw_config._validate_occupation_smearing already SAYS ("the MPA
        # Sigma emits the mesh-sharded omega cube only").  That is a fact
        # about MPA, not about metals, so the refusal is generalised here
        # rather than left to fire on one material class.  Refusing (not
        # gathering) is the standing ruling: the sharded layout exists
        # precisely to elide the P-independent full-cube gather, so
        # "replicated" would be an allgather sold as a fallback
        # (decisions.md 2026-08-05).
        #
        # BUT REFUSE ONLY A DECK THAT SAID IT.  ``sigma_omega_layout``'s
        # DEFAULT is ``replicated``, so a bare refusal on the resolved
        # value fires on every insulating MPA deck that never mentioned
        # the key -- a flag day for decks that are not wrong about
        # anything.  TASTE 13 draws exactly this line: an off-dial may
        # refuse, a typo never does, and a value nobody typed is not a
        # request.  The parser records the raw keys it saw
        # (``GWConfig.raw_input_keys``), so the question is asked of that
        # -- the same idiom ``restart_q_storage.deck_named_the_key`` uses,
        # including its conservative answer when the record is absent.
        # A deck that DID name ``replicated`` still refuses: honouring it
        # would mean gathering the full cube on every rank, which is the
        # P-independent collective the sharded layout exists to elide
        # (decisions.md 2026-08-05, refuse rather than gather).
        _layout = str(config.sigma.omega_layout)
        if _layout != "sharded":
            _seen = getattr(config, "raw_input_keys", None)
            try:
                _named = _seen is not None and "sigma_omega_layout" in _seen
            except TypeError:                            # pragma: no cover
                _named = False
            if _named:
                raise ValueError(
                    f"compute_mode = mpa requires sigma_omega_layout = "
                    f"sharded, and this deck NAMES {_layout!r}.  The MPA "
                    f"Sigma executor accumulates Sigma_c(w,k,m,n) directly "
                    f"at P(None,None,'x','y') on the existing mesh and has "
                    f"no replicated plan; honoring 'replicated' would mean "
                    f"gathering the full cube on every rank, which is the "
                    f"P-independent collective the sharded layout exists "
                    f"to elide.  Set sigma_omega_layout = sharded, or drop "
                    f"the key (its default resolves to sharded under mpa).")
            print_fn(
                f"  sigma_omega_layout: resolved to 'sharded' for "
                f"compute_mode = mpa (deck did not name the key; its "
                f"default {_layout!r} has no MPA plan -- the executor's "
                f"accumulator is born P(None,None,'x','y')).  A deck that "
                f"NAMES 'replicated' under mpa is refused rather than "
                f"gathered.")
        # The effective Sigma broadening, from the SAME resolver the PPM
        # driver uses.  MPA used to take ``regularization_ev`` raw while
        # GN-PPM silently raised it to a window-dependent conditioning
        # floor -- 1.90x apart on the sodium 48b deck, 5.7x on a +/-15 eV
        # window -- so every cross-ansatz comparison was confounded and
        # neither output said what xi it ran at.
        _xi = sigma_regularization_for_config(config)
        print_fn(_xi.describe())
        if config.mpa.material_class == "metal":
            # Metal deck-key consistency is refused at config parse
            # (_validate_occupation_smearing); here the run-level facts:
            # the one-occupation-state rule, and head/body provenance.
            if occupation_state is None:
                raise ValueError(
                    "MPA Sigma under mpa_material_class = metal requires "
                    "the iteration's occupation_state (fixed-N MP1 solve); "
                    "got None. The QSGW driver passes it; a direct caller "
                    "must construct one from the current spectrum.")
            # No stamp assert here: this is a SAME-RUN site (the fit store
            # was written by this run's screening step), and W4 rules that
            # stamps are asserted at REUSE sites only — a same-run
            # write-then-read cannot detect the cross-iteration leak it
            # would claim to guard (claim 0194: the assert here was
            # unsatisfiable while no writer path carried the state).
            # assert_occupation_stamps remains the cross-run reuse gate.
        # fermi_reference: resolved by the one owned resolver, which also
        # returns the provenance string the sigma_mnk.h5 stamp needs.
        # AFTER the metal block on purpose: that block owns the more
        # specific "metal needs an occupation_state" message, and the
        # resolver's own refusal for the same case would otherwise pre-empt
        # it with a less situated one.
        sigma_efermi_ry, sigma_efermi_provenance = resolve_sigma_efermi_ry(
            config.sigma.fermi_reference,
            occupation_state=occupation_state, wfn=wfn)
        body = compute_sigma_c_mpa_omega_grid(
            wfns, fit_path, meta, mesh_xy,
            omega_grid_ry=config.omega_grid_ry,
            efermi_ry=sigma_efermi_ry,
            occupation_state=occupation_state,
            regularization_width_ry=_xi.resolved_ry,
            edge_factor=float(config.sigma.window_edge_factor),
            target_error=float(config.mpa.sigma_sector_target_error),
            crossing_target_error=float(
                config.mpa.sigma_crossing_target_error),
            max_rank=int(config.mpa.sigma_max_nodes),
            crossing_max_nodes=max(
                CROSSING_NODE_FLOOR, int(config.mpa.sigma_max_nodes)),
            omega_cluster_gap_ry=float(
                config.mpa.sigma_omega_cluster_gap_ry),
            occupation_window_threshold=float(
                config.mpa.occupation_window_threshold),
            pole_batch_size=int(config.mpa.pole_batch_size),
            # PROVENANCE ASSERT AT LOAD: these poles were fitted to a W
            # this run's screening_diagrams either did or did not produce,
            # and the two are indistinguishable in the bytes.
            expected_screening_diagrams=config.screening.diagrams,
            print_fn=print_fn)
        head = mpa_store.read_head_fit_collective(
            fit_path, mesh_xy=mesh_xy, to_unit="Ry")
        # One occupation state per iteration: head fit vs body (skips when
        # occupation_state is None; refuses an unstamped head under metal).
        from .mpa.sigma import assert_head_body_occupation_match
        assert_head_body_occupation_match(
            head.get("occupation_stamps") or {}, occupation_state)
        if iteration_head is None:
            sigma_bands = wfns.slices.sigma
            head_enk = np.asarray(wfns.enk[:, sigma_bands])
            head_occ = np.asarray(wfns.occ[:, sigma_bands])
            head_efermi = sigma_efermi_ry
        else:
            head_enk = np.asarray(iteration_head.sigma_energies_ry)
            head_occ = np.asarray(iteration_head.sigma_occupations)
            head_efermi = float(iteration_head.efermi_ry)
        head_diag = compute_complex_pole_head_sigma_diag(
            omega_grid_ry=np.asarray(config.omega_grid_ry),
            enk_ry=head_enk,
            efermi_ry=head_efermi,
            occupations=head_occ,
            poles_ry=head["Omega_p"], residues_ry=head["B_p"],
            cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot))
        return finalize_dynamic_sigma(
            body.sigma_c_kij, head_diag,
            sig_x=sig_x, sig_h=sig_h,
            v_h_scalar=v_h_scalar, h_transverse=h_transverse,
            e_qp_ev=e_qp_ev,
            config=config, meta=meta, mesh_xy=mesh_xy,
            sym=sym, wfn=wfn, band_slices=band_slices,
            input_dir=input_dir,
            write_sigma_omega_h5=write_sigma_omega_h5,
            print_fn=print_fn,
            # The MPA grid was built against this reference (one per
            # iteration); the finalizer must read it back against the same
            # one, and STAMP which one it was -- the `efermi_ry is None`
            # proxy the finalizer falls back to would label every explicit
            # reference "fixed-N mu", so a midgap MPA run would be written
            # into sigma_mnk.h5 as a metal's chemical potential.
            efermi_ry=sigma_efermi_ry,
            efermi_provenance=sigma_efermi_provenance)

    # ── THE EXHAUSTIVENESS SEAM ─────────────────────────────────────────
    # What follows is the two-point plasmon-pole pipeline, and until this
    # guard it was reached by ELSE: anything that was not X_ONLY and not
    # COHSEX ran it.  That is fine while the enum's only remaining members
    # ARE the two PPM fits, and it silently mis-runs the first member that
    # is not — a multipole run would have taken a GN fit of two W samples
    # and reported it as Σ_c(ω) with no stage able to tell.  So the pole
    # model is now asked for by name: ``ppm_model`` is 'gn' or 'hl' for
    # exactly the two PPM modes and None for every other member, present
    # or future.
    if mode.ppm_model is None:
        raise NotImplementedError(
            f"compute_sigma_xc: compute_mode = "
            f"{getattr(mode, 'value', mode)} reaches the Σ dispatch with no "
            f"Σ kernel of its own.  The static channels are handled above "
            f"(X_ONLY, COHSEX) and the dynamic branch below is the two-point "
            f"plasmon-pole pipeline, which this mode is not; it is refused "
            f"here rather than run under another ansatz's name.  A new mode "
            f"needs its own branch here, a row in "
            f"gw_config.MODE_SIGMA_CHANNELS, and a case in "
            f"gw.screening.screening_requests_for.")

    # Dynamic PPM modes: need W_static + W_probe.
    if "probe" not in W_by_role:
        raise KeyError(
            f"compute_sigma_xc: PPM mode {mode!r} requires "
            f"W_by_role['probe'] (set by screening_requests_for).")

    ppm_outputs = compute_ppm_sigma_pipeline(
        wfns=wfns,
        V_q=V_q,
        W_static_q=W_static, W_probe_q=W_by_role["probe"],
        quad=quad,
        config=config, meta=meta, mesh_xy=mesh_xy,
        head_resolver=head_resolver,
        band_slices=band_slices, wfn=wfn, sym=sym,
        iteration_head=iteration_head,
        occupation_state=occupation_state,
        print_fn=print_fn,
    )

    return finalize_dynamic_sigma(
        ppm_outputs.sigma_c_body_omega,
        ppm_outputs.head_sigma_diag_w_kn_ry,
        sig_x=sig_x, sig_h=sig_h,
        v_h_scalar=v_h_scalar, h_transverse=h_transverse,
        e_qp_ev=e_qp_ev,
        config=config, meta=meta, mesh_xy=mesh_xy,
        sym=sym, wfn=wfn, band_slices=band_slices,
        input_dir=input_dir,
        write_sigma_omega_h5=write_sigma_omega_h5,
        band_extrapolation=ppm_outputs.band_extrapolation,
        sigma_c_body_omega_unextrap=(
            ppm_outputs.sigma_c_body_omega_unextrap),
        print_fn=print_fn,
    )


__all__ = [
    "SigmaResult",
    "finalize_dynamic_sigma",
    "compute_sigma_xc",
    "ROTATED_TO_DFT_FIELDS",
    "SIGMA_BASIS_FIELDS",
    "DFT_BASIS_FIELDS",
    "BASIS_FREE_FIELDS",
]
