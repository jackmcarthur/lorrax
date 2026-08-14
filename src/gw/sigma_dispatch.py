"""Mode-orthogonal Σ_xc dispatch.

A single entry point :func:`compute_sigma_xc` that the QSGW iteration
map calls regardless of compute mode (X_ONLY, COHSEX, GN_PPM, HL_PPM).
The dispatch decides which Σ kernel runs internally; the iteration map
sees one signature and one result type.

The dispatch is EXHAUSTIVE over ``gw_config.ComputeMode``: a mode with
no kernel here is refused by name, not absorbed by the last branch.  MPA
is that case today.

Returned :class:`SigmaResult` always contains ``v_h_kij_ry``,
``sigma_x_kij_ry``, and a single ``sigma_xc_kij_ry`` representing the
total exchange-correlation contribution to ``H_QP = kin_ion + V_H +
Σ_xc``.  Dynamic-mode-only diagnostics (full ω-grid Σ_c, on-shell diagonals,
head decomposition) live as optional fields and are populated only when
the mode produces them.

Every band-indexed field comes back in the basis of the ``wfns`` bundle
this module was handed; the four field tuples beside the dataclass say
which of them a consumer sees in which basis.

This module owns *no compute* of its own — every kernel lives under
``cohsex_sigma`` (static channels), ``ppm_pipeline`` (dynamic Σ_c) or
``qsgw_utils`` (the QSGW Hermitisation).  It only orchestrates.
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
    ComputeMode, SigmaChannel, mode_builds_channels,
    refuse_unimplemented_compute_mode)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SigmaResult:
    """Outputs of one full Σ pipeline call.

    Always populated
    ----------------
    v_h_kij_ry           : (nk, nb, nb)   Hartree (replicated)
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
    sigma_sx_kij_ry: jax.Array | None = None
    sigma_coh_kij_ry: jax.Array | None = None
    sigma_c_omega_kij_ry: jax.Array | None = None
    sigma_c_at_dft_diag_ev: np.ndarray | None = None
    omega_dft_rel_ev: np.ndarray | None = None
    omega_grid_ev: np.ndarray | None = None
    omega_grid_ry: np.ndarray | None = None
    head_sigma_diag_w_kn_ry: np.ndarray | None = None
    sigma_omega_h5_path: str | None = None
    efermi_dft_ev: float | None = None


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
    "sigma_x_kij_ry",
    "sigma_xc_kij_ry",
    "sigma_sx_kij_ry",
    "sigma_coh_kij_ry",
)

#: Left in the Σ compute basis on purpose — do NOT rotate these.
#: ``sigma_c_omega_kij_ry`` is the operand of the QSGW ansatz
#: ``Σ_ij^QSGW = ½[Σ_ij(E_i) + Σ_ij(E_j)]ʰ`` (``qsgw_utils.py``:402),
#: whose band indices must label the states whose energies E_i, E_j it
#: is evaluated at, so the construction is only itself in that basis;
#: it is also the (nω, nk, nb, nb) sharded tensor and the contents of
#: sigma_mnk.h5.  The other two are already band DIAGONALS, on which a
#: basis rotation does not act element-wise.
SIGMA_BASIS_FIELDS = (
    "sigma_c_omega_kij_ry",
    "sigma_c_at_dft_diag_ev",
    "head_sigma_diag_w_kn_ry",
)

#: Band-indexed but read from the WFN file, hence DFT basis on every
#: path: ``omega_dft_rel_ev`` is E_DFT − E_F (``ppm_pipeline.py``:219-223,
#: from ``get_enk_bandrange``).  TRAP: under self-consistency it labels
#: bands by the DFT index while ``sigma_c_at_dft_diag_ev`` — the
#: interpolation it drives, ``ppm_pipeline.py``:236 — labels them by the
#: QP index.
DFT_BASIS_FIELDS = (
    "omega_dft_rel_ev",
)

#: No band index; basis-independent.
BASIS_FREE_FIELDS = (
    "omega_grid_ev",
    "omega_grid_ry",
    "sigma_omega_h5_path",
    "efermi_dft_ev",
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
    """``(source, V_H_kij_ry | None)`` for this run's ``hartree_source``.

    ``source`` is one of ``'stored' | 'folded' | 'isdf' | 'gspace'``.
    The array is returned only for ``stored`` / ``gspace``; ``folded``
    means "V_H is inside kin_ion's values, add nothing", and ``isdf``
    means "keep the ISDF quadrature".
    """
    from file_io.kin_ion import resolve_hartree_source, load_hartree_submatrix

    path = config.paths.kin_ion_file
    requested = getattr(config, "hartree_source", "auto")
    source = resolve_hartree_source(path, requested, print_fn=print_fn)
    key = (os.path.abspath(path), int(band_slices.b0), int(band_slices.b3),
           source, _mesh_cache_key(mesh_xy))
    if key in _hartree_cache:
        return _hartree_cache[key]

    v_h = None
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
        v_h_np = compute_hartree_matrix(
            wfn, sym, meta,
            truncation_2d=(int(config.sys_dim) == 2),
            nb=int(band_slices.b3), mesh=mesh_xy, print_fn=print_fn)
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
    elif source == "folded":
        print_fn("  V_H: LEGACY folded kin_ion.h5 — V_H is inside its values; "
                 "the ISDF sig_h is suppressed to avoid double counting.")
    else:
        print_fn("  V_H: ISDF V_q[0] quadrature (hartree_source=isdf); H0 "
                 "therefore depends on the centroid count.")

    _hartree_cache[key] = (source, v_h)
    return source, v_h


# ---------------------------------------------------------------------------
# Dynamic-Sigma finalization (shared by every frequency ansatz)
# ---------------------------------------------------------------------------

def finalize_dynamic_sigma(
    sigma_c_body_omega: jax.Array,
    head_sigma_diag_w_kn_ry: np.ndarray | None,
    *,
    sig_x: jax.Array,
    sig_h: jax.Array,
    e_qp_ev: np.ndarray,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    band_slices,
    input_dir: str,
    write_sigma_omega_h5: bool = True,
    print_fn: Callable = print,
) -> SigmaResult:
    """Finalize one dynamic Sigma ansatz without knowing its pole model.

    The ansatz supplies only its body cube and band-diagonal q->0 head.
    This seam performs the common head injection, interpolation at DFT
    energies, canonical file write and QSGW build, then returns the uniform
    :class:`SigmaResult`.  The fixed-point solve and optional scissor remain
    in :func:`gw.qsgw_utils.solve_qp`, which consumes the retained omega cube;
    there is still one owner of that energy-update policy.
    """
    import common.timing as timing
    from .dynamic_sigma import (
        add_head_sigma_diag,
        eval_sigma_c_at_dft_energies,
        sigma_omega_output_path,
        write_sigma_omega,
    )
    from .qsgw_utils import build_qsgw_sigma_xc

    with timing.section("gw_jax.dynamic_sigma_finalize"):
        sigma_c_omega = add_head_sigma_diag(
            sigma_c_body_omega, head_sigma_diag_w_kn_ry)

        (sigma_c_at_dft_ev,
         omega_dft_rel_ev,
         efermi_dft_ev) = eval_sigma_c_at_dft_energies(
            sigma_c_omega,
            config=config,
            band_slices=band_slices, wfn=wfn, sym=sym, meta=meta,
            mesh_xy=mesh_xy, print_fn=print_fn,
        )

        if write_sigma_omega_h5:
            sigma_omega_h5_path = write_sigma_omega(
                sigma_c_omega,
                sig_x=sig_x, sig_h=sig_h,
                config=config, input_dir=input_dir,
                meta=meta, mesh_xy=mesh_xy,
                sym=sym, print_fn=print_fn,
            )
        else:
            sigma_omega_h5_path = sigma_omega_output_path(config, input_dir)

        # Static Sigma_x is added in the QSGW kernel.  E_F is the same WFN
        # reference used above for the DFT-frequency interpolation.
        omega_grid_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
        e_qp_rel_ev = (
            np.asarray(e_qp_ev, dtype=np.float64)
            - float(wfn.efermi) * RYD_TO_EV)
        sig_x_rep = device_put_process_local(
            sig_x, NamedSharding(mesh_xy, P(None, None, None)))
        sigma_xc_qsgw, qsgw_diag = build_qsgw_sigma_xc(
            sigma_c_omega, sig_x_rep,
            omega_grid_ev, e_qp_rel_ev, mesh_xy,
        )
        print_fn(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
                 f"({100*qsgw_diag['frac_clipped']:.1f}%)")

        # Only append when this call created the base file.  SC iterations
        # pass False and append once, in the cube's own basis, at convergence.
        if write_sigma_omega_h5:
            from .qsgw_utils import write_qsgw_sigma_cube
            write_qsgw_sigma_cube(
                sigma_omega_h5_path, sigma_xc_qsgw,
                config=config, print_fn=print_fn)

    return SigmaResult(
        v_h_kij_ry=sig_h,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_qsgw,
        sigma_c_omega_kij_ry=sigma_c_omega,
        sigma_c_at_dft_diag_ev=sigma_c_at_dft_ev,
        omega_dft_rel_ev=omega_dft_rel_ev,
        omega_grid_ev=config.omega_grid_ev,
        omega_grid_ry=config.omega_grid_ry,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
        sigma_omega_h5_path=sigma_omega_h5_path,
        efermi_dft_ev=efermi_dft_ev,
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
    e_ref: float,
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
    write_sigma_omega_h5: bool = True,
    hartree_basis_rotation: jax.Array | None = None,
    omit_v_h: bool = False,
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

        ``X_ONLY`` ignores ``W_by_role`` entirely.  Adding a new mode
        means picking the role labels it needs in
        :func:`gw.screening.screening_requests_for`, giving it a row in
        ``gw_config.MODE_SIGMA_CHANNELS``, and reading the roles here —
        no plumbing changes elsewhere.  Until it has a branch here it is
        refused by name; it is never served by the PPM one.
    e_qp_ev
        Per-(k, n) QP energies (eV) used by the PPM QSGW build to evaluate
        Σ_c(E_m, E_n).  Required for PPM modes; ignored for static.
    static_head_terms, head_resolver
        q→0 head plumbing; ``static_head_terms`` is None when ``do_G0`` is
        false in the config.
    quad, e_ref
        Static minimax quadrature for χ₀; produced by
        ``minimax_screening.build_static_quadrature`` once per W solve.
    config, meta, mesh_xy, sym, wfn, band_slices, input_dir
        Standard driver scaffolding.
    Gij
        Optional band-space occupation projector; ``None`` builds the
        default DFT-occ projector inside the static kernels.
    wfns_transverse, bispinor_v_q_path
        Bispinor Σ^B channel (transverse-centroid ψ bundle + V^{i,j}
        tile file).  Both-or-neither; Σ^B is folded into ``sig_x`` by
        the static kernels.  ``None`` for scalar runs.
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
    if builds_static_screened:
        cohsex = compute_cohsex_sigma(
            wfns, V_q, W_static, meta, mesh_xy,
            Gij=Gij,
            do_screened=True,
            static_head_terms=static_head_terms,
            compute_bare_x=True,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
        )
    else:
        cohsex = compute_v_h_sigma_x(
            wfns, V_q, meta, mesh_xy,
            Gij=Gij,
            static_head_terms=static_head_terms,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
        )
    sig_h = cohsex["sig_h"]
    sig_x = cohsex["sig_x"]

    # ── The V_H-source seam (single point of truth) ──────────────────────
    # ``sig_h`` above is the ISDF V_q[0] quadrature.  This is the ONE
    # place it enters ``SigmaResult``, so resolving the source here makes
    # every downstream consumer consistent by construction rather than by
    # each remembering the rule: the eigh operand ``sigma_total =
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
    source, v_h_ext = resolve_external_hartree(
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
    sig_sx = cohsex["sig_sx"]                    # zero placeholders for V-only path
    sig_coh = cohsex["sig_coh"]

    if mode is ComputeMode.X_ONLY:
        # sigma_sx ← sig_x so the static sigma_diag.dat writer's sigSX
        # column reports Σ_X (incl. the bispinor Σ^B fold-in) instead of
        # zeros; sigTOT = sigSX + sigCOH stays consistent.
        return SigmaResult(
            v_h_kij_ry=sig_h,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sig_x,
            sigma_sx_kij_ry=sig_x,
            sigma_coh_kij_ry=jnp.zeros_like(sig_x),
        )
    if mode is ComputeMode.COHSEX:
        sigma_xc = sig_sx + sig_coh
        return SigmaResult(
            v_h_kij_ry=sig_h,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sigma_xc,
            sigma_sx_kij_ry=sig_sx,
            sigma_coh_kij_ry=sig_coh,
        )

    if mode is ComputeMode.MPA:
        from file_io import mpa_store
        from .head_correction import compute_complex_pole_head_sigma_diag
        from .mpa.sigma import compute_sigma_c_mpa_omega_grid

        if e_qp_ev is None:
            raise ValueError("MPA Sigma requires QP evaluation energies")
        try:
            fit_path = W_by_role["mpa_fit"]
        except KeyError as exc:
            raise KeyError("MPA Sigma requires W_by_role['mpa_fit']") from exc
        quadrature = config.sigma_quadrature_config
        body = compute_sigma_c_mpa_omega_grid(
            wfns, fit_path, meta, mesh_xy,
            omega_grid_ry=config.omega_grid_ry,
            efermi_ry=float(wfn.efermi),
            regularization_width_ry=(
                float(config.sigma.regularization_ev) / RYD_TO_EV),
            edge_factor=float(config.sigma.window_edge_factor),
            target_error=float(quadrature.target_error),
            max_rank=int(quadrature.max_nodes),
            crossing_max_nodes=int(quadrature.crossing_max_nodes),
            pole_batch_size=int(config.mpa.pole_batch_size),
            print_fn=print_fn)
        head = mpa_store.read_head_fit(fit_path, to_unit="Ry")
        sigma_bands = wfns.slices.sigma
        head_diag = compute_complex_pole_head_sigma_diag(
            omega_grid_ry=np.asarray(config.omega_grid_ry),
            enk_ry=np.asarray(wfns.enk[:, sigma_bands]),
            efermi_ry=float(wfn.efermi),
            occupations=np.asarray(wfns.occ[:, sigma_bands]),
            poles_ry=head["Omega_p"], residues_ry=head["B_p"],
            cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot))
        return finalize_dynamic_sigma(
            body.sigma_c_kij, head_diag,
            sig_x=sig_x, sig_h=sig_h, e_qp_ev=e_qp_ev,
            config=config, meta=meta, mesh_xy=mesh_xy,
            sym=sym, wfn=wfn, band_slices=band_slices,
            input_dir=input_dir,
            write_sigma_omega_h5=write_sigma_omega_h5,
            print_fn=print_fn)

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
    if e_qp_ev is None:
        raise ValueError(
            f"compute_sigma_xc: PPM mode {mode!r} requires e_qp_ev "
            "(QP energies for the QSGW Σ_c evaluation).")
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
        print_fn=print_fn,
    )

    return finalize_dynamic_sigma(
        ppm_outputs.sigma_c_body_omega,
        ppm_outputs.head_sigma_diag_w_kn_ry,
        sig_x=sig_x, sig_h=sig_h, e_qp_ev=e_qp_ev,
        config=config, meta=meta, mesh_xy=mesh_xy,
        sym=sym, wfn=wfn, band_slices=band_slices,
        input_dir=input_dir,
        write_sigma_omega_h5=write_sigma_omega_h5,
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
