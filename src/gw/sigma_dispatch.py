"""Mode-orthogonal Σ_xc dispatch.

A single entry point :func:`compute_sigma_xc` that the QSGW iteration
map calls regardless of compute mode (X_ONLY, COHSEX, GN_PPM, HL_PPM).
The dispatch decides which Σ kernel runs internally; the iteration map
sees one signature and one result type.

Returned :class:`SigmaResult` always contains ``v_h_kij_ry``,
``sigma_x_kij_ry``, and a single ``sigma_xc_kij_ry`` representing the
total exchange-correlation contribution to ``H_QP = kin_ion + V_H +
Σ_xc``.  PPM-mode-only diagnostics (full ω-grid Σ_c, on-shell diagonals,
head decomposition) live as optional fields and are populated only when
the mode produces them.

This module owns *no compute* of its own — every kernel lives under
``cohsex_sigma`` (static channels), ``ppm_pipeline`` (dynamic Σ_c) or
``qsgw_utils`` (the QSGW Hermitisation).  It only orchestrates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.units import RYD_TO_EV
from .gw_config import ComputeMode


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
                                          head).  PPM modes: Σ_x + Σ_c^QSGW.

    Static-mode-only (None in PPM)
    ------------------------------
    sigma_sx_kij_ry      : (nk, nb, nb)   Σ_SX with head
    sigma_coh_kij_ry     : (nk, nb, nb)   Σ_COH with head

    PPM-only (None in static)
    -------------------------
    sigma_c_omega_kij_ry      : (nω, nk, nb, nb), sharded P(None,None,'x','y')
                                Full ω-grid Σ_c (post-head); drives eqp1
                                Z-factor central difference.
    sigma_c_at_dft_diag_ev    : (nk, nb)  diag(Σ_c) at E_DFT (eV).
    omega_dft_rel_ev          : (nk, nb)  E_DFT − E_F (eV).
    omega_grid_ev             : (nω,)     ω-grid in eV.
    omega_grid_ry             : (nω,)     ω-grid in Ry.
    head_sigma_diag_w_kn_ry   : (nω, nk, nb)  PPM analytic head diagonal.
    sigma_omega_h5_path       : str       on-disk Σ_c(ω) HDF5 path.
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
# H₀'s Hartree term: resolve the source once, cache the array
# ---------------------------------------------------------------------------

#: (kin_ion path, b0, b3, resolved source) → (source, V_H (nk,nb,nb) Ry | None).
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
           source, id(mesh_xy))
    if key in _hartree_cache:
        return _hartree_cache[key]

    v_h = None
    if source == "stored":
        v_h = load_hartree_submatrix(
            path, band_slices.b0, band_slices.b3,
            mesh=mesh_xy, backend=config.backend.slab_io)
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
        from gw.kin_ion_io import compute_hartree_matrix, replicate_to_mesh
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
        :func:`gw.screening.screening_requests_for` and reading them
        here — no plumbing changes elsewhere.
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
    from .qsgw_utils import build_qsgw_sigma_xc

    # Static channels: sig_h (V_H) and sig_x (bare exchange) are needed
    # by every mode; sig_sx / sig_coh use W(ω=0) and only matter for
    # COHSEX.  Route to a separate top-level entry point for the
    # V-only path so PPM / X_ONLY modes never invoke the W-touching
    # kernels and the two paths each get their own jit-cached graph.
    W_static = W_by_role.get("static", V_q)
    backend = config.backend.slab_io
    if mode is ComputeMode.COHSEX:
        cohsex = compute_cohsex_sigma(
            wfns, V_q, W_static, meta, mesh_xy,
            Gij=Gij,
            do_screened=True,
            static_head_terms=static_head_terms,
            compute_bare_x=True,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
            backend=backend,
        )
    else:
        cohsex = compute_v_h_sigma_x(
            wfns, V_q, meta, mesh_xy,
            Gij=Gij,
            static_head_terms=static_head_terms,
            wfns_transverse=wfns_transverse,
            bispinor_v_q_path=bispinor_v_q_path,
            backend=backend,
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
    # The ISDF quadrature runs regardless; it is cheap next to Σ, and
    # running it unconditionally keeps the graph shape source-independent.
    source, v_h_ext = resolve_external_hartree(
        config, meta, band_slices, mesh_xy, wfn=wfn, sym=sym, print_fn=print_fn)
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
            U = jnp.asarray(hartree_basis_rotation, dtype=sig_h.dtype)
            v_h_ext = jnp.einsum('kpm,kpq,kqn->kmn',
                                 jnp.conj(U), v_h_ext, U, optimize=True)
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
        sig_x=sig_x, sig_h=sig_h,
        quad=quad, e_ref=e_ref,
        config=config, meta=meta, mesh_xy=mesh_xy,
        head_resolver=head_resolver,
        band_slices=band_slices, wfn=wfn, sym=sym,
        input_dir=input_dir,
        write_sigma_omega_h5=write_sigma_omega_h5,
        print_fn=print_fn,
    )

    if ppm_outputs.sigma_c_omega is None:
        # Streamed Σ_c (kij_stream): no in-memory ω-tensor → no QSGW
        # build.  The eigh family degrades to the static-COHSEX stand-in
        # (same behavior the pre-unification driver had); eqp0/eqp1 stay
        # correct — they are written downstream from the at-DFT
        # diagnostics, which the streamed h5 path fills in.  Config
        # validation already rejects kij_stream × fixed_point /
        # self_consistent, so this branch is one-shot-only.
        statics = compute_cohsex_sigma(
            wfns, V_q, W_static, meta, mesh_xy,
            Gij=Gij,
            do_screened=True,
            static_head_terms=static_head_terms,
            compute_bare_x=False,
        )
        return SigmaResult(
            v_h_kij_ry=sig_h,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=statics["sig_sx"] + statics["sig_coh"],
            sigma_sx_kij_ry=statics["sig_sx"],
            sigma_coh_kij_ry=statics["sig_coh"],
            sigma_c_at_dft_diag_ev=ppm_outputs.sigma_c_at_dft_ev,
            omega_dft_rel_ev=ppm_outputs.omega_dft_rel_ev,
            omega_grid_ev=config.omega_grid_ev,
            omega_grid_ry=config.omega_grid_ry,
            head_sigma_diag_w_kn_ry=ppm_outputs.head_sigma_diag_w_kn_ry,
            sigma_omega_h5_path=ppm_outputs.sigma_omega_h5_path,
            efermi_dft_ev=ppm_outputs.efermi_dft_ev,
        )

    # QSGW Σ_xc^QSGW evaluated at e_qp_ev.  Static Σ_x is added inside
    # the kernel, so the result already includes Σ_x.  The E_F reference
    # is the LORRAX-canonical midgap (``wfn.efermi`` — the same value
    # the PPM pipeline used for ``omega_dft_rel_ev``), so calling this
    # with ``e_qp_ev = E_DFT`` evaluates at exactly the pipeline's
    # at-DFT frequencies (textbook G0W0 / SC-iteration-1 equivalence).
    omega_grid_ev = np.asarray(
        config.omega_grid_ev, dtype=np.float64)
    efermi_ry = float(wfn.efermi)
    e_qp_rel_ev = np.asarray(e_qp_ev, dtype=np.float64) - efermi_ry * RYD_TO_EV
    # Process-local replication (plain ``device_put`` of a host/uncommitted
    # array onto a multi-process sharding fires JAX's hidden ``assert_equal``
    # all-gather — scorecard AA.1).  ``sig_x`` is replicated post-Σ output,
    # identical on every rank; ``LORRAX_CHECK_REPLICA=1`` re-arms the check.
    sig_x_rep = device_put_process_local(
        sig_x, NamedSharding(mesh_xy, P(None, None, None)))
    sigma_xc_qsgw, qsgw_diag = build_qsgw_sigma_xc(
        ppm_outputs.sigma_c_omega, sig_x_rep,
        omega_grid_ev, e_qp_rel_ev, mesh_xy,
    )
    print_fn(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
             f"({100*qsgw_diag['frac_clipped']:.1f}%)")

    return SigmaResult(
        v_h_kij_ry=sig_h,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_qsgw,
        sigma_c_omega_kij_ry=ppm_outputs.sigma_c_omega,
        sigma_c_at_dft_diag_ev=ppm_outputs.sigma_c_at_dft_ev,
        omega_dft_rel_ev=ppm_outputs.omega_dft_rel_ev,
        omega_grid_ev=config.omega_grid_ev,
        omega_grid_ry=config.omega_grid_ry,
        head_sigma_diag_w_kn_ry=ppm_outputs.head_sigma_diag_w_kn_ry,
        sigma_omega_h5_path=ppm_outputs.sigma_omega_h5_path,
        efermi_dft_ev=ppm_outputs.efermi_dft_ev,
    )


__all__ = ["SigmaResult", "compute_sigma_xc"]
