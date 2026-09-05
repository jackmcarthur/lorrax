"""Exciton bandstructure E_S(Q) along a high-symmetry Q path (TDA).

Finite-momentum TDA BSE ``H_Q = D_Q + V_Q − W`` in the pair basis
``|v k, c k+Q⟩`` (LORRAX convention: electron leg shifted by +Q), solved at
every Q of a user-supplied high-symmetry path with ONE compiled engine:

  * **Q path** — read from the SAME ``K_POINTS crystal_b`` block (QE-style,
    labels via ``#``) that the htransform bandstructure driver consumes;
    parsing / node labels / path-distance accumulation reuse
    ``bandstructure.htransform`` (``generate_kpath_from_qe_segments`` /
    ``initialize_kpath``).  One path format in the package.
  * **ψ_c(k+Q, r_μ), ε_c(k+Q)** — htransform (``bse_setup.compute_wfns_fi``
    with an explicit q-list = {k + Q}): mutually consistent eigenpairs of
    one interpolated fH_{k+Q}; no shifted NSCF, no Sternheimer
    (arbitrary_q_bse.md §1/§2a-b).  htransform returns cell-periodic u at
    wrapped labels — the same torus convention as the stored grid ψ.  The fH
    is built over the FULL loaded band window (all valence + all conduction of
    the input, e.g. nband=40 = 26v+14c), EXACTLY as the standard SP
    bandstructure driver builds it; the BSE conduction bands are a sub-window
    returned interior to it, guarded above by the extra conduction bands so no
    selection boundary cuts a near-degenerate pair.  Running on a sliver
    conduction window instead makes the off-grid caches ring 100-1000 meV
    (spurious exciton dips); use a full-band input.
  * **Direct kernel W(k−k')** — UNCHANGED coarse tiles + coarse-k FFT
    convolution (all k-differences stay on-grid when every conduction leg
    shifts by the same Q; §2c per-element verification).  The matvec is the
    ONE production trial-stack matvec (``bse_stack_matvec``) with the
    conduction slots holding the Q-shifted caches — no parallel kernel.
  * **Exchange V_Q** — from ``bse.vq_interp``:
      ``--vq-mode=interp``  F-scheme + b26p interpolation (fast; ONE jitted
                            evaluator, per-Q dispatch-only).  SLAB ONLY —
                            the model's per-|G_z| channels exist only on a
                            q_z=0 slab, and a bulk deck is refused by name.
      ``--vq-mode=refit``   per-Q ζ refit (compute-don't-interpolate — the
                            off-grid GROUND TRUTH; expensive).  Nothing in
                            it is 2-D: on a ``sys_dim=3`` deck it contracts
                            with the PRODUCER's own Coulomb door, so it is
                            the arbitrary-Q exchange a bulk crystal runs,
                            not merely a checking mode.
      ``--vq-mode=both``    interp on the full path + refit on
                            ``--refit-points`` spot checks, solved in the
                            SAME compiled scan (extra scan rows) so the
                            interp-vs-refit ΔE_S table is apples-to-apples.
      ``--vq-mode=ongrid``  no model at all: the stored ``V_qmunu[wrap(−Q)]``
                            tile, exact, but only at Q on the coarse
                            exchange-tile grid — which ``--q-per-segment``
                            (a FLOOR of 16 by default, because E_S(Q) on a
                            path is a bandstructure) puts most Q off.
    Momentum labeling: the pair density conj(ψ_c[k+Q]) ψ_v[k] pairs with
    the tile at TILE momentum q = wrap(−Q) (reference-metric convention,
    ``vq_interp`` docstrings); on-grid this is exactly the stored
    ``V_qmunu[wrap(−Q)]`` slot.

    WHICH WINDOW THE REFIT FITS ζ' ON — ``--refit-window``, and it decides
    which gate certifies the run.  ``zeta`` (default) fits on the producer's
    own ζ-fit window, reproduces the stored ``V_qmunu`` tiles, and is
    certified BY that tile identity (``vq_interp.refit_ongrid_null``).  That
    needs the parent ISDF basis to span the ζ-fit window's Galerkin rank
    bound, ``n_μ,parent · n_s ≥ nk · nb_ζ``, and a wide GW window can exceed
    it — on the Si 4×4×4 / 960-centroid / 60-band lineage the bound is 3840
    against a basis of 1920 and ``build_fH_R`` correctly refuses.  ``bse``
    fits ζ' on the DECK's window instead, i.e. the BSE window plus its
    conduction guards: the exchange contracts BSE-window pairs and nothing
    else, so that window carries every pair density the kernel will ask for,
    and the bound drops with it.  The price is that ζ' ≠ ζ, so the stored
    tiles do NOT come back and the tile null is unavailable — the gate moves
    up to the CONTRACTED object (:func:`_certify_refit_against_stored`): at
    every path Q that lands on the coarse exchange-tile grid the BSE is solved
    twice in the same scan, once through the refit exchange and once through
    the producer's stored tile, and the eigenvalues must agree to
    the tolerance ``--cert-grade`` names.  The two modes are not a tolerance
    apart; they certify different objects, and the windowed one certifies the
    object that is published.  ``--cert-grade`` selects between exactly two
    NAMED module constants and never a number — :data:`CERT_TOL_BY_GRADE`,
    whose two entries carry the whole argument; on a pass the grade and the
    certified worst travel into the provenance line, the ``.dat`` and the plot.
  * **Γ endpoint** — at exactly Γ the production q=0 tile is used (stored
    head-body convention: compute_vcoul zeroes G=0, the mini-BZ-averaged
    head is the loader's rank-1 injection).  At every finite Q the G=0 term
    of v(Q+G) is KEPT (BGW ``energy_loss`` convention; finite_q_bse.md) —
    the physical nonanalytic exchange branch, so E(Q→0) need not equal
    E(Γ) on the longitudinal states.  Both facts are written into the .dat
    header.
  * **THE SCAN** — the per-Q solve (hoisted pair amplitudes + block-Lanczos
    over the stack matvec) is one jitted function ``lax.scan``-ned over the
    whole Q list: ONE compile serves every Q (verified by the
    JAX_LOG_COMPILES census; per-q-recompile lesson, PHASE2_LOG).
    Interpolated V_Q are Hermitized (0.5(V+V^H)) — commutes with the pair
    contraction; the anti-Hermitian residue is stencil noise.

Outputs: ``<prefix>.dat`` (path distance, Q, lowest n_eig in eV, labeled
header; refit rows flagged) and ``<prefix>.png`` (bands along the path,
high-symmetry ticks; refit spot checks overlaid as markers).

Band-pad guard: mesh padding adds ψ=0 conduction/valence slots; with ε=0
they would alias into the LOW spectrum (ε_c,pad − ε_v ≈ −ε_v).  Pad
energies are pushed out of the window (ε_c,pad=+1e3 Ry, ε_v,pad=−1e3 Ry) —
pad transitions then sit at +2e3 Ry, harmless to the lowest n_eig.  (No-op
at 1×1 mesh: there are no pads.)

Run (never on a login node; module-free srun+shifter runner):
    python -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 \
        --n-eig 6 --vq-mode both --refit-points 0,8,15,22,29
The input file must carry a K_POINTS crystal_b block (the Q path).
"""
from __future__ import annotations

import os
import time
from functools import partial

import h5py
import numpy as np

# THE startup call (runtime module docstring): env defaults, SLURM-aware
# ``jax.distributed.initialize``, CPU fallback, the run's clique-warmed
# square ('x','y') mesh, compile cache, rank-0 report.  Must run
# BEFORE this module's own ``import jax`` and any ``jax.devices()`` /
# mesh creation so a multi-node srun yields the full global device set.
# This driver names ``--px/--py``; ``create_mesh_xy_from_flags(px, py)`` in
# main() hands back this startup mesh when they are omitted (the default
# since 2026-08-27) and refuses any explicit shape that is not it, so the
# mesh every NamedSharding embeds is the one the report above describes.
from runtime import (debug_print, debug_print_enabled,
                     initialize_communicator_stack, rank0_print)
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from solvers.lanczos import (alpha_herm_sink, block_lanczos_eig_jit,
                             report_alpha_herm, split_alpha_sink)
from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY, MODES,
                                    check_band_window)
from common.collectives import device_put_process_local
from common.fft_helpers import make_sharded_ifftn_3d
from common.preprocessing_output import ScientificProductionReport
from common.progress import LoopProgress
from common.provenance import lorrax_version, provenance_header
from common.scientific_output import abs_path, band_range, policy
from runtime.production_stream import ProductionStdout
from .bse_io import (_find_restart_file, load_bse_data_from_restart_sharded,
                     decimate_W_q_to_subgrid, make_w_densifier,
                     build_w_head_channel, resolve_w_head_densify,
                     _resolve_head_params, PAD_EPS_GUARD_RY)
from .bse_ring_comm import create_mesh_xy_from_flags, make_bse_shardings
from .bse_serial import compute_pair_amplitude
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_window import refuse_eqp_on_a_qp_wfn
from . import vq_interp

RY2EV = 13.6056980659
# PAD_EPS_GUARD_RY is imported, not defined: it belongs at the loader seam
# (``bse_io``) so every BSE driver inherits the same guard, instead of this
# one repairing a wrong ε pad locally as it used to.

#: ``--refit-window=bse`` certification tolerance, in meV, on the CONTRACTED
#: object: BSE eigenvalues at an on-grid Q through the refit exchange against
#: the same eigenvalues through the producer's stored ``V_qmunu`` tile.
#:
#: NOT A FLAG, DELIBERATELY.  A windowed ζ' does not reproduce the stored
#: TILES (see ``vq_interp.refit_window_view``), so the tile-level bracket is
#: unavailable and this is the only thing standing between a re-fitted ζ' and
#: a published curve.  A CLI knob on it is a knob for making a failed
#: certification pass, and the failure mode it guards — an off-grid exchange
#: built on a ζ' that is subtly not the same operator — is silent by
#: construction.  0.01 meV is two orders below the ~1 meV scale on which
#: exciton multiplets are resolved on this lineage and four below the 14.1 meV
#: the un-orthonormal Galerkin route would have printed.  If it fails, the
#: window is wrong or the basis cannot span it; report, do not widen.
REFIT_CERT_TOL_MEV = 0.01

#: The SECOND certification grade, and it is a grade rather than a dial.
#:
#: 1.0 meV, for a deliverable whose features live at TENS of meV: an exciton
#: band structure read as a picture — where the dispersion is, which valley is
#: lowest, how the multiplets order along the path.  A curve certified here is
#: NOT a reference number and must never be quoted as one.
#:
#: WHY IT EXISTS.  The windowed-ζ' route carries a representability error with
#: no downfold in it at all, and on the Si 4×4×4 / μ=960 lineage it was 0.858
#: meV — 86x over :data:`REFIT_CERT_TOL_MEV`, so at reference grade that route
#: produces no picture, and the only path to 0.01 meV off-grid is a
#: 2x-centroid GW re-run.  Refusing to draw a 10 meV feature because a 0.9 meV
#: floor is not a 0.01 meV floor is the wrong trade for a picture and the right
#: one for a number, so both are named and the default stays the number.
#:
#: **1.0 meV IS NOT A CLAIM THAT ANY PARTICULAR DECK CLEARS IT, and on that
#: deck it does not.**  The 0.858 meV was measured at the four on-grid Q a
#: ``--q-per-segment 1`` path reaches, all of them segment CORNERS.  Run the
#: same parent on a real bandstructure path (``--q-per-segment 16``, which is
#: the default) and the interior on-grid Q are a different population: 22.952
#: meV at (0, ¼, ¼) and 6.110 meV at (¼, ¼, ¼), against 0.841 / 0.034 / 0.020
#: at the corners the old control shared.  See
#: ``tests/known_failures/2026-08-11-refit-vq-sharded-fetch-and-cert-grades.md``
#: §4.  So this grade is a statement about what a PICTURE needs, never a
#: prediction about a route, and a deck that misses it is telling you its
#: exchange error is the size of the features — which is the one thing a
#: figure must not be drawn through.
#:
#: WHAT IT IS NOT.  Not a free dial: there are exactly TWO grades, both module
#: constants, and ``--cert-grade`` selects between them by name.  Nothing on
#: the command line, in the environment or in a deck can produce a third
#: tolerance.  The dual-solve certification RUNS unchanged at this grade and
#: REFUSES above it — a visualization-grade run that fails is still a run that
#: wrote no ``.dat`` and no ``.png``.  And on a pass the grade and the
#: certified NUMBER are stamped into the run's provenance block, the ``.dat``
#: header and the rendered plot, so a figure cannot be separated from the
#: tolerance it was drawn under.
CERT_TOL_VISUALIZATION_MEV = 1.0

#: The whole tolerance surface, by name.  A grade is chosen from this table
#: and nowhere else; ``--cert-grade``'s ``choices`` are its keys, so an
#: unknown grade is an argparse refusal rather than a silent default.
CERT_TOL_BY_GRADE = {
    "reference": REFIT_CERT_TOL_MEV,
    "visualization": CERT_TOL_VISUALIZATION_MEV,
}

#: What a stamped artefact says about itself.  One spelling, used by the
#: provenance line, the ``.dat`` header and the plot, so the three cannot
#: drift apart and a grep for the words finds all three.
CERT_GRADE_LABEL = {
    "reference": "reference grade",
    "visualization": "visualization grade",
}


def cert_grade_stamp(grade: str, worst_mev: float) -> str:
    """The one-line certification stamp: the GRADE and the NUMBER, together.

    A grade without its number invites "certified" to be read as "exact", and
    a number without its grade invites a 0.9 meV curve to be quoted as a
    reference. They travel as one string or the claim is not made.
    """
    return (f"CERTIFIED {CERT_GRADE_LABEL[grade]}: worst max|dE_S| "
            f"{worst_mev:.5f} meV against the "
            f"{CERT_TOL_BY_GRADE[grade]:g} meV {grade} gate")


def _gather_host(x):
    """Gather a ``jax.Array`` to a full host numpy array, identical on every
    process, whether it is REPLICATED or PROCESS-SPANNING.

    ONE line of delegation to :func:`common.collectives.gather_to_host`, whose
    docstring owns the three-arm branch and the incidents behind it.  The
    wrapper is not free-standing: ``tests/test_bse_gather_and_mesh.py``
    ratchets it, and its four siblings, to REACH that service — the gate
    exists because five hand-rolled copies of this branch had drifted, three
    of them onto ``tiled=False``, which raises on exactly the arrays their
    callers feed them.

    WHAT THIS DRIVER FEEDS IT, NOW.  Only ``evs_dev`` — the block-Lanczos
    Ritz values, ``P()``-sharded.  The driver's own log line records that such
    an array reports ``fully_addressable=False`` at P=64 (job 7882507), so it
    takes the service's ``is_fully_replicated`` arm: a LOCAL buffer read, no
    collective.  The one caller that used to hand this function a μ-sharded,
    genuinely process-spanning array — the on-grid diagnostic gate — no longer
    does; it contracts μ on device instead (see
    :func:`_gate_stats_on_device` for the three warm-up attempts that failed
    to make that gather work under impl=mpi).  So this driver now issues no
    cross-process host gather at all.
    """
    from common.collectives import gather_to_host
    return gather_to_host(x)


# ===========================================================================
# the single-compile path solver: scan(per-Q block-Lanczos) over the Q list
# ===========================================================================
def build_path_solver(mesh_xy: Mesh, nkx: int, nky: int, nkz: int,
                      nc_pad: int, nv_pad: int, *, n_eig: int,
                      block_size: int, max_iter: int, n_reorth: int | None = None,
                      head_tensor: bool = False):
    """One jitted ``solve_path`` for a whole Q list.

        solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R)              (default)
        solve_path(..., W_R, D_head, M_Q)                     (head_tensor)
            -> (evs (nQ, n_eig), alpha (per-Q α-Hermiticity scalars))

    ``head_tensor`` adds the cell-averaged nonanalytic exchange head as a
    rank-three-over-transitions term (``bse_stack_matvec``'s ``head_tensor``).
    ``D_head`` is ``conj(d)`` on the BSE window, ``(nk, nc_pad, nv_pad, 3)``,
    Q-INDEPENDENT — it is the q→0 coefficient of the pair amplitude, a
    property of the k-point transition — while ``M_Q`` is the per-Q cell
    moment ``(nQ, 3, 3)`` and therefore rides the scan xs beside ``V_Q``.
    Default False traces the pre-existing program unchanged.

    The second output is the α-Hermiticity report as DATA: it is what keeps
    this program free of host callbacks and therefore persistable in JAX's
    compilation cache.  Feed it to :func:`_report_alpha_over_path` together
    with ``solve_path.alpha_labels`` (filled at trace time) to run the gate on
    the host.  Nothing about the invariant changes — only where it is emitted.

    The scan body per Q: hoist the exchange pair amplitudes M_X/M_Y from
    the Q-shifted conduction ψ (audit-P3 contract — matvec args, computed
    once per solve, reused across all Lanczos iterations), then run the
    fixed-iteration block Lanczos with the ONE production stack matvec.
    Everything Q-dependent arrives as scan xs; Q-independent operands
    (valence ψ/ε, W_R) are loop constants.  ONE XLA compile serves every
    Q — and every later call with the same nQ (compile census deliverable).
    """
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    n_flat = nc_pad * nv_pad * nk
    if n_reorth is None:
        # FULL reorthogonalisation by default: exciton windows are small
        # (n_flat = nc·nv·nk ~ 10²-10⁴), the Krylov space often saturates
        # (solvers.lanczos clamps at floor(n/bs)), and partial reorth at
        # saturation breeds ghost duplicates.  Full reorth costs
        # O(M·n·bs²) — negligible next to the matvec at every BSE size.
        n_reorth = max_iter
    matvec = build_bse_stack_matvec(mesh_xy, nkx, nky, nkz, kernel="bse",
                                    head_tensor=head_tensor)
    # Filled at TRACE time by the sink below (see the module note on the
    # persistable-scan fix); the static half of the α-Hermiticity report.
    alpha_labels: list = []

    @jax.jit
    def solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R, D_head=None, M_Q=None):
        def body(carry, xs):
            if head_tensor:
                psi_c_X, psi_c_Y, eps_c, V, M_head = xs
            else:
                psi_c_X, psi_c_Y, eps_c, V = xs
                M_head = None
            psi_c_X = lax.with_sharding_constraint(psi_c_X, sh.psi_x)
            psi_c_Y = lax.with_sharding_constraint(psi_c_Y, sh.psi_y)
            V = lax.with_sharding_constraint(V, sh.V)
            M_X = lax.with_sharding_constraint(
                compute_pair_amplitude(psi_c_X, psi_v_X), sh.psi_x)
            M_Y = lax.with_sharding_constraint(
                compute_pair_amplitude(psi_c_Y, psi_v_Y), sh.psi_y)

            def matvec_block(Vb):
                X = Vb.reshape(block_size, nc_pad, nv_pad, nk)
                X = lax.with_sharding_constraint(X, sh.X)
                extra = (D_head, M_head) if head_tensor else ()
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                            eps_c, eps_v, W_R, V, M_X, M_Y, *extra)
                HX = HX.reshape(block_size, -1)
                return HX

            # The α-Hermiticity report leaves the trace as DATA, not as a host
            # callback.  jax._src.compiler._cache_write refuses to persist any
            # module carrying a host callback, and this jit wraps a lax.scan
            # over the whole Q path — the single most expensive compile in the
            # driver.  Un-sunk, it was rebuilt on every warm run.  The sink is
            # opened INSIDE the scan body because that is the trace the solver
            # is traced in; the three scalars per Q ride out as scan ys.
            with alpha_herm_sink() as _sink:
                evs, _ = block_lanczos_eig_jit(
                    matvec_block, n_flat, n_eig=n_eig, block_size=block_size,
                    max_iter=max_iter, n_reorth=n_reorth)
            _labels, _payload = split_alpha_sink(_sink)
            alpha_labels[:] = _labels
            return carry, (evs[:n_eig].real, _payload)

        xs = ((psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q, M_Q) if head_tensor
              else (psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q))
        _, (evs_all, alpha_all) = lax.scan(body, None, xs)
        return evs_all, alpha_all

    solve_path.alpha_labels = alpha_labels
    return solve_path


def _report_alpha_over_path(labels, alpha_all, log=print):
    """Run the α-Hermiticity gate on the per-Q scalars the scan returned.

    ``alpha_all`` is the scan-stacked payload: one ``(dev, scale, worst)``
    triple per label, each of shape ``(nQ,)``.  The gate's verdict is a
    per-solve ratio ``dev/scale`` against ``ALPHA_HERM_RTOL``, so reporting the
    Q that MAXIMISES that ratio reports the worst violator on the whole path —
    a stricter statement than any single Q, in one line instead of nQ lines.
    The Q index is named so a failure is locatable.

    Returns True only if the worst Q on every label passes (and therefore, the
    ratio being maximised, only if every Q passes).
    """
    if not labels:
        return True
    ok = True
    for (name, form), (dev, scale, worst) in zip(labels, alpha_all):
        dev = np.asarray(dev)
        scale = np.asarray(scale)
        worst = np.asarray(worst)
        rel = dev / np.maximum(scale, np.finfo(dev.dtype).tiny)
        iq = int(np.argmax(rel))
        log(f"[alpha-herm] worst of {dev.size} Q is Q#{iq} "
            f"(dev/scale = {rel[iq]:.3e})")
        # ``form`` is a LOOKUP KEY into solvers.lanczos._ALPHA_FORMS, not free
        # text; the per-path context goes in ``name``, which is.
        ok = report_alpha_herm(
            ((f"{name} (worst of {dev.size} Q, at Q#{iq})", form),),
            ((dev[iq], scale[iq], worst[iq]),)) and ok
    return ok


# ===========================================================================
# stacks: htransform conduction caches + V_Q tiles for the whole path
# ===========================================================================
def build_head_dipole_operand(args, nk, nc_pad, nv_pad, n_val, n_cond,
                              *, log=print):
    """``conj(d)`` on the BSE window, ``(nk, nc_pad, nv_pad, 3)`` complex.

    The cell-averaged exchange head is
    ``K^head_{t,t'} = (1/N_k) conj(d_a(t)) M_ab d_b(t')``, and ``d`` is the
    dipole LORRAX already computes and ships in ``dipole.h5``: the head's
    q-linear coefficient is exactly ``∂_q M_cv(k,q,0)|_0 = -i d``, so the
    ``∂_q ζ̃`` the μ-basis route would have needed never appears
    (``LT_HEAD_PROBLEM.md`` §6).

    Returned pre-conjugated and in ``M_X``'s ``(k, c, v, ·)`` layout, with a
    Cartesian axis of length 3 where μ was, so the matvec's head term reads
    exactly as its exchange term (``bse_stack_matvec``, ``head_tensor``).
    The ``(c, v)`` window and its padding match the loader's:
    ``val_indices = arange(n_occ-n_val, n_occ)`` lowest-first, which is the
    same slice ``slice_dipole_to_bse_window`` takes.

    REPRESENTATION CONSISTENCY, STATED RATHER THAN GLOSSED.  The exchange
    BODY lives in the ISDF centroid basis and this head does not — the
    dipoles are exact Cartesian matrix elements on the G-sphere.  That is a
    deliberate, controlled inconsistency (``LT_HEAD_PROBLEM.md`` §6.4): the
    μ-basis reconstruction of ``lim_{q→0} M_cv`` is precisely the quantity
    whose failure to vanish leaves the spurious Γ residual, and the exact
    dipole has no such residual, so the trade favours the dipole.  Two
    consequences follow and are real: head and body no longer cancel each
    other's ISDF error, and no run with the head on is bit-comparable to one
    without it.  That is why the feature is opt-in and why the off arm is
    gated bit-identical.

    The dipoles also carry the velocity-sign convention of the FILE they came
    from.  Since the head is quadratic in ``d`` it inherits that convention
    at full strength.  As of 2026-08-09 the tree's default arm is ``+1`` and
    the tracked fixtures were re-cut on it, so the provenance stamp is read
    and logged rather than assumed.
    """
    from .absorption_common import (build_dipole_vector_bse, load_dipole_h5,
                                    slice_dipole_to_bse_window)
    from .bse_io import resolve_n_occ

    path = args.dipole if getattr(args, "dipole", None) else \
        os.path.join(os.path.dirname(os.path.abspath(args.input)), "dipole.h5")
    dipole_cart, deltaE, attrs = load_dipole_h5(path)
    with h5py.File(path, "r") as f:
        sign = f.attrs.get("prov_vnl_velocity_sign", None)
        skip_vnl = f.attrs.get("skip_vnl", None)
    if int(attrs["nk"]) != int(nk):
        raise SystemExit(
            f"{path} has nk={attrs['nk']} but the BSE grid has nk={nk}.  The "
            f"head tensor contracts the dipole against the BSE transition "
            f"index, so the two k-lists must be the same list in the same "
            f"order.  Rebuild the dipole from this deck's WFN.")
    n_occ = resolve_n_occ(None, input_file=args.input)
    d_alpha, _ = slice_dipole_to_bse_window(dipole_cart, deltaE,
                                            n_occ, n_val, n_cond)
    d_vec = build_dipole_vector_bse(d_alpha, n_cond_pad=nc_pad,
                                    n_val_pad=nv_pad)      # (3, c, v, k)
    log(f"  [head-tensor] dipoles from {os.path.basename(path)} "
        f"(nk={attrs['nk']}, nbands={attrs['nbands']}, n_occ={n_occ}, "
        f"velocity_sign={sign!r}, skip_vnl={skip_vnl!r}); "
        f"max|d|^2 = {float(np.max(np.abs(d_vec)**2)):.4g} bohr^2")
    return np.conj(np.transpose(d_vec, (3, 1, 2, 0)))      # (k, c, v, 3)


def resolve_isdf_basis(restart_file, params, input_file, *, n_rmu_bundle,
                       log=print):
    """``(centroid_table, keep_idx | None)`` — WHICH ISDF basis to fit ψ in.

    On a natively-fitted bundle this is the deck's ``centroids_file`` and
    nothing else happens: the restart's μ basis IS the deck's centroid set,
    and ``keep_idx`` is ``None``.  That arm is byte-for-byte the program this
    driver has always run.

    ON A DOWNFOLDED BUNDLE THE TWO ARE DIFFERENT OBJECTS, and conflating them
    is the second of the three ways this driver failed on one (measured
    2026-08-10, PIPELINE_HEALTH.md).  The small bundle's basis is a SUBSET of
    the parent's centroid points — ``downfold.md``: "the new
    wavefunction-at-centroid coefficients are a literal column slice of the
    ones already on disk" — and the htransform leg must evaluate its fitted
    whole-state basis at those same child points.  The basis itself is selected
    from full Bloch states by the one randomized-QRCP implementation in
    ``isdf.galerkin``; centroid rows are evaluation points, not a second
    state-space fit or rank authority.  This function resolves the parent table
    only so ``keep_idx`` can identify the child coordinates without inventing
    another downfold mapping.

    The parent's table comes off the bundle's own ``downfold_provenance``,
    not off the deck: a deck copied beside the small bundle names a file that
    is not there (the walk's first failure — a bare ``FileNotFoundError`` out
    of ``np.loadtxt``, with nothing to say it was a downfold consequence).
    The deck is still honoured when it names a readable table of the right
    size, so a user who has arranged their own is not overruled.
    """
    from file_io import read_downfold_provenance

    input_dir = os.path.dirname(os.path.abspath(input_file))

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(input_dir, p)

    deck_path = _resolve(str(params.get("centroids_file",
                                        "centroids_frac.txt")))
    prov = read_downfold_provenance(restart_file)

    if prov is None:
        if not os.path.isfile(deck_path):
            raise SystemExit(
                f"exciton_bands: the deck's centroids_file resolves to "
                f"{deck_path}, which does not exist.  This driver refits "
                f"psi(k+Q) at the centroid points rather than reading the "
                f"stored coefficients, so it needs the coordinate table the "
                f"restart's ISDF basis was built on.  Point centroids_file "
                f"at it, or copy it beside {os.path.basename(input_file)}.  "
                f"(bse.bse_jax reads this bundle without a centroid table at "
                f"all, which is why the two drivers disagree about whether "
                f"this file is needed.)")
        return deck_path, None

    keep_idx = prov.get("keep_idx")
    mu_parent = int(prov.get("parent_mu", 0))
    if keep_idx is None:
        raise SystemExit(
            f"exciton_bands: {restart_file} carries a downfold_provenance "
            f"group with no keep_idx dataset, so there is no way to map its "
            f"mu={n_rmu_bundle} basis back to the parent's centroid points.  "
            f"The bundle predates the stamp; re-run gw.downfold_cli on the "
            f"parent to produce one this driver can open.")
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    if keep_idx.shape[0] != int(n_rmu_bundle):
        raise SystemExit(
            f"exciton_bands: {restart_file} stores mu={n_rmu_bundle} but its "
            f"downfold_provenance keeps {keep_idx.shape[0]} parent centroid "
            f"rows.  The bundle and its provenance describe different bases; "
            f"re-run the downfold rather than trusting either.")

    # The parent table: the bundle's own record first, the deck second.
    cands = []
    stamped = prov.get("parent_centroids_file")
    if stamped:
        cands.append((str(stamped), "downfold_provenance"))
    cands.append((deck_path, "the deck's centroids_file"))
    for path, whence in cands:
        if not os.path.isfile(path):
            continue
        n_rows = int(np.loadtxt(path, ndmin=2).shape[0])
        if mu_parent and n_rows != mu_parent:
            log(f"  [downfold] {path} ({whence}) has {n_rows} rows, not the "
                f"parent's {mu_parent} — not the table this bundle was "
                f"downfolded from; skipped.")
            continue
        log(f"  [downfold] this bundle is a DOWNFOLD of "
            f"{prov.get('parent_file', '?')} (mu {mu_parent} -> "
            f"{n_rmu_bundle}).  Resolved parent table {path} ({whence}) and "
            f"its ordered {keep_idx.shape[0]}-row child subset.  The published "
            f"whole-state htransform basis is evaluated directly on these "
            f"child rows.")
        return path, keep_idx

    raise SystemExit(
        f"exciton_bands: {restart_file} is a downfolded bundle (mu "
        f"{mu_parent} -> {n_rmu_bundle}) and this driver needs the PARENT "
        f"run's centroid table to refit psi(k+Q); none of "
        f"{[c[0] for c in cands]} is a readable {mu_parent}-row table.\n"
        f"  Why the parent table: keep_idx names rows of that table; it is "
        f"the coordinate authority needed to evaluate the whole-state basis "
        f"directly at the child points.  It is not a parent-width fit.\n"
        f"  Fix: re-run gw.downfold_cli with parent_centroids_file set to "
        f"the parent deck's centroids_file (the driver then records its path "
        f"on the bundle and this resolves itself), or point this deck's "
        f"centroids_file at that table directly.")


def _certify_refit_against_stored(cert_idx, Qpath, evs_refit_route,
                                  evs_stored_route, kgrid_vq, log=print,
                                  grade: str = "reference",
                                  window_mode: str = "bse"):
    """The CONTRACTED gate: eigenvalues, not tiles.

    ``window_mode`` says what this gate IS on this run, and it changes the
    wording and nothing else — the comparison, the grade and the tolerance are
    the same object either way.  Under ``bse`` it REPLACES the tile null (ζ'
    ≠ ζ, so the tile identity does not exist).  Under ``zeta`` it rides
    ALONGSIDE it and adds what a relative tile bracket cannot say: how many
    meV the published eigenvalues move.

    CERTIFY WHERE CONSUMED, one level up.  With ζ' fitted on the BSE window
    the refit no longer computes the producer's ``V_qmunu``, so the tile
    identity that guards ``--refit-window=zeta`` is unavailable — and widening
    its bracket would be widening a comparison that has no reason to hold.
    What is still exactly comparable is the thing the driver actually
    delivers: at a Q that lies ON the coarse exchange-tile grid, the producer
    wrote an exact tile, so the BSE at that Q can be solved twice — once with
    the refit's tile and once with the producer's — with every other operand
    (ψ_c(k+Q), ε_c(k+Q), W_R, the solver, the Lanczos block) literally the
    same array. The eigenvalue difference is then attributable to the exchange
    and to nothing else, and it is measured in the units the curve is
    published in.

    THE TOLERANCE IS A NAMED GRADE, NOT A NUMBER THIS FUNCTION ACCEPTS.  It
    takes ``grade``, a key of :data:`CERT_TOL_BY_GRADE`, and reads the
    tolerance out of the module constant behind it — ``reference`` (default,
    :data:`REFIT_CERT_TOL_MEV`, the published-number gate) or
    ``visualization`` (:data:`CERT_TOL_VISUALIZATION_MEV`, the picture gate).
    There are two and there is no third: a float parameter here would be a
    route to making a failed certification pass, which is what the original
    constant's docstring refused and what this signature keeps refusing.  A
    failure at EITHER grade means the windowed ζ' is not the same exchange
    operator on the BSE pair space — the window is wrong, or the basis cannot
    span it — and the honest response is to report the number.

    Returns ``(rows, worst)`` — the per-Q ``(iQ, tile_index, max|Δ| meV)`` and
    the worst of them — so the caller can stamp both the numbers and the
    certified worst into the ``.dat`` header, the provenance line and the
    plot: a certification whose numbers do not travel with the data is a
    claim, not a record.
    """
    if grade not in CERT_TOL_BY_GRADE:
        raise SystemExit(
            f"exciton_bands: cert grade {grade!r} is not one of "
            f"{sorted(CERT_TOL_BY_GRADE)}.  Grades are module constants; "
            f"there is no third tolerance to reach.")
    tol_mev = CERT_TOL_BY_GRADE[grade]
    kg = np.asarray(kgrid_vq, dtype=int)
    rows, worst, worst_iQ = [], 0.0, -1
    for j, iQ in enumerate(cert_idx):
        q_tile = -Qpath[iQ] - np.round(-Qpath[iQ])
        tile = tuple(int(v) for v in
                     (np.round(q_tile * kg).astype(int) % kg))
        d_mev = np.abs(np.asarray(evs_refit_route[iQ])
                       - np.asarray(evs_stored_route[j])) * RY2EV * 1e3
        dmax = float(np.max(d_mev))
        rows.append((int(iQ), tile, dmax))
        if dmax > worst:
            worst, worst_iQ = dmax, int(iQ)
        log(f"  [refit-cert] Q#{iQ} = "
            f"{np.array2string(Qpath[iQ], precision=4)} (tile {tile}): "
            f"max|ΔE_S| refit-route vs stored-route = {dmax:.5f} meV "
            f"over {len(d_mev)} levels")
    log(f"  [refit-cert] WORST {worst:.5f} meV at Q#{worst_iQ} over "
        f"{len(rows)} on-grid Q, gate {tol_mev:g} meV ({grade} grade) — "
        + ("this is the BSE-window ζ' certification, and it replaces (does "
           "not relax) the tile-level on-grid null, which a windowed ζ' "
           "cannot satisfy." if window_mode == "bse" else
           "this run's ζ' is the producer's ζ, so the tile null certified "
           "the TILES and this certifies the EIGENVALUES; both are gates and "
           "neither relaxes the other."))
    if not np.isfinite(worst) or worst > float(tol_mev):
        raise SystemExit(
            f"exciton_bands --refit-window={window_mode}: the refit does "
            f"NOT reproduce the stored-tile BSE eigenvalues on the coarse "
            f"grid — worst max|ΔE_S| = {worst:.5f} meV at Q#{worst_iQ} "
            f"against the {tol_mev:g} meV {grade} gate.  Every OFF-grid "
            f"exchange tile in this run comes from the same ζ', so nothing "
            f"on this path is worth plotting.\n"
            + ("  THE TILE NULL PASSED AND THIS DID NOT, which is exactly "
               "what this gate exists to catch: a 5.0e-02 RELATIVE bracket "
               "on a tile is not a meV statement about an eigenvalue.  Report "
               "both numbers.\n" if window_mode == "zeta" else "")
            + ("  --cert-grade=visualization is the ONE other gate that "
               f"exists ({CERT_TOL_VISUALIZATION_MEV:g} meV, for a picture "
               f"and never for a quoted number); this run is already at it, "
               f"so there is nothing further to select.\n"
               if grade == "visualization" else
               f"  --cert-grade=visualization is the ONE other gate that "
               f"exists ({CERT_TOL_VISUALIZATION_MEV:g} meV).  It is for a "
               f"FIGURE whose features live at tens of meV and never for a "
               f"quoted number, and this run missed the reference gate by "
               f"{worst / max(float(tol_mev), 1e-300):.0f}x — decide whether "
               f"the deliverable is a picture before reaching for it.\n")
            + f"  This is not a tolerance to widen.  What it says is that the "
            f"re-fitted ζ' is not the same exchange operator on the BSE pair "
            f"space as the producer's ζ, which is a statement about the "
            f"WINDOW and the BASIS:\n"
            f"    * the refit window may be too narrow to carry the BSE "
            f"window's pair densities — widen the deck's nval/ncond and "
            f"re-run;\n"
            f"    * or the fitted whole-state basis may not represent this "
            f"window — check the QRCP projection receipts from "
            f"build_fH_R and the full-r residual from refit_prepare;\n"
            f"    * or the BSE window itself cuts a multiplet — check the "
            f"band-window lines above.\n"
            f"  Report the number.  Do not raise REFIT_CERT_TOL_MEV, and do "
            f"not read --cert-grade as a way to: it selects between two "
            f"NAMED grades that mean different things about the artefact, "
            f"and neither of them is a number to edit.")
    log(f"  [refit-cert] {cert_grade_stamp(grade, worst)} — stamped into the "
        f"provenance line, the .dat header and the plot below, so no figure "
        f"from this run can be separated from the tolerance it was drawn "
        f"under.")
    return rows, worst


def apply_q_per_segment(params, q_per_segment, log=print) -> int:
    """Raise every ``K_POINTS crystal_b`` segment to at least N points.

    IT IS A BANDSTRUCTURE, SO SAMPLE IT LIKE ONE.  The deck format carries a
    per-segment point count, and the decks in this campaign carried counts of
    one and two — which draws Γ–X–W–L–Γ–Σ with eight diagonalisations in
    total and produces a plot of straight lines between corners.  The owner's
    reading of that plot ("it looks like shit ... i thought that was
    obvious") is the reason this floor exists and is on by default.

    A FLOOR, not an override, and the distinction is the whole gate:

      * a segment the deck leaves sparser than ``q_per_segment`` is raised to
        it, so the DEFAULT (16) draws at least 16 diagonalisations per
        segment on every deck that ever ran, without editing any deck;
      * a segment the deck already asks for MORE points on keeps its own
        count, so a hand-tuned dense deck is not silently coarsened;
      * ``--q-per-segment 1`` is the identity — every deck count is already
        >= 1 — so the sparse behaviour is still reachable, by asking for it
        explicitly and only that way.

    Only ``bse.exciton_bands`` calls this.  The mutation is on the parsed
    ``params`` dict this driver owns for the length of one run, so the
    single-particle bandstructure driver reading the same deck is untouched.

    Returns the number of path points the raised path will carry.
    """
    seginfo = params.get("kpoints_crystal_b") or {}
    segs = seginfo.get("segments", [])
    if len(segs) < 2:
        return 0
    floor = int(q_per_segment)
    if floor < 1:
        raise SystemExit(
            f"exciton_bands: --q-per-segment={floor} is not a point count.  "
            f"1 is the identity (every deck count is already >= 1) and 16 is "
            f"the default.")
    # segs[0] is the path's first corner; segs[i>0] carries the count for the
    # segment that ENDS at it (generate_kpath_from_qe_segments).
    deck_counts = [max(1, int(s.get("n", 1))) for s in segs[1:]]
    for seg, n_deck in zip(segs[1:], deck_counts):
        seg["n"] = max(n_deck, floor)
    raised = [(i, a, max(a, floor))
              for i, a in enumerate(deck_counts, start=1) if a < floor]
    n_pts = 1 + sum(max(a, floor) for a in deck_counts)
    if raised:
        log(f"  [q-per-segment] floor {floor}: raised {len(raised)} of "
            f"{len(deck_counts)} segment(s) "
            f"({', '.join(f'#{i}: {a}->{b}' for i, a, b in raised)}); "
            f"the path now carries {n_pts} Q "
            f"(deck alone: {1 + sum(deck_counts)}).  It is a bandstructure — "
            f"pass --q-per-segment 1 for the deck's own counts.")
    else:
        log(f"  [q-per-segment] floor {floor}: every deck segment already "
            f"asks for at least that many; path unchanged at {n_pts} Q.")
    return n_pts


def require_zeta_for_interp(restart_file, vq_mode, kgrid) -> str:
    """The ζ file the off-grid exchange needs — or a refusal that explains it.

    ANNOUNCE OR REFUSE, BEFORE h5py GETS TO SAY IT ITS WAY.
    ``vq_interp.build_vq_evaluator`` looks for ``zeta_q.h5`` beside the
    restart, and a missing one is not an exotic condition: it is what EVERY
    downfolded bundle written before 2026-08-10 looks like, and what one
    whose parent's ζ was stored on the q-IBZ wedge still looks like.  The
    bare ``OSError`` from that path names a file and nothing else — no hint
    that the absence is a downfold consequence, and no hint that
    ``--vq-mode=ongrid`` needs no ζ at all.  Third of the three ways this
    driver failed on a downfolded bundle (PIPELINE_HEALTH.md, 2026-08-10).

    A separate function rather than an inline block so the refusal can be
    gated without a solve — same reason ``rerun_check_enabled`` is one.
    """
    if vq_mode == "ongrid":
        return ""
    path = os.path.join(os.path.dirname(os.path.abspath(restart_file)),
                        "zeta_q.h5")
    if os.path.isfile(path):
        return path
    prov = None
    try:
        from file_io import read_downfold_provenance
        prov = read_downfold_provenance(restart_file)
    except Exception:                                          # noqa: BLE001
        pass
    nx, ny, nz = (int(v) for v in kgrid)
    raise SystemExit(
        f"exciton_bands: --vq-mode={vq_mode} builds the exchange tile at "
        f"arbitrary Q by interpolating the stored zeta, and there is no "
        f"{path}.\n"
        + ("  This bundle is a DOWNFOLD.  gw.downfold_cli transports zeta "
           "into the small basis (zeta_S = conj(T[q]) zeta_L) and writes it "
           "beside the bundle, so a bundle without one was either written "
           "before that existed or has a PARENT whose zeta is stored on the "
           "q-IBZ wedge — which vq_interp refuses on the parent too, so "
           "--vq-mode=interp was never available on this lineage.  Re-fit "
           "the parent with LORRAX_FORCE_FULL_BZ=1 and re-run the "
           "downfold.\n"
           if prov is not None else
           "  The GW run that wrote this bundle left no zeta_q.h5 beside it "
           "(a moved tmp/, or a restart written without the fit).\n")
        + f"  --vq-mode=ongrid needs no zeta at all and is EXACT at every Q "
          f"that lands on the {nx}x{ny}x{nz} BSE grid.")


def build_conduction_stacks(bundle, nQ, nk, n_cond, n_cond_pad, n_rmu,
                            n_rmu_pad, mesh_xy):
    """Reshape the htransform bundle over the concatenated {k+Q} list into
    per-Q conduction caches, padded to the loader's mesh extents — one
    jitted reshape+pad+reshard, no host round-trip (the bundle stays on
    device; at 40 path points the old device_get→np.pad→2×device_put moved
    ~1.7 GB through the host).

    Downfold selections are applied when the whole-state basis is evaluated;
    the incoming bundle is already child-width.  There is no legacy
    parent-width slice in this consumer.

    THE RESTART'S μ IS THE AUTHORITY, and the assertion below says so: the
    htransform's basis is an input to this function, ``n_rmu`` is what every
    other operand in the solve was sized by, and a mismatch is a wrong-basis
    contraction that every shape check downstream would pass once the pad
    absorbed it.  (Same shape of reasoning as the post-snap window authority
    at 7449ece0: one place resolves the extent, everything else asserts
    against it rather than re-deriving it.)

    Returns (psi_cQ_X, psi_cQ_Y, eps_cQ):
        psi_cQ_[XY]: (nQ, nk, nc_pad, ns, n_rmu_pad), μ on x / y
        eps_cQ:      (nQ, nk, nc_pad) — pad bands at +PAD_EPS_GUARD_RY
    """
    x5 = NamedSharding(mesh_xy, P(None, None, None, None, "x"))
    y5 = NamedSharding(mesh_xy, P(None, None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    n_rmu_src = int(bundle.psi_rmu_Y.shape[-1])
    if n_rmu_src != int(n_rmu):
        raise ValueError(
            f"exciton_bands: the htransform fitted psi at {n_rmu_src} "
            f"centroid point(s), but the restart bundle stores mu={n_rmu}.  "
              f"The conduction "
              f"caches would be contracted against a W and V in a different "
              f"basis.  The restart's mu is the authority here: point the "
              f"deck's centroids_file at the table this bundle's basis came "
              f"from (a downfolded bundle records the parent's path in its "
              f"downfold_provenance group).")

    @partial(jax.jit, out_shardings=(x5, y5, rep))
    def _stacks(psi, eps):
        ns = psi.shape[2]
        psi = psi.reshape(nQ, nk, n_cond, ns, n_rmu_src)
        eps = eps.reshape(nQ, nk, n_cond)
        psi = jnp.pad(psi, ((0, 0), (0, 0), (0, n_cond_pad - n_cond),
                            (0, 0), (0, n_rmu_pad - n_rmu)))
        eps = jnp.pad(eps, ((0, 0), (0, 0), (0, n_cond_pad - n_cond)),
                      constant_values=PAD_EPS_GUARD_RY)
        return psi, psi, eps

    return _stacks(bundle.psi_rmu_Y, bundle.enk_full)


def _gate_stats_on_device(
    eps_ht, eps_st, psi_ht, psi_st, nc, mesh_xy,
    *, htransform_k_indices=None, stored_k_indices=None,
):
    """Both gate numbers computed ON DEVICE; only replicated results read back.

    Returns ``(max|Δε_c|, Gram(nk, nc, nc))`` as REPLICATED arrays, read with
    ``addressable_data(0)`` — a purely local buffer read that issues no
    collective at all.

    WHY NOT ``gather_to_host``.  Three separate attempts to make a host gather
    work here failed on hardware at P=16 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi``, always in the same frame and
    always with "Communicator requested from a thread that is not the one MPI
    was initialized from": the mesh-clique warm-up did not cover it (job
    7882523), a ``process_allgather`` warm-up on a host operand did not (job
    7882531), and a path-exact warm-up on a genuinely sharded operand did not
    either (job 7882555/7882561).  ``multihost_utils.process_allgather``
    identity-jits to ``P()``, and at this point in the program that jit
    acquires a communicator from an intra-op pool worker; the warm-up
    mechanism that fixes ``shard_map(psum)`` cliques does not fix it.

    So stop trying to gather.  The μ contraction rides a mesh-AXIS psum, which
    IS warmed and works; everything else here is already replicated, so the
    host side needs no collective whatsoever.  This is also strictly cheaper:
    the old code pulled ``nk·nc·ns·n_μ`` complex numbers to every process
    inside a diagnostic, on the one axis the whole BSE is sharded over.

    Pad slots are exact zeros on both ψ legs, so the padded μ extent is
    carried through rather than sliced — slicing a sharded axis would be a
    reshard, and adding zeros changes neither a norm nor an inner product.
    """
    rep = NamedSharding(mesh_xy, P())
    ht_idx = (None if htransform_k_indices is None else
              np.asarray(htransform_k_indices, dtype=np.int32))
    st_idx = (None if stored_k_indices is None else
              np.asarray(stored_k_indices, dtype=np.int32))

    @partial(jax.jit, out_shardings=(rep, rep, rep))
    def _f(e_ht, e_st, A, B):
        # A diagnostic caller may hold two internally coherent native tables
        # on different grids.  Select its paired physical rows ON DEVICE
        # before any subtraction or mu contraction.  The index tables are
        # closed-over integer metadata from symmetry_maps, so this remains one
        # jit and neither mu-sharded wavefunction is gathered.  The production
        # bse_k_grid path is NOT such a caller: its general loader has already
        # rebuilt both operands on the fine grid and compares every row.
        if ht_idx is not None:
            e_ht = jnp.take(e_ht, ht_idx, axis=0)
            A = jnp.take(A, ht_idx, axis=0)
            e_st = jnp.take(e_st, st_idx, axis=0)
            B = jnp.take(B, st_idx, axis=0)
        # ORDER.  htransform returns eigenvalues ASCENDING (they come out of an
        # eigensolve); the stored restart holds them in DFT-BAND-INDEX order.
        # QP corrections reorder bands, so those two orders differ BY
        # CONSTRUCTION, not by error.  Differencing them index-by-index — which
        # is what this gate did — measures the permutation, not the
        # interpolation.
        #
        # Measured on run_EXB_c785 (2026-07-31): the index-wise number is
        # 80.694 meV and is reproducible from ``eqp1.dat`` ALONE, with no
        # htransform, no Galerkin basis and no interpolation in the arithmetic:
        #   max_k max_{b in [26,34)} |sort(EQP)[b] - EQP[b]| = 80.6935 meV
        #   at k=(0.25,-0.5,0), band 30, where the QP shifts push band 30 above
        #   31 and 32.
        # The order-matched number on the same run is 6.5e-11 meV.  The 80.694
        # was read as this path's accuracy limit for a whole campaign.  The
        # Perlmutter campaign hit the identical trap at a fixed 57.902 meV.
        #
        # ``d`` is therefore the SET comparison — did htransform recover the
        # same conduction manifold — which is the question this gate asks.
        # ``d_idx`` is kept and reported alongside so the permutation stays
        # VISIBLE rather than silently folded into the accuracy number: a large
        # d_idx with a tiny d is band reordering and is expected; a large d is
        # the interpolation basis failing.
        d = jnp.max(jnp.abs(jnp.sort(e_ht[:, :nc], axis=-1)
                            - jnp.sort(e_st[:, :nc], axis=-1)))
        d_idx = jnp.max(jnp.abs(e_ht[:, :nc] - e_st[:, :nc]))
        A = A[:, :nc]
        B = B[:, :nc]
        nA = jnp.sqrt(jnp.sum(jnp.abs(A) ** 2, axis=(2, 3)))[:, :, None, None]
        nB = jnp.sqrt(jnp.sum(jnp.abs(B) ** 2, axis=(2, 3)))[:, :, None, None]
        G = jnp.einsum("kasm,kbsm->kab", jnp.conj(A / nA), B / nB)
        return d, d_idx, G

    return _f(eps_ht, eps_st, psi_ht, psi_st)


def _local(x):
    """This process's own buffer of a REPLICATED device array — no collective.

    ``addressable_data(0)`` is a local read.  For a ``P()``-sharded array
    every process holds the whole thing, so this IS the full array; unlike
    ``gather_to_host`` it cannot issue (or refuse) a communicator request.
    """
    return np.asarray(jax.device_get(x.addressable_data(0))
                      if hasattr(x, "addressable_data") else x)


def gate_htransform_vs_stored(
    psi_cQ_gamma, eps_cQ_gamma, data, mesh_xy, *,
    htransform_k_indices=None, stored_k_indices=None, log=print,
):
    """On-grid consistency gate at a Γ path point: htransform conduction
    ε vs the stored restart ε (max |Δ|), and the gauge-free per-k subspace
    overlap min singular value of ⟨ψ_ht|ψ_stored⟩ (bands × bands, spin+μ
    contracted).  Values printed; hard-fails only on gross breakage (>0.5
    overlap loss / >0.1 Ry ε drift) — htransform accuracy is
    centroid-count-governed and reported, not silently trusted.

    ψ IS NOT GATHERED, and ε is not either: both are contracted inside
    :func:`_gate_stats_on_device`, which returns replicated results, so the
    host side of this gate issues NO collective at all.  That function's
    docstring carries the measurement — a μ-sharded host gather here is an
    ``O(N_μ)`` all-gather per process inside a DIAGNOSTIC, and at P=16 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` it does not work at any of
    three warm-ups.

    A lower-level diagnostic with two internally coherent native k lattices
    may pass ``htransform_k_indices`` and ``stored_k_indices`` as the aligned
    exact intersection returned by
    :func:`symmetry_maps.common_uniform_grid_indices`.  This cannot repair a
    bundle whose energy and wavefunction tables have different k axes; that
    is refused before either index table is used.  The production
    ``bse_k_grid`` path compares every fine-grid row because its general
    loader has already densified the complete BSE bundle.
    """
    paired = (htransform_k_indices is not None, stored_k_indices is not None)
    if paired[0] != paired[1]:
        raise ValueError(
            "htransform@Gamma gate requires htransform_k_indices and "
            "stored_k_indices together.")
    n_ht_eps = int(eps_cQ_gamma.shape[0])
    n_ht_psi = int(psi_cQ_gamma.shape[0])
    n_st_eps = int(data["eps_c"].shape[0])
    n_st_psi = int(data["psi_c_X"].shape[0])
    if n_ht_eps != n_ht_psi or n_st_eps != n_st_psi:
        raise ValueError(
            "htransform@Gamma gate found an internally inconsistent BSE "
            "bundle: each energy table and its wavefunction table must share "
            "one k axis, but htransform has "
            f"eps={n_ht_eps}, psi={n_ht_psi} and stored/data has "
            f"eps={n_st_eps}, psi={n_st_psi}.  A bse_k_grid run with --eqp "
            "can cause exactly this if coarse restart energies are applied "
            "after psi was densified; apply the QP ladder before "
            "densification, or use WFN_qp.h5 without the redundant --eqp.")
    if paired[0]:
        ht_idx = np.asarray(htransform_k_indices, dtype=np.int32)
        st_idx = np.asarray(stored_k_indices, dtype=np.int32)
        if (ht_idx.ndim != 1 or st_idx.ndim != 1
                or ht_idx.shape != st_idx.shape or ht_idx.size == 0):
            raise ValueError(
                "htransform@Gamma gate matched-k indices must be nonempty "
                f"aligned vectors; got {ht_idx.shape} and {st_idx.shape}.")
        n_ht = n_ht_eps
        n_st = n_st_eps
        if (np.any(ht_idx < 0) or np.any(ht_idx >= n_ht)
                or np.any(st_idx < 0) or np.any(st_idx >= n_st)):
            raise ValueError(
                "htransform@Gamma gate matched-k index outside its native "
                f"table: htransform rows={n_ht}, stored rows={n_st}, "
                f"htransform range=[{int(ht_idx.min())},{int(ht_idx.max())}], "
                f"stored range=[{int(st_idx.min())},{int(st_idx.max())}].")
        log(f"  [gate] htransform@Gamma native-grid intersection: "
            f"{ht_idx.size} matched k point(s) from htransform "
            f"{n_ht} and stored {n_st}")
    else:
        ht_idx = st_idx = None
        if int(eps_cQ_gamma.shape[0]) != int(data["eps_c"].shape[0]):
            raise ValueError(
                "htransform@Gamma gate received different native k-grid "
                f"sizes ({int(eps_cQ_gamma.shape[0])} transformed vs "
                f"{int(data['eps_c'].shape[0])} stored) without the canonical "
                "common-grid index map.")
    nc = int(data["n_cond"])
    d_dev, d_idx_dev, G_dev = _gate_stats_on_device(
        eps_cQ_gamma, data["eps_c"], psi_cQ_gamma, data["psi_c_X"],
        nc, mesh_xy, htransform_k_indices=ht_idx,
        stored_k_indices=st_idx)
    d_eps = float(_local(d_dev))
    d_eps_idx = float(_local(d_idx_dev))
    G = _local(G_dev)
    smin = 1.0
    k_worst = -1
    sv_worst = None
    for k in range(G.shape[0]):
        sv = np.linalg.svd(G[k], compute_uv=False)
        if float(sv.min()) < smin:
            smin, k_worst, sv_worst = float(sv.min()), k, sv
    d_meV = d_eps * RY2EV * 1e3
    d_meV_idx = d_eps_idx * RY2EV * 1e3
    log(f"  [gate] htransform@Γ vs stored: max|Δε_c| = {d_meV:.6f} meV "
        f"(order-matched; this is the accuracy number), "
        f"conduction-subspace overlap min-sval = {smin:.4f}")
    if sv_worst is not None:
        # The WHOLE spectrum at the worst k, because the min alone is not
        # diagnostic: svals all ≈ s < 1 is a normalisation/weight error, while
        # svals ≈ (1, …, 1, s) is one genuinely missing subspace direction —
        # the window boundary cutting a degenerate multiplet.  Different bugs,
        # different fixes, and the gate used to print a number that could be
        # either.
        where = (f"matched row {k_worst}, htransform k={int(ht_idx[k_worst])}, "
                 f"stored k={int(st_idx[k_worst])}"
                 if ht_idx is not None else f"k={k_worst}")
        log(f"  [gate] overlap svals at the worst k ({where}): "
            f"{np.array2string(np.asarray(sv_worst), precision=4)}")
    log(f"  [gate] band-index-wise |Δε_c| = {d_meV_idx:.3f} meV — this is "
        f"NOT an accuracy figure.  htransform returns energies ascending and "
        f"the restart stores them in DFT-band order, so QP band reordering "
        f"alone produces a large value here.  Compare the order-matched "
        f"number above.")
    # The min-sval (subspace overlap) does NOT see energy corruption: an
    # over-packed interp window keeps the ψ_c SPAN (min-sval healthy) while the
    # recovered conduction ENERGIES drift by ~eV (640c/nband=80: min-sval 0.86
    # but on-grid |Δε_c| ~955 meV — 10_lorrax_exciton_bands_80interp_8v8c).  So
    # the ENERGY gate is the authoritative interp-basis check, tightened here.
    if d_meV > 20.0:
        log(f"  [warn] on-grid conduction ENERGY error {d_meV:.1f} meV ≫ the "
            f"~1-2 meV htransform floor.  Inspect the all-coarse transformed-"
            f"energy and whole-state QRCP projection receipts; then reduce "
            f"nband toward the BSE window or raise the registered QRCP search "
            f"ceiling only if those receipts identify missing state space.")
    assert d_eps < 0.05 and smin > 0.5, (
        f"htransform conduction cache grossly inconsistent with the stored grid "
        f"(max|Δε_c|={d_meV:.1f} meV, min-sval={smin:.3f}) — interp basis broken. "
        f"Check the all-coarse transformed-energy and whole-state QRCP "
        f"projection receipts (historical window finding: "
        f"runs/MoS2/04_mos2_12x12_bands_2026-07-18/"
        f"10_lorrax_exciton_bands_80interp_8v8c).")
    return d_eps, smin


# ===========================================================================
# driver
# ===========================================================================
_PROBE_TICK = None      # instrument: set by an out-of-tree profiling probe


def rerun_check_enabled(args) -> bool:
    """Is the diagnostic warm re-run of the solve scan switched on?

    The re-run is a pure reproducibility assert: it re-solves the ENTIRE Q
    scan a second time and compares the two tables.  It buys no physics, and
    it was measured at **37.7 % of driver wall** at P=4 / 41 Q (and 38.1 % at
    P=64 / 91 Q, job 7882533) — the single largest row in the stage table.
    That is why the default flipped OFF on 2026-08-08 and the check now has to
    be asked for, with ``--rerun-check``.

    ``--skip-rerun-check`` predates the flip and is still accepted, because
    the campaign harnesses and the archived launch recipes pass it.  It now
    *names* the default instead of changing it, and it still wins if both
    flags are given: a flag whose name says "skip" must never switch the
    re-run on.

    This predicate is the ONE place the decision is taken — the driver calls
    it rather than inlining the boolean, so that
    ``tests/test_exciton_bands_rerun_default.py`` gates the expression the
    driver actually evaluates rather than a copy of it.
    """
    if getattr(args, "skip_rerun_check", False):
        return False
    return bool(getattr(args, "rerun_check", False))


def build_parser():
    """The driver's argparse parser, built where a test can reach it — so a
    flag's default is observable without running a full solve."""
    import argparse
    # ``eigh_backend`` + ``use_low_mem_eigh`` are ONE axis with ONE
    # resolver, and the CLI vocabulary is the resolver's own list rather
    # than a hand-copied tuple that drifts (it had: this flag accepted
    # auto|off|cusolvermp|slate while distrib_la had grown ``distributed``
    # and ``scalapack``, so the CPU distributed eigh was unreachable from
    # here).  Function-local because gw_config parses decks and this
    # module is imported by things that do not.
    from gw.gw_config import (
        distrib_la_batched_route_choices,
        eigh_backend_choices,
    )

    ap = argparse.ArgumentParser(allow_abbrev=False,
        description="Exciton bandstructure E_S(Q) along a K_POINTS crystal_b path")
    ap.add_argument("-i", "--input", required=True,
                    help="cohsex.in with a K_POINTS crystal_b Q-path block")
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-cond", type=int, default=4)
    ap.add_argument("--band-degeneracy", choices=MODES, default=DEFAULT_MODE,
                    help="what to do when --n-val/--n-cond cut a degenerate "
                         "multiplet (Kramers pair under SOC+TRS, any irrep "
                         "of dimension > 1): 'strict' (the default since "
                         "2026-08-10) refuses, naming the counts that would "
                         "work; 'snap' widens the window OUTWARD to the "
                         "multiplet boundary and says so loudly; 'off' "
                         "proceeds on the cut multiplet.  Half a multiplet "
                         "breaks the exciton multiplets by an amount with no "
                         "convergence parameter (Si 12x12 SOC: eV-scale "
                         "off-grid), and a widened window is a different "
                         "calculation, so widening is something you ask for.")
    ap.add_argument("--degeneracy-tol-ry", type=float,
                    default=DEGENERACY_TOL_RY, dest="degeneracy_tol_ry",
                    help=f"'same multiplet' tolerance in Ry (default "
                         f"{DEGENERACY_TOL_RY:.4e} = 1 meV; the smallest "
                         f"between-pair boundary gap measured on the Si "
                         f"12x12 SOC deck was 5.9 meV)")
    ap.add_argument("--q-per-segment", type=int, default=16,
                    dest="q_per_segment",
                    help="MINIMUM number of Q the driver puts on each "
                         "K_POINTS crystal_b segment (default 16).  It is a "
                         "FLOOR on the deck's own per-segment counts, not an "
                         "override: a segment the deck asks for MORE points "
                         "on keeps its own count, and '--q-per-segment 1' is "
                         "the identity, which is the only way to get the "
                         "deck's counts verbatim.  The default exists "
                         "because E_S(Q) along a high-symmetry path is a "
                         "BANDSTRUCTURE: the decks in tree carried counts of "
                         "1 and 2, which drew Γ-X-W-L-Γ-Σ with eight "
                         "diagonalisations and straight lines between them.  "
                         "Each extra Q is one more row of the SAME compiled "
                         "solve scan (no extra compile), so density is "
                         "cheap; it is the htransform ψ_c(k+Q) cache that "
                         "grows linearly, and that is what to watch on a "
                         "densified (bse_k_grid) bundle.")
    ap.add_argument("--n-eig", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=40,
                    help="block-Lanczos iterations (Krylov = block·iter)")
    ap.add_argument("--vq-mode", choices=("interp", "refit", "both", "ongrid"),
                    default="interp")
    ap.add_argument("--eqp", type=str, default=None,
                    help="BGW-format eqp1.dat (the one LORRAX's GW writes): "
                         "run the whole BSE on QUASIPARTICLE energies instead "
                         "of DFT.  Both legs are corrected — the stored "
                         "valence/conduction eps from the restart AND the "
                         "htransform's enk_sigma, so the interpolated "
                         "eps_c(k+Q) is a QP band and the on-grid gate "
                         "compares QP against QP.")
    ap.add_argument("--refit-points", type=str, default=None,
                    help="comma list of path indices to refit "
                         "(vq-mode=both; default ~5 evenly spaced)")
    ap.add_argument("--refit-window", choices=("zeta", "bse"), default="zeta",
                    help="which band window the per-Q exchange REFIT fits "
                         "zeta' on.  'zeta' (default, unchanged): the "
                         "producer's own zeta-fit window, so the refit "
                         "reproduces the stored V_qmunu tiles and is "
                         "certified BY that tile null. A wide producer window "
                         "also enlarges the whole-state QRCP fit; inspect its "
                         "search/projection receipts.  "
                         "'bse': fit zeta' on the DECK's window (the BSE "
                         "window plus its conduction guards).  The exchange "
                         "only ever contracts BSE-window pairs, so that "
                         "window carries every pair density the kernel asks "
                         "for -- but zeta' is NOT the producer's zeta, the "
                         "stored tiles will NOT come back, and the "
                         "certification moves up to the CONTRACTED object: "
                         "on-grid BSE eigenvalues through the refit route "
                         "against the same eigenvalues through the stored "
                         "tiles, at "
                         f"{REFIT_CERT_TOL_MEV:g} meV.")
    ap.add_argument("--refit-guard-bands", type=int, default=None,
                    metavar="N",
                    help="guard bands the refit's htransform (fH) window "
                         "carries ABOVE the window zeta' is fitted on "
                         f"(default {vq_interp.REFIT_N_GUARD_DEFAULT}).  THE "
                         "TWO-WINDOW CONTRACT: f(eps) is identically zero for "
                         "eps >= max_k eps of fH's OWN top band, so pinning "
                         "the two windows together asks compute_wfns_fi for "
                         "exactly the bands fH cannot represent -- the top "
                         "band vanishes at the k that defines the shift and a "
                         "whole shoulder below it carries ~1%% of fH's "
                         "weight, so eigh returns arbitrary null-space "
                         "directions there (measured: on-grid tile null 1.267 "
                         "with zero guards, against a 5.0e-02 bracket).  The "
                         "guards move that shoulder onto bands nobody reads; "
                         "zeta' is still fitted on the producer's window "
                         "exactly.  Costs Galerkin capacity "
                         "(n_mu*n_s >= nk*(nb_zeta+N)) and needs the WFN to "
                         "carry the bands.  0 is the RED TWIN and must fail "
                         "the tile null.")
    ap.add_argument("--cert-grade", choices=tuple(CERT_TOL_BY_GRADE),
                    default="reference",
                    help="which NAMED tolerance the --refit-window=bse "
                         "contracted certification is held to.  'reference' "
                         f"(default, unchanged): {REFIT_CERT_TOL_MEV:g} meV, "
                         "the gate a published NUMBER has to clear.  "
                         "'visualization': "
                         f"{CERT_TOL_VISUALIZATION_MEV:g} meV, for a "
                         "deliverable that is a PICTURE — an exciton band "
                         "structure whose features live at tens of meV — on "
                         "a route whose own floor is ~0.9 meV.  The "
                         "certification still RUNS and still REFUSES above "
                         "the grade it was given; on a pass the grade and "
                         "the certified number are stamped into the "
                         "provenance line, the .dat header and the .png.  "
                         "These two are the whole surface: both are module "
                         "constants (CERT_TOL_BY_GRADE) and there is no flag, "
                         "env var or deck key that produces a third number.")
    ap.add_argument("--refit-r-chunk", type=int, default=2048,
                    help="r-grid chunk of the refit Z build (vq_interp."
                         "refit_prepare); the per-chunk pair-density temp "
                         "is (nk, nb, nb, r_chunk) c128 — shrink on dense "
                         "k-grids (e.g. 512 at 12x12) to fit device memory")
    ap.add_argument("--a-band", type=int, default=None,
                    help="htransform window-RELATIVE band index whose "
                         "bandwidth sets the f-transform width a = 4*BW "
                         "(bandstructure.htransform._f_params_from_energies). "
                         "Default (None) matches the standard SP driver: a "
                         "from the top band of the full window.  Set this to a "
                         "low-bandwidth conduction band only if the selected "
                         "conduction caches land in the f'->0 compression zone "
                         "(a large default a from a dispersive top guard band "
                         "can collapse off-grid eps_c(k+Q) by eV).")
    ap.add_argument("--alpha", type=float, default=vq_interp.ALPHA)
    ap.add_argument("--eps-tik", type=float, default=vq_interp.EPS_TIK)
    ap.add_argument("--head-minibz-average", action="store_true", default=None,
                    help="Per-Q mini-BZ Coulomb head cell-averaging: replace "
                         "the finite-Q exchange head POINT value v(Q+G*) with "
                         "the mini-BZ cell average <v_LR(Q+G*)>_mBZ (fixes the "
                         "4-13%% near-Γ/zone-boundary head error, "
                         "arbitrary_q_bse.md §16.4).  Overrides the cohsex.in "
                         "``head_minibz_average`` key; default (unset) uses it.")
    ap.add_argument("--dipole", default=None,
                    help="dipole.h5 for the mini-BZ head TENSOR (only read "
                         "when head_minibz_average is ON).  Default: "
                         "dipole.h5 beside the input file.  The head's "
                         "q-linear coefficient IS this dipole, which is why "
                         "the cell-averaged head needs no d(zeta)/dq.")
    ap.add_argument("--eigh-backend", default=None,
                    choices=eigh_backend_choices(),
                    help="OVERRIDES the input-file ``eigh_backend`` key "
                         "(default: use the key, which defaults to auto).  "
                         "Hermitian eigensolver for BOTH distributed-eigh "
                         "sites: the coarse exchange tiles C_q (vq_interp) and "
                         "the htransform fH_q (bse_setup).  auto|off = the "
                         "q-BATCHED native path (every device solves its own "
                         "q-shard).  cusolvermp|slate route ONE tile at a time "
                         "through the distributed-linalg FFI — the regime "
                         "where a single matrix no longer fits on one device "
                         "(a WIDE fH band window), at the cost of nq "
                         "sequential solves.  Needs a square mesh and one JAX "
                         "process per device.")
    ap.add_argument(
        "--distrib-la-batched-route", default=None,
        choices=distrib_la_batched_route_choices(),
        help="OVERRIDES the input-file distrib_la_batched_route key for "
             "both htransform and coarse-V batched linalg. auto preserves "
             "the backend's distributed route; batch_reshard moves q onto "
             "the mesh and runs local whole-matrix JAX linalg.")
    ap.add_argument("--px", type=int, default=None,
                    help="mesh rows; default = the run's square startup mesh")
    ap.add_argument("--py", type=int, default=None,
                    help="mesh columns; must equal --px, and px*py must be "
                         "the job's device count")
    ap.add_argument("--rerun-check", action="store_true",
                    help="RUN the diagnostic warm re-run of the solve scan "
                         "(reproducibility assert + dispatch-only per-Q "
                         "timing).  OFF by default since 2026-08-08: the "
                         "re-run is a full second solve pass and measured "
                         "37.7%% of driver wall at P=4/41 Q.  Ask for it when "
                         "a configuration is new or under suspicion.")
    ap.add_argument("--skip-rerun-check", action="store_true",
                    help="skip the diagnostic warm re-run.  This is now the "
                         "DEFAULT and the flag is a no-op kept so existing "
                         "harnesses and launch recipes keep parsing; it wins "
                         "over --rerun-check if both are passed.")
    ap.add_argument("--extra-q", type=str, default=None,
                    help="';'-separated extra fractional Q (each 'x,y,z') "
                         "appended to the path and solved in the SAME scan "
                         "(one compile, no extra cost per point beyond the "
                         "scan row).  They are NOT plotted; they are written "
                         "to the .dat with mode 'extra'.  The intended use is "
                         "the reference-free SYMMETRY test: pass the "
                         "point-group images of a few off-grid path Q — "
                         "symmetry-equivalent Q must give identical E_S, so "
                         "any disagreement is off-grid interpolation error, "
                         "and its size localises the leg.")
    ap.add_argument("--out-prefix", type=str, default="exciton_bands")
    ap.add_argument(
        "--report-file", type=str, default=None,
        help="human-readable report (default: <out-prefix>.out)")
    ap.add_argument("--w-coarse-grid", type=str, default=None,
                    help="NX,NY,NZ — sample the screened W on this COARSE BZ "
                         "sub-grid (a divisor of the WFN/BSE grid), then "
                         "zero-pad W_R back to the fine grid (exact trig "
                         "interpolation) for the direct term.  Enables cheap "
                         "coarse-W + fine exciton sampling.  Default (unset) "
                         "keeps the native fine W byte-identical.")
    ap.add_argument("--w-head-densify", type=str, default=None,
                    choices=("c1", "legacy"),
                    help="How W's Γ head crosses the coarse→fine densifier. "
                         "'c1' (default) splits it off first and re-attaches "
                         "it analytically per fine q — the head is never "
                         "interpolated, which is what BerkeleyGW's kernel.x / "
                         "intkernel split exists to guarantee.  'legacy' lets "
                         "the Kronecker-delta head ride through the "
                         "trigonometric interpolant: that is the documented "
                         "defect and this arm exists ONLY as the A/B control "
                         "that prices it.  No effect without "
                         "--w-coarse-grid.")
    ap.add_argument("--w-head-gamma-cell", type=str, default="fine",
                    choices=("fine", "coarse"),
                    help="Which mini-BZ cell the re-attached Γ head is "
                         "averaged over.  'fine' (default) is correct.  "
                         "'coarse' is the design's RED TWIN — invisible when "
                         "the grids are equal, and it breaks the head sum "
                         "rule everywhere else.  For gate work only.")
    return ap


def _gamma_only_grid(kgrid, value):
    """A head-scalar grid that is ``value`` at Γ and zero everywhere else.

    The array shape :func:`gw.head_densify.attach_head_channel` consumes, used
    here to SUBTRACT the head the loader already injected (pass a negative
    value) so the densifier's operand is the body alone.  Spelled through the
    same attach helper rather than as a bespoke ``.at[...,0,0,0].add()`` so the
    detach and the re-attach cannot disagree about which slice, which
    projector, or which 1/Ω the head lives on — a mismatched pair here would
    leave a residual rank-one term with no shape error to catch it.
    """
    S = np.zeros(tuple(int(s) for s in kgrid), dtype=np.float64)
    S[0, 0, 0] = float(value)
    return S


def _resolve_native_w_head(restart_file, input_file, wfn, *, log=print):
    """The ``whead`` the loader injected into this NATIVE-grid restart's W.

    ``--w-coarse-grid`` decimates a natively fine W, so the head sitting in
    ``data["W_q"][:,:,0,0,0]`` is the FINE cell's average — the number C1 has
    to remove before sub-sampling and put back after densifying.  Returns
    ``None`` (with a reason logged) when there is no head to move, in which
    case the caller runs the plain densifier and nothing is lost.
    """
    try:
        import h5py
        with h5py.File(restart_file, "r") as f:
            whead_restart = (np.asarray(f["whead"][:], dtype=np.complex128)
                             if "whead" in f else None)
            w0_ready = bool(f["W0_qmunu"].attrs.get("W0_ready", False)
                            if "W0_qmunu" in f else False)
    except Exception as exc:
        log(f"[coarse-W] cannot read the restart's head ({exc}); "
            f"falling back to w_head_densify=legacy for this run")
        return None
    if not w0_ready:
        log("[coarse-W] the restart carries no ready screened W0, so the "
            "loader injected no whead and there is no head to split off; "
            "the densifier operand is already head-free")
        return None
    _vhead, whead, cell_volume, _src = _resolve_head_params(
        input_file, None, whead_restart, float(wfn.cell_volume))
    if whead is None or cell_volume is None:
        log("[coarse-W] no whead resolved (no deck override, no restart "
            "scalar); the densifier operand is already head-free")
        return None
    return {"whead": float(complex(whead[0]).real),
            "cell_volume": float(cell_volume)}


def main(argv=None):
    # ``resolve_eigh_backend`` stays a function-local import for the reason
    # given in ``build_parser``: gw_config parses decks, and this module is
    # imported by things that do not.
    from gw.gw_config import (
        resolve_distrib_la_batched_route,
        resolve_eigh_backend,
    )

    ap = build_parser()
    args = ap.parse_args(argv)

    report_path = os.path.abspath(
        args.report_file if args.report_file else args.out_prefix + ".out")
    debug = debug_print_enabled()
    report = ScientificProductionReport(
        report_path, runtime=RUNTIME, debug=debug, stdout=rank0_print,
        driver_name="bse.exciton_bands",
        calculation_name="exciton bandstructure")
    production_stdout = ProductionStdout(
        debug=debug, rank=RUNTIME.process_index,
        warning_fn=report.legacy_print)
    production_stdout.install()
    report.stdout = rank0_print if debug else production_stdout.emit
    log = report.legacy_print
    report.begin(input_file=args.input)
    report.architecture(mesh_role="BSE transition axes X x Y")
    report.pathways((
        "Hamiltonian    : finite-Q Tamm-Dancoff BSE, D_Q + V_Q - W",
        f"Exchange V_Q   : {args.vq_mode} "
        "(other choices: interp, refit, both, ongrid)",
        "Screened W    : " + (
            f"coarse {args.w_coarse_grid} grid, trigonometric densification"
            if args.w_coarse_grid else "native BSE k grid"),
        f"Eigensolver    : block Lanczos; block={int(args.block_size)}; "
        f"maximum iterations={int(args.max_iter)}",
        "Energy source : " + (
            f"quasiparticle corrections from {abs_path(args.eqp)}"
            if args.eqp else "DFT eigenvalues from the WFN"),
        "Refit cert     : " + policy(
            args.cert_grade, tuple(CERT_TOL_BY_GRADE)),
        "Warm rerun     : " + (
            "enabled as a reproducibility diagnostic"
            if rerun_check_enabled(args) else
            "off (pass --rerun-check to enable)"),
    ))
    # ---- Stage timing -----------------------------------------------------
    # DELIBERATELY a driver-local two-column table (``name  seconds``) and NOT
    # ``common.timing.report``: eight live campaign harnesses parse this table
    # with ``grep htransform_psi_cQ | awk '{print $2}'``, and the collector's
    # table puts COUNT in column 2.  Switching formats would leave every one of
    # them reading a small integer as a wall time and reporting it as green —
    # the void-instrument failure mode this campaign has already paid for nine
    # times.  What IS fixed here is completeness: every phase between
    # ``t_wall`` and the report now has a row, and the table closes with an
    # explicit ``(untimed)`` residual so it always sums to TOTAL.  A reader can
    # therefore tell "this accounting is complete" from "43% of the wall is
    # somewhere else" without doing arithmetic — which is exactly what went
    # wrong when job 7882533's 4633 s read as two phases and ~2000 s of
    # mystery (the mystery was ``solve_scan_cold``, a fully-executed pass that
    # WAS in the table; the reader summed the wrong two rows).
    t_wall = time.time()
    timers: dict[str, float] = {}

    def tick(name, t0):
        timers[name] = timers.get(name, 0.0) + (time.time() - t0)
        if _PROBE_TICK is not None:                       # instrument:
            _PROBE_TICK(name, timers[name])               # instrument:

    # Work done BEFORE main(): this module's ``initialize_communicator_stack()`` and every import
    # under it.  75.0 s to first output on a cold Frontera node vs 2.1 s warm
    # (job 7881949), and previously outside the table's clock entirely, so the
    # printed TOTAL could be a minute short of the job's own wall.
    from common.timing import process_elapsed_s as _proc_elapsed
    _pre_main = _proc_elapsed()
    if _pre_main is not None:
        timers["imports_and_runtime"] = _pre_main
        t_wall -= _pre_main
    t_prologue = time.time()

    # Rank-0 I/O guard.  In a multi-node run (one process per GPU) the file
    # writes (.dat / .png) and progress prints must run on process 0 ONLY —
    # otherwise 16 processes race on the same paths.  Non-I/O host numpy (Q
    # path / k-roll construction, the per-Q mini-BZ head QMC) runs redundantly
    # on every process; it is deterministic, so all processes agree and the
    # sharded solve consumes identical operands.  ``log`` is the rank-0 print;
    # it is also threaded into the heavy helpers as ``log_fn`` so their
    # progress is not emitted 16×.
    _rank0 = jax.process_index() == 0

    # Omitted --px/--py = the run's canonical square mesh (RUNTIME.mesh by
    # identity, not a twin), not 1x1; a given shape must BE that mesh.  The
    # log line below then states the mesh in use, because args.px/args.py
    # carry the resolved shape rather than the request.
    mesh_xy = create_mesh_xy_from_flags(args.px, args.py)
    args.px, args.py = tuple(int(n) for n in mesh_xy.devices.shape)
    log(f"[dist] jax.device_count()={jax.device_count()} "
        f"process_count()={jax.process_count()} "
        f"local_device_count()={jax.local_device_count()}; "
        f"mesh_xy.shape={dict(mesh_xy.shape)} (px={args.px}, py={args.py})")

    # ── Q path from the ONE K_POINTS crystal_b machinery ─────────────────
    from gw.gw_config import linalg_resolution, read_lorrax_input
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi

    params = read_lorrax_input(args.input)
    # CLI backend flags remain debug overrides of the implementation selected
    # by the parser-cached public layout profile.
    args.eigh_backend = resolve_eigh_backend(
        params, override=args.eigh_backend)
    args.distrib_la_batched_route = resolve_distrib_la_batched_route(
        params, override=args.distrib_la_batched_route)
    _use_low_mem_eigh = linalg_resolution(params).layout == "distributed"
    if not params.get("kpoints_crystal_b"):
        raise ValueError(f"{args.input} has no K_POINTS crystal_b block — "
                         "the exciton Q path comes from it (same format as "
                         "the htransform bandstructure driver)")
    # Sampling density BEFORE the path is generated: a floor on the deck's
    # per-segment counts, default 16, so the default output is a
    # bandstructure rather than a line drawing between corners.  See
    # apply_q_per_segment for why this is a floor and not an override.
    apply_q_per_segment(params, args.q_per_segment, log=log)

    # Resolve and print the numerical environment before either htransform
    # Galerkin setup can emit progress. Whole-state QRCP owns basis selection;
    # the only eigensolver plan here is the fine-k plan executed downstream.
    _wfn_name = str(params["wfn_file"])
    if not os.path.isabs(_wfn_name):
        _wfn_name = os.path.join(
            os.path.dirname(os.path.abspath(args.input)), _wfn_name)
    wfn, sym = ht.setup_wfn_and_sym(_wfn_name, mesh_xy=mesh_xy)
    from distrib_la import plan as _linalg_plan
    _fine_plan = _linalg_plan(
        "eigh", mesh_xy, backend=args.eigh_backend, n=None,
        batched_route=args.distrib_la_batched_route)
    _basis_multiplier = ht.resolve_galerkin_rank_multiplier(
        params.get("htransform_rank_multiplier", 20.0))
    restart_file = _find_restart_file(args.input)
    report.environment(wfn=wfn, lines=(
        f"Restart tensors: {abs_path(restart_file)}",
        "Transition data: distributed band and centroid blocks on X x Y",
        "Galerkin basis: whole-state randomized QRCP; search ceiling "
        f"{_basis_multiplier:g} x fitted bands; qr_eps="
        f"{float(params.get('htransform_qr_eps', 1.0e-3)):.3e}; seed="
        f"{int(params.get('htransform_qrcp_seed', 0))}; no Gram eigensolve",
        f"Fine-k eigensolve: {_fine_plan.requested} -> "
        f"{_fine_plan.backend}; matrix extent follows retained Galerkin rank",
        f"Batched LA schedule: {_fine_plan.requested_batched_route} -> "
        f"{_fine_plan.batched_route}",
        "LA alternatives : native JAX; cuSOLVERMp on GPU; SLATE or "
        "ScaLAPACK on CPU. Batched LA is a schedule, not a backend",
    ))
    report.sampling(wfn=wfn, sym=sym)
    stage_progress = LoopProgress(
        7, report.progress, title="exciton-band calculation",
        item_name="major stage", max_updates=7)
    stage_progress.start()

    # Everything above — the startup call, jax.distributed, mesh creation and its
    # MPI clique warm-up, the input parse — is the driver prologue.  Named
    # rather than left in the residual: at P=16 ``jax.distributed`` init alone
    # measured 43.8 s, and on a cold node the software stack boots off Lustre
    # for 75 s (job 7881949).  Neither is physics and neither had a row.
    tick("prologue", t_prologue)

    # ── load the Q-independent BSE data (production loader) ──────────────
    t0 = time.time()
    _bse_grid_rank_records = []
    _bse_grid_quality_records = []
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond,
        mesh_xy=mesh_xy, input_file=args.input, inject_head=True,
        # ``refit`` NEEDS the stored exchange tensor as much as ``ongrid``
        # does, and for a stricter reason: ongrid CONSUMES it, refit is
        # CERTIFIED AGAINST it — the tile null under --refit-window=zeta, the
        # contracted on-grid eigenvalue gate under =bse.  Both read
        # ``V_q_full`` (or, after a bse_k_grid densification, the coarse tiles
        # the densifier parks in ``V_q_coarse``, which is that same array), so
        # a refit run that did not ask for it reached its own gate and refused
        # itself for the bundle "carrying none" — measured, job 56612363 step
        # .60, the first time the refit ever got that far.
        load_v_full=(args.vq_mode in ("ongrid", "refit")),
        degeneracy_mode=args.band_degeneracy,
        degeneracy_tol_ry=args.degeneracy_tol_ry,
        distrib_la_batched_route=args.distrib_la_batched_route,
        htransform_a_band=args.a_band,
        htransform_rank_record_fn=_bse_grid_rank_records.append,
        htransform_quality_record_fn=_bse_grid_quality_records.append)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
    nv_pad, nc_pad = int(data["n_val_pad"]), int(data["n_cond_pad"])
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    # ── QP energies (--eqp) ───────────────────────────────────────────────
    # BOTH legs of the pair basis have to move together or the diagonal
    # D_Q = eps_c(k+Q) - eps_v(k) mixes QP conduction with DFT valence.  The
    # stored leg is re-sliced here (n_occ is RE-resolved on the corrected
    # energies, so a QP-driven gap change cannot mis-slice); the interpolated
    # leg is corrected below, right after ``initialize_wfns``.
    #
    # ``args.input`` is passed, and the eqp file is the IRREDUCIBLE WEDGE.
    # This used to pass ``input_file=None`` to reach a second code path that
    # matched full-BZ k to wedge blocks by mean-field energy to 0.01 eV, on
    # the stated grounds that "LORRAX's own GW writes eqp1.dat on the FULL
    # BZ".  It does not — ``gw_output.py`` subsets through ``kirr_to_kfull``
    # and ``gw_jax.py`` passes ``wfn.kpoints`` — so the assertion that
    # branch was avoiding is the one that PASSES, and the heuristic was
    # bespoke unfolding standing in for the symmetry service.  Both are
    # gone; ``apply_eqp_corrections`` now unfolds through the service's
    # single ``star_broadcast`` adapter and ``input_file`` is required.
    enk_qp_full = None
    if args.eqp:
        # A QP WFN's psi and E are a MATCHED PAIR, and --eqp is a second,
        # DFT-band-labelled ladder.  Applying it on top overwrites the
        # canonical QP eigenvalues that produced the rotation and relabels the
        # rotated orbitals with the mean-field band ordering -- silently, and
        # with the right shapes throughout.  The shape guard further down sees
        # this only when a bse_k_grid densification has already moved psi to a
        # different k axis; without densification the two tables are the same
        # size and the overwrite is invisible.  Measured on the MoS2 run-82
        # parent smoke (JID 57269074 step .128): the deck selects WFN_qp.h5 and
        # the wrapper passes --eqp eqp1.dat.
        #
        # The refusal lives in ``bse_window`` beside the other eqp helpers and
        # ``_parse_wfn_path`` — one owner for "which WFN does this deck name",
        # and it asks the FILE's stamp rather than its name.
        refuse_eqp_on_a_qp_wfn(args.input, args.eqp)
        from .bse_io import (apply_eqp_and_reslice_bands, apply_eqp_corrections,
                             resolve_n_occ)
        import h5py as _h5py
        with _h5py.File(restart_file, "r") as _f:
            _enk_dft_full = np.asarray(_f["enk_full"][:])
        n_occ_in = resolve_n_occ(_enk_dft_full, input_file=args.input)
        data["eps_v"], data["eps_c"], n_occ_qp = apply_eqp_and_reslice_bands(
            restart_file, args.eqp, args.input, n_val, n_cond, n_occ_in,
            mesh_xy.devices.shape[0], mesh_xy.devices.shape[1],
            degeneracy_mode=args.band_degeneracy,
            degeneracy_tol_ry=args.degeneracy_tol_ry)
        if int(data["eps_c"].shape[0]) != int(data["psi_c_X"].shape[0]):
            raise ValueError(
                "exciton_bands: --eqp rebuilt energies on the native restart "
                f"grid ({int(data['eps_c'].shape[0])} k) after bse_k_grid "
                "had already densified the BSE wavefunctions to "
                f"{int(data['psi_c_X'].shape[0])} k.  This would mix coarse "
                "energies with fine-grid psi.  If wfn_file is WFN_qp.h5, "
                "remove --eqp: that file already carries the canonical QP "
                "eigenvalues paired with its rotated wavefunctions.  A "
                "mean-field WFN plus diagonal eqp corrections needs the eqp "
                "ladder applied inside the htransform densification, not "
                "patched onto its output here.")
        enk_qp_full = apply_eqp_corrections(
            _enk_dft_full, args.eqp, args.input)
        _shift_ev = (enk_qp_full - _enk_dft_full) * RY2EV
        log(f"  [eqp] {os.path.basename(args.eqp)}: n_occ={n_occ_qp}, "
            f"QP shifts min/max = {_shift_ev.min():+.4f} / {_shift_ev.max():+.4f} eV; "
            f"BSE runs on QUASIPARTICLE energies")
    if nv_pad > n_val:
        # The loader (and apply_eqp_and_reslice_bands, above) now write the
        # signed guard themselves, so this is no longer a repair — it is the
        # CHECK that they did.  Kept because this driver is where the wrong
        # number was first noticed; a silent regression to a zero ε pad puts
        # spurious transitions BELOW the exciton onset on every BSE driver,
        # not just this one, and this is the cheapest place that would see it.
        _pad_eps_v = jnp.asarray(data["eps_v"])[:, n_val:]
        _worst = float(jnp.max(_pad_eps_v.real))
        if _worst > -0.5 * PAD_EPS_GUARD_RY:
            raise ValueError(
                f"exciton_bands: loader returned an unguarded valence pad — "
                f"max eps_v over the {nv_pad - n_val} pad bands is {_worst:.3e} "
                f"Ry, expected <= {-0.5 * PAD_EPS_GUARD_RY:.3e}. A zero pad "
                f"here makes DeltaE = eps_c - 0 a spurious transition BELOW "
                f"every physical one. See bse_io.PAD_EPS_GUARD_RY.")
    tick("load_bse", t0)
    stage_progress.step()
    for receipt in _bse_grid_rank_records:
        report.spectral_compression(
            receipt, title="BSE-grid htransform spectral compression")
    for receipt in _bse_grid_quality_records:
        report.htransform_quality(
            receipt, title="BSE-grid htransform interpolation quality")

    # Densify/eqp already check this join on their dominating paths.
    if "V_q_coarse" not in data and not args.eqp:
        from file_io.qp_wfn import refuse_conflicting_qp_state_sources
        from .bse_window import _parse_wfn_path
        refuse_conflicting_qp_state_sources(
            wfn_path=_parse_wfn_path(args.input),
            state_artifact_path=restart_file,
            where="exciton_bands restart/htransform state")

    # ── which ISDF basis the htransform fits in ──────────────────────────
    # BEFORE initialize_wfns, because that call is the one that reads the
    # centroid table.  On a plain bundle this returns the deck's own path
    # and ``None``, and the program below is unchanged; on a downfolded one
    # it returns the PARENT's table and the column slice back to this
    # bundle's basis.  See resolve_isdf_basis for why those are different
    # questions and why the deck alone cannot answer the second.
    centroids_path, keep_idx = resolve_isdf_basis(
        restart_file, params, args.input, n_rmu_bundle=n_rmu, log=log)
    params["centroids_file"] = centroids_path

    # ── htransform setup + Q path ────────────────────────────────────────
    t0 = time.time()
    # The sole published whole-state fit is born directly at the downfold's
    # ordered child rows.  ``0`` is only an archived spelling of the default
    # 20*nb QRCP search ceiling, not an exact parent-fit alternate.  Therefore
    # no parent-width B or projected-wavefunction cache is ever formed.
    _fit_subset = keep_idx
    _path_rank_records = []
    (wfn, sym, meta, _mesh, basis,
     enk_sigma) = ht.initialize_wfns(
         args.input, params, log, mesh_xy=mesh_xy,
         centroid_subset_idx=_fit_subset, wfn_sym=(wfn, sym),
         rank_record_fn=_path_rank_records.append,
         distrib_la_batched_route=args.distrib_la_batched_route)
    ctilde, B_at_mu = basis.ctilde, basis.basis_at_nodes
    if enk_qp_full is not None:
        # The interpolated leg.  ``initialize_wfns(eqp_file=...)`` is NOT used:
        # this driver has already parsed and symmetry-unfolded the columnar IBZ
        # eqp table once through ``bse_io.read_bgw_eqp``.  Parsing it again in
        # htransform would duplicate that ownership and risk a second ordering
        # convention.  Both stored and interpolated legs slice the same
        # ``enk_qp_full`` authority here.
        _b0 = int(wfn.nelec) - int(params["nval"])
        _b1 = int(wfn.nelec) + int(params["ncond"])
        enk_sigma = jnp.asarray(enk_qp_full[:, _b0:_b1].T)      # (nb, nk) Ry
        log(f"  [eqp] htransform enk_sigma <- QP bands [{_b0},{_b1})")
    kpath_frac, x_path, node_idx, node_labels, _gp = ht.initialize_kpath(
        wfn, params)
    Qpath = np.asarray(kpath_frac, dtype=np.float64)
    nQ_path = Qpath.shape[0]
    # --extra-q rows ride the SAME scan (extra xs rows, one compile).  They are
    # appended to Qpath so every Q-dependent stage — htransform caches, V_Q,
    # the solve — treats them exactly like path points; only the plot and the
    # path-distance axis stop at nQ_path.
    Q_extra = np.zeros((0, 3))
    if args.extra_q:
        Q_extra = np.array([[float(v) for v in seg.split(",")]
                            for seg in args.extra_q.split(";") if seg.strip()],
                           dtype=np.float64)
        Qpath = np.concatenate([Qpath, Q_extra], axis=0)
        log(f"  +{Q_extra.shape[0]} --extra-q point(s) appended to the scan")
    nQ = Qpath.shape[0]
    log(f"Q path: {nQ_path} path points (+{nQ - nQ_path} extra), nodes at "
        f"{list(map(int, node_idx))} labels {node_labels}")
    tick("htransform_setup", t0)
    stage_progress.step()
    if len(_path_rank_records) != 1:
        raise RuntimeError(
            "exciton_bands htransform returned "
            f"{len(_path_rank_records)} spectral receipts; expected one")
    report.spectral_compression(
        _path_rank_records[0],
        title="Exciton-path htransform spectral compression")
    report.heading("Exciton momentum path")
    report.emit(f"Path sampling  : {int(nQ_path)} plotted Q points; "
                f"{int(nQ - nQ_path)} extra diagnostic points")
    for inode, (idx, label) in enumerate(zip(node_idx, node_labels), start=1):
        point = Qpath[int(idx)]
        report.emit(
            f"  N{inode:02d}  {label or '-':>3}  "
            f"Q=({point[0]: .5f} {point[1]: .5f} {point[2]: .5f})  "
            f"path index {int(idx) + 1}")

    # ── conduction caches ψ_c(k+Q), ε_c(k+Q) for the whole path ──────────
    # FULL-BAND htransform basis — the single lever that removes the off-grid
    # window-cache ringing.  compute_wfns_fi builds fH from ALL bands in
    # ``ctilde`` (the entire loaded window = input nval+ncond) and only
    # RETURNS the sub-window [b_min, b_max); so a full-band ctilde gives a
    # full-band fH regardless of how few conduction bands the BSE keeps.  With
    # the standard driver's window (nband=40 = 26v+14c) the BSE conduction
    # bands [b_min, b_max) sit strictly INTERIOR, guarded above by the extra
    # conduction bands — every selection boundary stays off any near-
    # degenerate (Kramers) pair.  A SLIVER conduction window whose top
    # boundary cuts a near-degenerate pair instead rings 100-1000 meV off-grid
    # (05_htransform_spbands/gap_scan; Si degeneracy root-cause 73e58f79).
    #
    # But the interpolation window is TWO-SIDED: too small rings off-grid,
    # while a larger window asks the shared whole-state QRCP basis to represent
    # more states.  The all-coarse transformed-energy receipt and this
    # consumer's on-grid energy gate measure that representation directly;
    # centroid count is not a state-space rank proxy in the published route.
    # Keep only physically required bands plus guards unless those receipts
    # justify a larger window.
    t0 = time.time()
    nb_window = int(ctilde.shape[1])    # bands in the htransform fH (= input nval+ncond)
    nval_in = int(params["nval"])       # window-relative CBM index (VBM = nval_in-1)
    b_min, b_max = nval_in, nval_in + n_cond
    n_guard = nb_window - b_max         # conduction bands ABOVE the BSE selection
    if b_max > nb_window:
        raise ValueError(
            f"BSE conduction window [{b_min},{b_max}) exceeds the htransform "
            f"fH window ({nb_window} bands): raise nband in {args.input} to "
            f">= {b_max}, or drop --n-cond to <= {nb_window - nval_in}")
    # The htransform window is the SECOND place a band boundary is cut in this
    # driver, and it is cut in a different index space (window-relative, not
    # absolute), so the loader's snap does not automatically make it safe.
    # ``enk_sigma`` is (nb, nk) in the SAME window-relative indexing as
    # b_min/b_max, so the check is exact here.  Report-only: b_max is pinned to
    # the already-resolved n_cond, and widening it here would desynchronise the
    # conduction caches from the BSE window the loader sized.  The prose above
    # has warned about exactly this failure since the Si root-cause; this makes
    # it a measurement instead of a warning about a possibility.
    check_band_window(
        np.asarray(enk_sigma).T, b_min, b_max,
        tol_ry=args.degeneracy_tol_ry, mode=args.band_degeneracy,
        where="exciton_bands htransform conduction window", log=log)
    if n_guard < 4:
        log(f"  [warn] only {n_guard} conduction guard band(s) above the BSE "
            f"selection — a selection boundary near a Kramers pair can ring "
            f"off-grid; widen the input's ncond/nband (>= {b_max + 4} bands).")
    if n_guard > 16:
        log(f"  [warn] htransform fH spans {nb_window} bands with {n_guard} "
              f"conduction guards above the BSE window — a LARGE interp window "
              f"is not automatically more accurate.  Check the mandatory "
              f"all-coarse transformed-energy receipt and this driver's "
              f"on-grid energy gate; keep nband just above the BSE window "
              f"unless both remain controlled.")
    log(f"  full-band htransform: fH over {nb_window} bands "
        f"({nval_in}v + {nb_window - nval_in}c); BSE conduction "
        f"[{b_min},{b_max}) = {n_cond} band(s) + {n_guard} guard(s)")
    _nelec = int(wfn.nelec)
    report.bands((
        f"Electrons      : {float(getattr(wfn, 'num_electrons', _nelec)):.5f}; "
        f"occupied-band boundary = {_nelec}",
        f"BSE valence    : {band_range(_nelec - n_val, _nelec)}",
        f"BSE conduction : {band_range(_nelec, _nelec + n_cond)}",
        f"htransform fit : {band_range(_nelec - nval_in, _nelec - nval_in + nb_window)} "
        f"({int(n_guard)} conduction guard bands)",
        f"Transition size: {int(nc_pad * nv_pad * nk)} padded; "
        f"{int(n_cond * n_val * nk)} physical",
        f"Centroid basis : {int(n_rmu)} interaction centroids",
        "Centroid table : " + (
            f"parent htransform sites {abs_path(centroids_path)}; "
            "the downfold selection is stored in the restart"
            if keep_idx is not None else abs_path(centroids_path)),
    ))
    k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                                  np.arange(nkz) / nkz, indexing="ij"),
                      axis=-1).reshape(-1, 3)
    q_list = (Qpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)
    # kgrid_co is the COARSE grid that ``ctilde`` lives on (= the WFN/restart
    # grid, from ``meta``), NOT ``(nkx,nky,nkz)`` — those come from ``data`` and
    # are the FINE grid after a ``bse_k_grid`` coarse→fine densification, which
    # ``k_frac`` above correctly uses for the fine BSE k-sum.  ``build_fH_R``
    # ifft-reshapes ctilde's k-axis into ``kgrid_co``, so it MUST equal
    # ``prod(coarse)``; passing the fine grid crashes (36-k ctilde ≠ 12×12).
    # No-op when bse_k_grid is unset (data grid == meta grid).
    kgrid_co_ct = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    _path_quality_records = []
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=kgrid_co_ct, band_window_fi=(b_min, b_max),
        mesh_xy=mesh_xy, q_list=q_list, a_band_index=args.a_band,
        batch_size=int(params.get("wfn_fi_q_chunk", 0)),
        eigh_backend=args.eigh_backend,
        use_low_mem_eigh=_use_low_mem_eigh, log_fn=log,
        distrib_la_batched_route=args.distrib_la_batched_route,
        htransform_quality_record_fn=_path_quality_records.append)
    if len(_path_quality_records) != 1:
        raise RuntimeError(
            "exciton_bands htransform returned "
            f"{len(_path_quality_records)} interpolation-quality receipts; "
            "expected one")
    report.htransform_quality(
        _path_quality_records[0],
        title="Exciton-path htransform interpolation quality")
    psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
        bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy)
    # Everything the htransform produced is now copied into the conduction
    # stacks and nothing below reads it again.  Drop it before the V_Q model
    # build: ``vq_interp.build_cq`` now returns C_q as a (μ, ν)-face SHARDED
    # device array (no per-proc host gather), but the coarse ζ / P_R
    # intermediates it builds still want the htransform leftovers gone.
    # ``bundle`` alone is ψ at every (Q, k) in both shardings.
    del bundle, ctilde, B_at_mu, enk_sigma
    tick("htransform_psi_cQ", t0)
    stage_progress.step()

    # on-grid gate at the first Γ node (path convention: starts at Γ)
    iGamma = [i for i in range(nQ)
              if np.linalg.norm(Qpath[i] - np.round(Qpath[i])) < 1e-9]
    if iGamma:
        # ψ stays ON DEVICE (μ-sharded): the gate contracts μ there and only
        # the (nk, nc, nc) Gram comes to host.  ε is replicated, so it is
        # gathered cheaply inside the gate.  See gate_htransform_vs_stored.
        # TIMED because it is a DIAGNOSTIC on the critical path: it runs one
        # host ``svd`` per k.  A diagnostic is allowed to cost something; it
        # is not allowed to cost something invisibly.
        t0 = time.time()
        # ``data`` is the POST-bse_k_grid bundle.  When densification is on,
        # both its psi/eps and this Q=Gamma cache live on the FINE grid; the
        # native coarse restart is no longer either operand.  Comparing a
        # coarse/fine intersection here used the coarse energy cardinality
        # left behind by the invalid post-densification --eqp path to index an
        # already-fine psi table, turning an operand-provenance bug into a
        # spurious subspace-overlap failure.  The gate below first verifies
        # each psi/eps pair has one k axis, then compares all actual rows.
        gate_htransform_vs_stored(
            psi_cQ_X[iGamma[0]], eps_cQ[iGamma[0]], data, mesh_xy, log=log)
        tick("gamma_gate", t0)

    # ── V_Q tiles ─ ONE shared arbitrary-Q model build.  The bse_k_grid
    #    coarse→fine general init (bse_io) calls the SAME
    #    ``vq_interp.build_vq_evaluator`` — there is a single exchange-interp
    #    orchestration, not one here and one there. ─────────────────────────
    t0 = time.time()
    # Q ON the coarse BZ grid needs NO exchange model at all: the production
    # tile V_qmunu[wrap(−Q)] IS the answer, and the driver already uses exactly
    # that at the one on-grid point it always has (Γ, below).  ``--vq-mode
    # ongrid`` extends that existing special case to every on-grid Q — no
    # interpolation error, no b26p stencil, no ζ.  It also makes the exciton
    # bandstructure runnable without reading ζ.  Pure refit is also
    # compatible with IBZ-only ζ: it reads metadata/provenance, then rebuilds
    # ζ(Q) from the canonical full-BZ WFN source.  Cost of ongrid: the full
    # (μ, ν, nkx, nky, nkz) exchange tensor alongside W_q.
    ongrid = (args.vq_mode == "ongrid")
    # ``refit`` used to refuse ("not wired") and to be unusable anyway: its
    # kernel was the slab one.  Since 2026-08-10 it is the exact arbitrary-Q
    # exchange on a bulk deck — the ONLY one, because the interpolation model
    # is slab-only — and it is what a dense Q path on a 3-D crystal runs.
    pure_refit = (args.vq_mode == "refit")
    if args.refit_window == "bse" and not pure_refit:
        raise SystemExit(
            f"exciton_bands: --refit-window=bse only means anything under "
            f"--vq-mode=refit, and this run is --vq-mode={args.vq_mode}.  In "
            f"--vq-mode=both the refit rows are the GROUND TRUTH the "
            f"interpolation model is scored against, so re-fitting them on a "
            f"different window would score interp against a moved reference; "
            f"in ongrid/interp no refit runs at all.")
    kgrid_bse = np.array([nkx, nky, nkz], dtype=np.int64)
    # WHICH GRID INDEXES THE EXCHANGE TILES.  Normally the bundle's own.  After
    # a ``bse_k_grid`` coarse→fine densification the bundle's grid is the FINE
    # one but the stored tiles are still the COARSE ones — exact where they
    # exist (V_{μν}(q) is a ζ/G-sphere object and never sees the k-grid) and
    # simply ABSENT at a fine q that is not a coarse q.  So the on-grid Q set
    # is the coarse grid on a densified bundle, and the refusal below says so
    # instead of letting a ``None`` subscript speak for it.
    V_ongrid = data.get("V_q_full")
    kgrid_vq = kgrid_bse
    vq_src = "the bundle grid"
    if V_ongrid is None and data.get("V_q_coarse") is not None:
        V_ongrid = data["V_q_coarse"]
        kgrid_vq = np.asarray(data["V_q_coarse_grid"], dtype=np.int64)
        vq_src = "the COARSE restart grid (bse_k_grid densification)"
    if ongrid:
        if V_ongrid is None:
            raise SystemExit(
                "--vq-mode=ongrid needs the stored exchange tensor V_qmunu and "
                "this bundle carries none.")
        frac = Qpath * kgrid_vq[None, :]
        off = np.max(np.abs(frac - np.round(frac)))
        if off > 1e-6:
            raise SystemExit(
                f"--vq-mode=ongrid needs every Q on the "
                f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} EXCHANGE-TILE grid "
                f"({vq_src}); the path is off by {off:.3e} grid units.  On a "
                f"bse_k_grid-densified bundle that grid is the COARSE restart "
                f"grid, not the fine BSE grid "
                f"({nkx}x{nky}x{nkz}): densifying moves the k-SUM, it does not "
                f"create exchange tiles at new q.  Use a K_POINTS block whose "
                f"segments land on the coarse grid, or --vq-mode=interp (which "
                f"needs FULL-BZ ζ storage and an exchange model valid for this "
                f"cell).\n"
                f"  CHECK --q-per-segment FIRST if this deck used to run: it "
                f"floors every segment at {args.q_per_segment} points by "
                f"default (2026-08-10 — E_S(Q) on a path is a bandstructure), "
                f"and a floor above ~2 puts most Q off any coarse grid.  "
                f"--q-per-segment 1 restores the deck's own counts exactly; "
                f"the dense path is what needs the full-BZ ζ.")
        log(f"  exchange: EXACT on-grid tiles V_qmunu[wrap(-Q)] "
            f"({nQ} Q, all on the "
            f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} grid = {vq_src}) "
            f"— no interpolation")
        head_mbz = False
        zx = prep = eval_vq = pinvF = coeffs_packed = None
    else:
        # Per-Q mini-BZ head cell-averaging: CLI --head-minibz-average overrides
        # the cohsex.in ``head_minibz_average`` key (default off = point value).
        head_mbz = (bool(args.head_minibz_average)
                    if args.head_minibz_average is not None
                    else bool(params.get("head_minibz_average", False)))
        log(f"  arbitrary-Q head: {'mini-BZ cell average' if head_mbz else 'point value'} "
            f"(head_minibz_average={head_mbz})")
        zeta_path = require_zeta_for_interp(restart_file, args.vq_mode,
                                            (nkx, nky, nkz))
        if pure_refit:
            # NO interpolation model is built at all.  The b26p long-range fit
            # is the slab-only half of vq_interp; the refit path fits ζ at the
            # target Q from the htransform ψ and contracts it with the
            # producer's own Coulomb door.  So this branch loads ζ and stops.
            zx = vq_interp.load_zeta_coarse(restart_file, zeta_path,
                                            mesh=mesh_xy, log_fn=log,
                                            require_slab=False,
                                            require_full_bz_zeta=False)
            prep = eval_vq = pinvF = coeffs_packed = None
            if head_mbz:
                raise SystemExit(
                    "exciton_bands: --head-minibz-average needs the "
                    "interpolation model's long-range pieces (minibz_head_vlr "
                    "reads prep), which --vq-mode=refit does not build.  The "
                    "refit keeps the full v(Q+G) including G=0 at every Q, "
                    "which is the pointwise head this flag would replace.")
        else:
            vqm = vq_interp.build_vq_evaluator(
                restart_file, mesh_xy, n_rmu_pad, alpha=args.alpha,
                eps_tik=args.eps_tik, eigh_backend=args.eigh_backend,
                distrib_la_batched_route=args.distrib_la_batched_route,
                head_minibz_average=head_mbz, log_fn=log)
            zx, prep = vqm.zx, vqm.prep
            eval_vq, pinvF, coeffs_packed = (vqm.eval_vq, vqm.pinvF,
                                             vqm.coeffs_packed)
    tick("vq_prepare", t0)
    report.heading("Resolved interaction kernels")
    report.emit("Exchange route : " + (
        f"exact stored tiles at {nQ} on-grid Q points" if ongrid else
        "per-Q ISDF refit with the producer Coulomb service" if pure_refit else
        "arbitrary-Q interpolation with exact stored Γ tile"))
    report.emit("Exchange head  : " + (
        "mini-BZ cell-averaged tensor" if head_mbz else
        "point value (stored Γ head-body tile at Γ)"))
    report.emit("Direct W       : " + (
        f"sampled on {args.w_coarse_grid}; densified to "
        f"{nkx} x {nky} x {nkz}" if args.w_coarse_grid else
        f"native {nkx} x {nky} x {nkz} grid"))
    report.emit(f"Window policy  : degeneracy={args.band_degeneracy}; "
                f"tolerance={float(args.degeneracy_tol_ry) * RY2EV * 1.0e3:.5f} meV")

    # ── the per-Q refit state + ITS GATE, before any tile is used ─────────
    rst = None
    zx_fit = zx
    cert_idx: list[int] = []
    if pure_refit:
        t0 = time.time()
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx,
                                      r_chunk=args.refit_r_chunk, log_fn=log,
                                      policy=zx.get("policy"),
                                      keep_idx=keep_idx,
                                      window_mode=args.refit_window,
                                      degeneracy_mode=args.band_degeneracy,
                                      degeneracy_tol_ry=args.degeneracy_tol_ry,
                                      n_guard=args.refit_guard_bands,
                                      distrib_la_batched_route=(
                                          args.distrib_la_batched_route))
        # THE ζ THE REFIT ACTUALLY FITS IN.  Identical to ``zx`` under
        # ``--refit-window=zeta``; a band-axis sub-window VIEW of it under
        # ``bse``.  Every refit_vq call below takes this, not ``zx`` — handing
        # it the unsliced bundle would fit ζ' on the wrong band count with no
        # shape error (the pair-density axes are contracted away).
        zx_fit = rst["zx_fit"]
        if V_ongrid is None:
            raise SystemExit(
                "exciton_bands: --vq-mode=refit certifies itself against the "
                "stored V_qmunu at on-grid q and this bundle carries none, so "
                "there is nothing to check the off-grid tiles against.  Load a "
                "bundle with the exchange tensor, or use --vq-mode=ongrid.")
        if args.refit_window == "zeta":
            # CERTIFY ON THE POPULATION THE PATH ACTUALLY TRAVERSES.
            # ``refit_ongrid_null``'s own default q_list is Γ plus the three
            # coarse q FURTHEST from Γ — a deliberate pick of the
            # head-dominated and zone-boundary slots, and it is the same
            # shape of four-point sample that
            # ``2026-08-11-refit-vq-sharded-fetch-and-cert-grades.md`` §4
            # showed had missed the worst point by a factor of 27 on the
            # windowed route: v(Q) ~ 1/|Q|² amplifies a ζ-fit error most at
            # SMALL |Q|, and the small-|Q| tiles are exactly the ones
            # "furthest from Γ" excludes.  So the null is additionally taken
            # at every distinct coarse tile this path lands on, de-duplicated
            # by tile momentum — the same population the ``bse``-window
            # branch below certifies its contracted object on.  This only
            # ADDS q to a gate; no bracket is touched.
            _frac = Qpath[:nQ_path] * kgrid_vq[None, :]
            _on = (np.max(np.abs(_frac - np.round(_frac)), axis=1) < 1e-6)
            _seen, _qlist = set(), []
            for _i in np.nonzero(_on)[0]:
                _key = tuple(int(v) for v in
                             (np.round((-Qpath[_i]) * kgrid_vq).astype(int)
                              % kgrid_vq))
                if _key not in _seen:
                    _seen.add(_key)
                    _qlist.append(_key)
            _kg = np.asarray(kgrid_vq, dtype=int)
            _idx = np.stack(np.meshgrid(*[np.arange(k) for k in _kg],
                                        indexing="ij"), axis=-1).reshape(-1, 3)
            _far = np.argsort(-np.sum(np.minimum(_idx, _kg[None, :] - _idx)**2,
                                      axis=1))
            for _j in [0] + [int(v) for v in _far[:3]]:
                _key = tuple(int(v) for v in _idx[_j])
                if _key not in _seen:
                    _seen.add(_key)
                    _qlist.append(_key)
            log(f"  [refit-null] tile null over {len(_qlist)} coarse q: every "
                f"distinct on-grid tile of the {nQ_path}-Q path, plus the "
                f"function's own default sample (Γ and the three q furthest "
                f"from it).  The path's small-|Q| tiles are the ones the "
                f"default sample cannot reach and the ones 1/|Q|² amplifies.")
            # ``zx_fit``, NOT ``zx``: identical on a stock bundle, and on a
            # ``zeta_nband``-decoupled one it is the producer's own ζ-fit
            # window sliced out of a wider stored band axis.  Handing the
            # unsliced bundle to the gate would refit ζ' on bands the
            # producer never fitted and then compare the result to the
            # producer's tiles — a refusal with no defect behind it.
            vq_interp.refit_ongrid_null(zx_fit, rst, V_ongrid, kgrid_vq,
                                        mesh_xy, log_fn=log, q_list=_qlist)
        # THE CONTRACTED CERTIFICATION — for BOTH windows, and for different
        # reasons.  Under ``bse`` it is the ONLY gate: ζ' is fitted on the BSE
        # window, so it is not the producer's ζ and cannot return the
        # producer's TILES; that identity is gone and re-tuning its bracket
        # would be widening a comparison with no reason to hold.  Under
        # ``zeta`` the tile null above already certifies the TILES, and this
        # certifies, in the units the curve is published in, the object the
        # driver actually delivers — a 5.0e-02 RELATIVE bracket on a tile is
        # not a meV statement about an eigenvalue, and the eigenvalue is the
        # deliverable.  It is strictly ADDITIVE there: it relaxes nothing, and
        # a run that passes the tile null and fails this one was never
        # certified by the tile null.
        #
        # Every path Q that lands on the coarse exchange-tile grid gets a TWIN
        # solve row carrying the producer's own stored tile, and the two
        # eigenvalue sets are compared after the scan
        # (``_certify_refit_against_stored``).  CERTIFY WHERE CONSUMED: the
        # refit's only consumer is that contraction, the twins ride the same
        # single-compile scan, and they reuse the path's own conduction
        # caches, so the gate costs no cache and no compile — only |cert|
        # extra Lanczos solves.
        frac = Qpath[:nQ_path] * kgrid_vq[None, :]
        on_grid = (np.max(np.abs(frac - np.round(frac)), axis=1) < 1e-6)
        finite = (np.linalg.norm(Qpath[:nQ_path]
                                 - np.round(Qpath[:nQ_path]), axis=1)
                  >= 1e-9)
        cert_idx = [int(i) for i in np.nonzero(on_grid & finite)[0]]
        # De-duplicate by TILE momentum: a path that passes through the
        # same coarse q twice (Γ appears twice on Γ–X–W–L–Γ–Σ, and its
        # neighbours can repeat) would otherwise pay for the same
        # comparison more than once.
        seen, uniq = set(), []
        for i in cert_idx:
            key = tuple(np.round((-Qpath[i]) * kgrid_vq).astype(int)
                        % kgrid_vq)
            if key not in seen:
                seen.add(key)
                uniq.append(i)
        cert_idx = uniq
        if not cert_idx and args.refit_window == "bse":
            raise SystemExit(
                f"exciton_bands --refit-window=bse: not one of the "
                f"{nQ_path} path Q is a FINITE point of the "
                f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} exchange-tile "
                f"grid, so there is nowhere the refit route and the "
                f"producer's stored tiles can be compared and nothing "
                f"certifies this run.  A windowed ζ' cannot be checked "
                f"tile-by-tile (it is a different ζ), so the contracted "
                f"gate is the only one, and it needs at least one shared "
                f"Q.  Use a K_POINTS path whose corners are on the coarse "
                f"grid — the standard Γ–X–W–L–Γ–Σ corners all are — or "
                f"run --refit-window=zeta on a bundle whose parent basis "
                f"spans nk·nb_ζ.")
        if not cert_idx:
            # ζ-window run with no shared Q.  NOT a refusal: the tile null
            # above is this mode's certification and it has already run.  Said
            # out loud so nobody reads a missing ``# refit-cert:`` stamp as a
            # gate that was skipped.
            log(f"  [refit-cert] none of the {nQ_path} path Q is a finite "
                f"point of the "
                f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} exchange-tile "
                f"grid, so the CONTRACTED certification has nowhere to stand "
                f"on this path.  The tile null above is this run's "
                f"certification; the .dat will carry no ``# refit-cert:`` "
                f"line and the figure no meV stamp.")
        else:
            log(f"  [refit-cert] "
                + ("BSE-window ζ': the tile null does NOT apply (ζ' ≠ ζ).  "
                   "Certifying the CONTRACTED object instead"
                   if args.refit_window == "bse" else
                   "ζ-fit-window ζ': the tile null above certifies the TILES; "
                   "this ADDITIONALLY certifies the CONTRACTED object, in the "
                   "meV the curve is published in")
                + f" at {len(cert_idx)} on-grid path Q "
                f"{[int(i) for i in cert_idx]} — each solved twice, once "
                f"through the refit exchange and once through the stored "
                f"V_qmunu tile, in the same scan.  Gate: "
                f"{CERT_TOL_BY_GRADE[args.cert_grade]:g} meV on every "
                f"eigenvalue ({CERT_GRADE_LABEL[args.cert_grade]}"
                + ("" if args.cert_grade == "reference" else
                   f", selected with --cert-grade={args.cert_grade}; the "
                   f"curve this run writes is a PICTURE and its levels are "
                   f"not reference numbers") + ").")
        tick("refit_prepare_and_null", t0)

    grid_xy = NamedSharding(mesh_xy, P("x", "y"))

    def _hermitize(V):
        return 0.5 * (V + jnp.conj(V).T)

    t0 = time.time()
    V_rows = []
    # Per-Q cell moment for the tensor head.  Zero for Γ and for every Q on
    # the OFF arm, which makes the head term an exact no-op there — the Γ
    # endpoint keeps the production q=0 tile and its rank-one head, as its
    # own docstring promises.
    M_rows = np.zeros((nQ, 3, 3), dtype=np.float64)
    head_scalars: list = []
    v_gamma = jax.device_put(data["V_q0"], grid_xy)
    n_eval_calls = 0
    for iQ in range(nQ):
        Qw = Qpath[iQ] - np.round(Qpath[iQ])
        if np.linalg.norm(Qw) < 1e-9:
            V_rows.append(_hermitize(v_gamma))       # production q=0 tile
            continue
        q_tile = -Qpath[iQ]                          # tile momentum = wrap(−Q)
        q_tile_np = q_tile - np.round(q_tile)
        if ongrid:
            ix, iy, iz = (np.round(q_tile_np * kgrid_vq).astype(int)
                          % kgrid_vq)
            V_rows.append(_hermitize(
                jax.device_put(V_ongrid[:, :, ix, iy, iz], grid_xy)))
            n_eval_calls += 1
            continue
        if pure_refit:
            V_np = vq_interp.refit_vq(zx_fit, rst, q_tile_np, mesh_xy,
                                      log_fn=log)
            V_pad = np.zeros((n_rmu_pad, n_rmu_pad), dtype=np.complex128)
            V_pad[:n_rmu, :n_rmu] = 0.5 * (V_np[:n_rmu, :n_rmu]
                                           + V_np[:n_rmu, :n_rmu].conj().T)
            V_rows.append(device_put_process_local(V_pad, grid_xy))
            n_eval_calls += 1
            continue
        q_tile = jnp.asarray(q_tile_np)
        if head_mbz:
            gstar, head_val, M_ab = vq_interp.minibz_head_vlr(
                zx, prep, q_tile_np, alpha=args.alpha, moment=True)
            # The head channel leaves the mu basis entirely: v[gstar] -> 0
            # removes the LR G* column from the tile, and the cell-averaged
            # head comes back as the rank-three tensor term in the matvec.
            # Injecting BOTH would double-count the head.
            M_rows[iQ] = M_ab
            V_rows.append(_hermitize(eval_vq(
                q_tile, prep["V_SRc"], pinvF, coeffs_packed,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(gstar, dtype=jnp.int32))))
            head_scalars.append((iQ, float(head_val), float(np.trace(M_ab))))
        else:
            V_rows.append(_hermitize(eval_vq(q_tile, prep["V_SRc"], pinvF,
                                             coeffs_packed)))
        n_eval_calls += 1
    if pure_refit:
        log(f"  exchange: per-Q ζ REFIT at all {n_eval_calls} finite Q "
            f"(compute-don't-interpolate; Γ keeps the production q=0 tile), "
            + ("certified against the stored V_qmunu by the on-grid tile null "
               "above" if args.refit_window == "zeta" else
               f"ζ' on bands {rst['window_abs']}; certification is the "
               f"contracted eigenvalue gate after the scan"))
    # CERTIFICATION TWINS.  One extra solve row per certification Q carrying
    # the PRODUCER's stored tile at that same wrap(−Q).  Appended after every
    # path row so ``evs_all[nQ + j]`` is the stored-route answer for
    # ``cert_idx[j]``, whose refit-route answer is already at ``evs_all[i]``.
    # Nothing else about the row differs — same conduction caches, same W_R,
    # same solver, same scan — so the eigenvalue difference is attributable to
    # the exchange tile and to nothing else.
    for iQ in cert_idx:
        q_tile_np = -Qpath[iQ] - np.round(-Qpath[iQ])
        ix, iy, iz = np.round(q_tile_np * kgrid_vq).astype(int) % kgrid_vq
        V_rows.append(_hermitize(
            jax.device_put(V_ongrid[:, :, ix, iy, iz], grid_xy)))
    refit_idx = []
    if args.vq_mode == "both":
        if args.refit_points:
            refit_idx = sorted({int(s) for s in args.refit_points.split(",")
                                if s.strip() != ""})
        else:
            refit_idx = sorted({int(i) for i in
                                np.linspace(1, nQ - 2, 5).round()})
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx,
                                      r_chunk=args.refit_r_chunk,
                                      policy=zx.get("policy"),
                                      keep_idx=keep_idx,
                                      n_guard=args.refit_guard_bands,
                                      distrib_la_batched_route=(
                                          args.distrib_la_batched_route))
        for iQ in refit_idx:
            q_tile = -Qpath[iQ]
            V_np = vq_interp.refit_vq(zx, rst, q_tile, mesh_xy)
            V_pad = np.zeros((n_rmu_pad, n_rmu_pad), dtype=np.complex128)
            V_pad[:n_rmu, :n_rmu] = 0.5 * (V_np + V_np.conj().T)
            # Process-local (AA.1): V_pad is host numpy, identical on every
            # rank; plain device_put would fire the hidden assert_equal
            # all-gather.  LORRAX_CHECK_REPLICA=1 re-arms it.
            V_rows.append(device_put_process_local(V_pad, grid_xy))
    # Row order in the stack, and the ONE place it is written down:
    #   [0, nQ)                                   the path (+ --extra-q)
    #   [nQ, nQ + n_cert)                         certification twins, stored
    #                                             tile at cert_idx[j]
    #   [nQ + n_cert, nQ + n_cert + n_refit)      --vq-mode=both refit spots
    # ``cert_idx`` and ``refit_idx`` are never both non-empty (one belongs to
    # --vq-mode=refit, the other to =both), but the readers below index off
    # this layout rather than off that fact.
    n_cert = len(cert_idx)
    n_solve = nQ + n_cert + len(refit_idx)
    assert len(V_rows) == n_solve, (
        f"V_rows {len(V_rows)} != n_solve {n_solve} "
        f"(nQ={nQ}, cert={n_cert}, refit={len(refit_idx)})")
    V_stack = jax.device_put(jnp.stack(V_rows),
                             NamedSharding(mesh_xy, P(None, "x", "y")))
    if head_mbz:
        # refit/cert rows carry no head tensor (they are the ground-truth
        # point-value comparison); zero is an exact no-op in the term.
        M_stack = np.concatenate(
            [M_rows, np.zeros((n_cert + len(refit_idx), 3, 3))], axis=0)
        for iQ, hv, trM in head_scalars[:4]:
            log(f"  [head-tensor] Q#{iQ}: <v_LR>_mBZ = {hv:.6f}, "
                f"tr M_ab = {trM * RY2EV:.6e} eV/bohr^2")
        if len(head_scalars) > 4:
            log(f"  [head-tensor] ... {len(head_scalars)} Q in total")
    tick("vq_eval", t0)
    stage_progress.step()

    # cert twins and refit rows reuse their Q's conduction caches: extend the
    # scan xs in the row order fixed above.  This is what makes the twin an
    # exchange-only comparison — the ψ_c(k+Q), ε_c(k+Q) operands are literally
    # the same rows, not a recomputation of them.
    if cert_idx or refit_idx:
        sel = jnp.asarray(list(range(nQ)) + list(cert_idx) + refit_idx)
        psi_cQ_X = psi_cQ_X[sel]
        psi_cQ_Y = psi_cQ_Y[sel]
        eps_cQ = eps_cQ[sel]

    # ── W_R once (the ONE sharded-FFT helper), then the single-compile scan ──
    t0 = time.time()
    sh = make_bse_shardings(mesh_xy)
    _ifftn = make_sharded_ifftn_3d(mesh_xy, sh.W.spec, sh.W.spec,
                                   axes=(2, 3, 4), norm="ortho")
    # W_q is DONATED on the two SAME-SHAPE paths, and the caller-side
    # reference dropped, copied from ``bse_lanczos``'s W_R build.  XLA grants
    # the alias only when the output shape matches the input, so W_R becomes
    # W_q's buffer rather than a second live tile for the whole band scan:
    # 56.25 MiB/rank on the Si 4x4x4 deck, 404 MB/rank at mu=10015 / P=64,
    # 4.1 GB/rank at mu=32k.  In-jit peak is unchanged; the win is entirely
    # caller-side (FFT_DONATION_AUDIT.md 2.3).  Value-identical.
    # The ``--w-coarse-grid`` path below is DELIBERATELY not donated: it
    # sub-samples W_q to a coarse sub-grid first, so the densifier's operand
    # is a different array of a different shape and donation is declined
    # anyway.  It also still needs ``data["W_q"]`` to build that sub-grid,
    # which is why the reference is dropped per-branch and not once up here.
    _ifftn_donated = jax.jit(_ifftn, donate_argnums=(0,))
    if args.w_coarse_grid is None:
        W_R = _ifftn_donated(data["W_q"])         # fast path: native fine W (byte-identical)
        data["W_q"] = None                        # release the caller-side reference
    else:
        cg = tuple(int(s) for s in args.w_coarse_grid.split(","))
        if len(cg) != 3:
            raise ValueError("--w-coarse-grid expects NX,NY,NZ")
        if cg == (nkx, nky, nkz):
            W_R = _ifftn_donated(data["W_q"])     # equal grids → no-op, byte-identical
            data["W_q"] = None                    # release the caller-side reference
        else:
            # Coarse-W → fine direct term.  Sub-sample the fine W_q onto the
            # coarse BZ sub-grid (same ISDF μ-basis; q=0 head-tile preserved),
            # then the ONE sharded densifier (bse_io.make_w_densifier: shard_map
            # ifft to the coarse R-lattice + jitted R zero-pad = exact trig
            # interpolation, (μ,ν) sharding preserved throughout — no eager pad
            # + device_put re-shard).  The convolution then runs on the fine
            # grid with the fine (nkx,nky,nkz) solver — cheap coarse W, fine
            # excitons.
            #
            # C1 (default, ``--w-head-densify c1``).  The Γ head is SPLIT OFF
            # before the sub-sampling, so what gets trig-interpolated is the
            # body alone, and the head is re-attached analytically at every
            # fine q inside the coarse Γ cell (gw.head_densify).  Note the
            # asymmetry with the ``bse_k_grid`` path: there the loader can
            # DEFER the injection because it knows a densification is coming,
            # whereas here the restart is natively fine and the loader has
            # already injected — so the head is subtracted back off, using the
            # same rank-one object with the same scalar.  That subtraction is
            # float-inexact at the 1e-16 level and is inherent to the harness,
            # not to C1; the legacy arm carries the identical rounding because
            # it decimates the very same injected tile.
            # Imported HERE, not at module scope.  ``gw.head_densify``
            # bootstraps the service search path at import time
            # (``ffi._services.ensure_on_path()``), and doing that from a
            # driver's import block reorders sys.path for every run of the
            # driver, including the ones that never densify.  Measured: it
            # moved the X-point exciton cluster of the default path by 3e-5 eV
            # — reproducibly, on a near-degenerate sextet, with no array in
            # the code changed.  A feature that is off by default must not be
            # able to do that, so the import lives inside the branch that
            # uses it.
            from gw.head_densify import attach_head_channel
            w_head_mode = resolve_w_head_densify(args.w_head_densify, params)
            W_q_fine_in = data["W_q"]
            head_ch = None
            if w_head_mode == "c1":
                head_ch = _resolve_native_w_head(
                    restart_file, args.input, wfn, log=log)
            if head_ch is not None:
                W_q_fine_in = attach_head_channel(
                    W_q_fine_in, data["g0_X"], data["g0_Y"],
                    _gamma_only_grid((nkx, nky, nkz), -head_ch["whead"]),
                    head_ch["cell_volume"])
            W_q_coarse = decimate_W_q_to_subgrid(W_q_fine_in, cg)
            densify_W = make_w_densifier(mesh_xy, sh.W.spec, (nkx, nky, nkz),
                                         output="k" if head_ch else "R")
            W_dense = densify_W(W_q_coarse)
            if head_ch is not None:
                S_fine = build_w_head_channel(
                    wfn, sym, meta, params, coarse_grid=cg,
                    fine_grid=(nkx, nky, nkz), whead=head_ch["whead"],
                    # THE REFERENCE GRID IS THE FINE ONE HERE.  The restart is
                    # natively fine, so its whead is the fine cell's average —
                    # which makes C1's Γ entry reproduce it EXACTLY, i.e. the
                    # arm that simulates a coarse W puts back the head the
                    # fine run really had.  Passing the coarse grid instead
                    # would silently rescale the head by ~m².
                    ref_grid=(nkx, nky, nkz),
                    input_file=args.input, restart_file=restart_file,
                    gamma_cell=args.w_head_gamma_cell, log_fn=log)
                W_dense = attach_head_channel(
                    W_dense, data["g0_X"], data["g0_Y"], S_fine,
                    head_ch["cell_volume"])
                W_R = _ifftn(W_dense)
            else:
                W_R = W_dense
            log(f"[coarse-W] W sampled on {cg[0]}x{cg[1]}x{cg[2]} sub-grid of "
                f"{nkx}x{nky}x{nkz}, zero-padded in R (trig-interp to fine "
                f"grid) [w_head_densify={w_head_mode}"
                + (f", gamma_cell={args.w_head_gamma_cell}]"
                   if head_ch else "]"))
    # ── the tensor head's transition-side operand: conj(d) on the BSE window ──
    # Q-INDEPENDENT.  d is the q→0 coefficient of the pair amplitude
    # (LT_HEAD_PROBLEM.md §6.1), so it belongs to the k-point transition, and
    # only the cell moment M_ab carries Q.
    head_args = ()
    if head_mbz:
        D_head = build_head_dipole_operand(
            args, nk, nc_pad, nv_pad, n_val, n_cond, log=log)
        head_args = (jnp.asarray(D_head),
                     jnp.asarray(M_stack, dtype=jnp.float64))
    solver = build_path_solver(
        mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=args.n_eig,
        block_size=args.block_size, max_iter=args.max_iter,
        head_tensor=head_mbz)
    tick("w_r_and_build", t0)
    stage_progress.step()
    t_c0 = time.time()
    evs_dev, alpha_dev = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                                data["psi_v_X"], data["psi_v_Y"],
                                data["eps_v"], W_R, *head_args)
    # THE ONLY HOST FETCH LEFT IN THIS DRIVER, and it issues no collective:
    # ``evs_dev`` is the block-Lanczos Ritz table out of a small replicated
    # eigh(T), ``P()``-sharded, so it takes the service's replicated arm (a
    # local buffer read).  See ``_gather_host``.  Logged so the run proves it.
    log(f"[dist] evs sharding={evs_dev.sharding}, "
        f"fully_addressable={evs_dev.is_fully_addressable}")
    evs_all = _gather_host(evs_dev)                   # (n_solve, n_eig) Ry
    # The α-Hermiticity invariant, replayed on the host from scalars the scan
    # returned.  Same tolerance, same message, same LORRAX_SANITY=strict raise
    # as the in-jit callback it replaces — see _report_alpha_over_path.
    alpha_ok = _report_alpha_over_path(
        solver.alpha_labels, jax.device_get(alpha_dev), log=log)
    t_first = time.time() - t_c0
    tick("solve_scan_cold", t_c0)
    stage_progress.step()
    if rerun_check_enabled(args):
        # warm re-run: census-clean per-Q cost + reproducibility assert.
        # Pure diagnostic — it re-executes the ENTIRE Q scan a second time.
        # Measured share of the wall when it is on: 1767.17 s of a 4633.36 s
        # run = 38.1% at P=64/91 Q (job 7882533), 47.74 s of 126.46 s = 37.7%
        # at P=4/41 Q — the single largest row in the table either way.  Which
        # is why it is OFF unless ``--rerun-check`` asks for it; see
        # ``rerun_check_enabled`` for the whole decision.
        t_w0 = time.time()
        evs2 = _gather_host(
            solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                   data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R,
                   *head_args)[0])
        t_warm = time.time() - t_w0
        tick("solve_scan_warm", t_w0)
        assert np.allclose(evs2, evs_all, atol=1e-10), \
            "scan re-run not reproducible"
        log(f"solve_path: cold {t_first:.2f}s (incl. ONE compile), warm "
            f"{t_warm:.2f}s = {t_warm/n_solve*1e3:.1f} ms/Q over {n_solve} Q")
        log(f"  [diagnostic cost] the warm re-run is a REPRODUCIBILITY CHECK, "
            f"not physics: it re-solves all {n_solve} Q and is "
            f"{100.0*t_warm/max(time.time()-t_wall, 1e-9):.0f}% of the wall so "
            f"far.  Drop --rerun-check once this configuration is trusted.")
    else:
        log(f"solve_path: cold {t_first:.2f}s (incl. ONE compile) over "
            f"{n_solve} Q; warm re-run check SKIPPED (the default since "
            f"2026-08-08 — pass --rerun-check to run it)")
    t0 = time.time()
    mem = solver.lower(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                       data["psi_v_X"], data["psi_v_Y"], data["eps_v"],
                       W_R, *head_args).compile().memory_analysis()
    log(f"solve_path memory_analysis: temp={mem.temp_size_in_bytes/2**20:.1f} MiB "
        f"args={mem.argument_size_in_bytes/2**20:.1f} MiB "
        f"out={mem.output_size_in_bytes/2**20:.1f} MiB")
    # TIMED: this is an AOT ``lower().compile()`` of the SAME program that was
    # just executed, for a memory report.  Whether it hits XLA's in-process
    # executable cache or recompiles from scratch is an XLA implementation
    # detail, and the difference is the whole compile (~157 s at P=64, 91 Q).
    # A diagnostic that can silently cost a compile gets its own row.
    tick("mem_analysis", t0)
    t_out0 = time.time()

    evs_path = evs_all[:nQ]
    evs_refit = {iQ: evs_all[nQ + n_cert + j]
                 for j, iQ in enumerate(refit_idx)}

    # ── THE CONTRACTED CERTIFICATION, before a single number is written.
    #    Refit route (already in ``evs_path``) against stored route (the twin
    #    rows) at the same Q, same caches, same solver.  Under ``bse`` it is
    #    the only gate; under ``zeta`` it rides alongside the tile null. ────
    cert_rows, cert_worst = [], None
    if cert_idx:
        cert_rows, cert_worst = _certify_refit_against_stored(
            cert_idx, Qpath, evs_path, evs_all[nQ:nQ + n_cert],
            kgrid_vq, log=log, grade=args.cert_grade,
            window_mode=args.refit_window)
        # PROVENANCE.  The run's own rank-0 block is what outranks every page
        # in the register (AGENT_PREAMBLE), so the grade and the certified
        # number are stated there before the first byte of output is written,
        # in the same words the .dat and the .png will carry.
        log(f"  [provenance] LORRAX {lorrax_version()}: "
            f"{cert_grade_stamp(args.cert_grade, cert_worst)}; "
            f"refit window={args.refit_window}, {len(cert_rows)} on-grid "
            f"certification Q, out-prefix {args.out_prefix}")

    report.heading("Exciton spectrum and validation")
    _path_ev = np.asarray(evs_path[:nQ_path], dtype=np.float64) * RY2EV
    report.emit(f"Computed levels: {int(args.n_eig)} at {int(nQ_path)} path points")
    report.emit(f"Energy extent  : [{float(np.min(_path_ev)):.5f}, "
                f"{float(np.max(_path_ev)):.5f}] eV")
    report.emit("Hermiticity    : " + ("PASS" if alpha_ok else "FAILED"))
    for inode, (idx, label) in enumerate(zip(node_idx, node_labels), start=1):
        levels = _path_ev[int(idx)]
        shown = "  ".join(f"{float(value):.5f}" for value in levels[:3])
        suffix = "  ..." if levels.size > 3 else ""
        report.emit(f"  N{inode:02d} {label or '-':>3}  lowest E_S (eV): "
                    f"{shown}{suffix}")
    if cert_worst is not None:
        report.emit(f"Refit cert     : {args.cert_grade}; worst "
                    f"{float(cert_worst):.5f} meV over {len(cert_rows)} "
                    "on-grid Q points")
    elif args.vq_mode in ("refit", "both"):
        report.emit("Refit cert     : tile-level certification completed "
                    "for the producer zeta window")
    else:
        report.emit("Refit cert     : not applicable to this exchange route")

    # ── outputs (rank 0 ONLY — the .dat / .png writes and the plot must not
    #    race across the 16 processes; evs_all is fully addressable on every
    #    process, so rank 0 holds the complete table) ──────────────────────
    if _rank0:
        labels = [(lbl or "") for lbl in node_labels]
        dat = args.out_prefix + ".dat"
        with open(dat, "w", encoding="utf8") as fh:
            fh.write("# Exciton bandstructure E_S(Q), TDA, LORRAX\n")
            fh.write(provenance_header())
            fh.write(f"# input: {os.path.abspath(args.input)}\n")
            fh.write(f"# window: n_val={n_val} n_cond={n_cond}; n_eig={args.n_eig}; "
                     f"kgrid {nkx}x{nky}x{nkz}; vq_mode={args.vq_mode}\n")
            if pure_refit:
                fh.write(f"# refit: window={args.refit_window}"
                         + (f" (zeta' re-fitted on absolute bands "
                            f"[{rst['window_abs'][0]}, {rst['window_abs'][1]})"
                            f", Galerkin bound nk*nb="
                            f"{zx_fit['nk'] * zx_fit['nb']})"
                            if args.refit_window == "bse" else
                            " (producer's zeta-fit window)") + "\n")
                # THE TWO-WINDOW CONTRACT TRAVELS WITH THE DATA.  A curve
                # drawn through a zero-guard refit is a curve through
                # arbitrary null-space directions at the top of its own
                # window, and the only way a reader tells the two apart after
                # the fact is this line.
                fh.write(f"# refit-fh-window: absolute bands "
                         f"[{rst['window_abs_fh'][0]}, "
                         f"{rst['window_abs_fh'][1]}) = zeta window + "
                         f"{rst['n_guard']} guard band(s); Galerkin residual "
                         f"{rst['galerkin_rel']:.3e} (fH) / "
                         f"{rst['galerkin_rel_zeta']:.3e} (zeta window)\n")
            if cert_rows:
                # THE CERTIFICATION TRAVELS WITH THE DATA.  A gate whose
                # numbers live only in a log is a claim about a file nobody
                # has; these are the numbers that say this curve is trusted.
                fh.write(f"# refit-cert: {cert_grade_stamp(args.cert_grade, cert_worst)}"
                         f", contracted on-grid gate (refit route vs stored "
                         f"V_qmunu route) over {len(cert_rows)} Q\n")
                if args.cert_grade != "reference":
                    # SAY IT IN WORDS, IN THE FILE.  A reader who meets this
                    # .dat with no log has to be able to tell a picture from
                    # a reference number without asking anyone.
                    fh.write(f"# refit-cert-grade: VISUALIZATION GRADE — "
                             f"certified only to "
                             f"{CERT_TOL_VISUALIZATION_MEV:g} meV, which is "
                             f"the tolerance for READING THIS AS A PICTURE "
                             f"(features at tens of meV).  These levels are "
                             f"NOT reference numbers; the "
                             f"{REFIT_CERT_TOL_MEV:g} meV reference grade is "
                             f"the default and is what a quoted eigenvalue "
                             f"needs.\n")
                for iQ, tile, dmax in cert_rows:
                    fh.write(f"#   Q#{iQ} tile {tile}: "
                             f"max|dE_S| = {dmax:.5f} meV\n")
            fh.write("# conventions: |v k, c k+Q>; exchange tile at wrap(-Q) "
                     "keeps G=0 at finite Q (energy_loss); Gamma uses the "
                     "production q=0 head-body tile; energies in eV\n")
            fh.write(f"# nodes: {' '.join(f'{int(i)}:{l}' for i, l in zip(node_idx, labels))}\n")
            fh.write("# iQ  s_path  Qx  Qy  Qz  mode  E_1..E_neig (eV)\n")
            for iQ in range(nQ):
                row = " ".join(f"{e*RY2EV:.6f}" for e in evs_path[iQ])
                # extras have no path distance: reuse the last path x so the
                # column stays numeric, and flag them in the mode column.
                sx = x_path[iQ] if iQ < nQ_path else x_path[nQ_path - 1]
                mode = "interp" if iQ < nQ_path else "extra "
                fh.write(f"{iQ:4d} {sx:.6f} "
                         f"{Qpath[iQ][0]: .6f} {Qpath[iQ][1]: .6f} "
                         f"{Qpath[iQ][2]: .6f} {mode} {row}\n")
            for iQ in refit_idx:
                row = " ".join(f"{e*RY2EV:.6f}" for e in evs_refit[iQ])
                fh.write(f"{iQ:4d} {x_path[iQ]:.6f} "
                         f"{Qpath[iQ][0]: .6f} {Qpath[iQ][1]: .6f} "
                         f"{Qpath[iQ][2]: .6f} refit  {row}\n")
        print(f"Wrote {dat}")

        if refit_idx:
            print("\ninterp vs refit (ground truth) at spot-check Q:")
            hdr = f"{'iQ':>4} {'s':>8} " + " ".join(
                f"{'dE'+str(j+1)+'(meV)':>10}" for j in range(args.n_eig))
            print(hdr)
            for iQ in refit_idx:
                d = (evs_path[iQ] - evs_refit[iQ]) * RY2EV * 1e3
                print(f"{iQ:4d} {x_path[iQ]:8.4f} "
                      + " ".join(f"{v:10.3f}" for v in d))

        # plot (Agg; no display)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for b in range(args.n_eig):
            ax.plot(x_path, evs_path[:nQ_path, b] * RY2EV, lw=1.2, color="C0",
                    alpha=0.9, label="interp" if b == 0 else None)
        for iQ in refit_idx:
            ax.scatter(np.full(args.n_eig, x_path[iQ]), evs_refit[iQ] * RY2EV,
                       s=22, facecolors="none", edgecolors="C3", zorder=5,
                       label="refit (ground truth)" if iQ == refit_idx[0] else None)
        ticks = x_path[np.asarray(node_idx, dtype=int)]
        for xpos in ticks:
            ax.axvline(xpos, color="k", lw=0.6, alpha=0.3)
        ax.set_xticks(ticks, labels)
        ax.set_xlim(x_path[0], x_path[-1])
        ax.set_ylabel("$E_S(Q)$ (eV)")
        ax.set_title("Exciton bandstructure (TDA)")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        if cert_worst is not None:
            # ON THE PICTURE ITSELF.  A .dat header travels with the data and
            # a log travels with nobody; the figure is the thing that ends up
            # in a talk, and it is the one artefact that must not be able to
            # arrive without its grade.  ``tight_layout`` first, then the
            # reserved strip, so the stamp cannot be laid over the axes.
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.01, 0.012,
                     cert_grade_stamp(args.cert_grade, cert_worst),
                     fontsize=6.5, color="#5A5A5A", ha="left", va="bottom")
        png = args.out_prefix + ".png"
        fig.savefig(png, dpi=180)
        print(f"Wrote {png}")

        timers["outputs"] = time.time() - t_out0
        total = time.time() - t_wall
        untimed = total - sum(timers.values())
        print("\n--- timings (s) ---")
        for k, v in timers.items():
            print(f"  {k:<22s} {v:9.2f}   {100.0*v/max(total,1e-9):5.1f}%")
        # The residual, ALWAYS printed.  A small number here is the table's own
        # evidence that it is complete; a large one names the gap instead of
        # leaving a reader to discover it by subtraction.
        print(f"  {'(untimed)':<22s} {untimed:9.2f}   "
              f"{100.0*untimed/max(total,1e-9):5.1f}%")
        print(f"  {'TOTAL':<22s} {total:9.2f}   100.0%")

    stage_progress.step()
    stage_progress.finish()

    # LEAVE TOGETHER.  Everything above this line inside ``if _rank0`` — the
    # .dat write, the interp-vs-refit table, matplotlib — is rank-0 only and
    # takes seconds.  Without this barrier ranks 1..P-1 return from main(),
    # exit, and tear their MPI process down while rank 0 is still in
    # matplotlib; MPI aborts the surviving rank and the job exits 134 (SIGABRT)
    # AFTER having written a completely correct .dat and .png.  Measured: job
    # 7882507 cell exb64s, rc=134 with exb_smoke_p64.dat/.png both present and
    # correct.  That is worse than a plain failure, because a harness cannot
    # tell it from one — every multi-rank exciton_bands run scored FAIL on a
    # successful calculation.  ``barrier`` is a no-op at process_count()<=1 and
    # is not on any solver path: one sync, once, after all the physics.
    from common.collectives import barrier
    barrier("exciton_bands.outputs_written")

    # The streamed refit owns a second mesh-aware WFN loader plus its bounded
    # PsiGStore.  Release the store first, then enter the loader's collective
    # close while every rank is still aligned at the output barrier.
    if rst is not None:
        try:
            vq_interp.close_refit_state(rst)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [exciton_bands] refit resource close failed "
                  f"({type(exc).__name__}: {exc}); continuing to exit")

    # CLOSE THE LOADER HERE, EXPLICITLY, WHILE THE RANKS ARE STILL IN STEP.
    # ``initialize_wfns`` hands back a MESH-AWARE ``WfnLoader``
    # (``htransform.setup_wfn_and_sym`` -> ``WfnLoader(wfn_file, mesh=mesh_xy)``),
    # so at P>1 it picks the phdf5 backend and owns a ``SlabIO`` whose
    # ``close()`` runs an UNCONDITIONAL COLLECTIVE barrier
    # (``file_io/_slab_io_ffi.py``, ``_barrier("slab_io_ffi_close_attrs")``).
    # Left to ``WfnLoader.__del__``, that collective fires whenever the
    # garbage collector happens to drop the object during interpreter
    # shutdown — a moment no two ranks agree on.  Measured on this tree
    # (jobs 56550230 steps .9/.10, 2026-08-09): three ranks parked in
    # ``__del__`` -> ``SlabIO.close`` -> ``sync_global_devices`` (NCCL,
    # spinning in ``cuStreamSynchronize``) while the fourth had already
    # reached ``ffi.io._atexit_close_all`` -> ``phdf5_close`` -> ``H5Fclose``
    # -> ``MPI_File_close`` -> ``MPI_Barrier`` (spinning in ``sched_yield``).
    # Two disjoint collective domains, neither ever satisfied: the payload
    # was complete and every output written, and the step then held its four
    # GPUs at 4x100% CPU until the allocation died.
    #
    # AFTER the barrier above, not before: that barrier is what guarantees
    # rank 0's matplotlib/.dat block has finished, so every rank enters this
    # close at the same point.  ``close()`` is idempotent and nulls the
    # handles, so the later ``__del__`` becomes a no-op on every rank and
    # nothing collective is left to interpreter shutdown.  A no-op at P=1,
    # where the SlabIO barrier is already a no-op.
    wfn.close()

    report.timings(tuple(timers.items()), wall=time.time() - t_wall)
    dat_path = os.path.abspath(args.out_prefix + ".dat")
    png_path = os.path.abspath(args.out_prefix + ".png")
    wfn_name = str(params.get("wfn_file", "WFN.h5"))
    if not os.path.isabs(wfn_name):
        wfn_name = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                wfn_name)
    file_rows = [
        ("human-readable report", "written", report_path),
        ("exciton bands", "written", dat_path),
        ("exciton plot", "written", png_path),
        ("GW/BSE restart", "read", restart_file),
        ("wavefunctions", "read", wfn_name),
        ("centroid coordinates", "read", centroids_path),
    ]
    if args.eqp:
        file_rows.append(("QP corrections", "read", args.eqp))
    if head_mbz:
        dipole_path = (args.dipole if args.dipole else
                       os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                    "dipole.h5"))
        file_rows.append(("dipole head tensor", "read", dipole_path))
    file_rows.append(("input deck", "read", args.input))
    report.files(file_rows)
    report.finish()
    barrier("exciton_bands.report_written")
    production_stdout.close()
    return 0


if __name__ == "__main__":
    # ``runtime.finalize_process``, not a bare ``SystemExit``: the explicit
    # ``wfn.close()`` above removes the collective from ``__del__``, and this
    # removes the SECOND unordered collective — ``ffi.io._atexit_close_all``,
    # whose ``H5Fclose`` on the restart/zeta contexts is collective too and
    # otherwise runs at whatever point each rank's interpreter teardown
    # reaches it.  ``finalize_process`` runs the effects barrier, the
    # distributed shutdown and the atexit hooks in ONE stated order on every
    # rank and then ends with ``os._exit``.  Same adopter pattern as
    # ``gw.gw_jax``, which is the sibling driver that has never hung.
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
