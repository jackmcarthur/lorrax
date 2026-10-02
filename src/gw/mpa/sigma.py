"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import dataclasses

import gc
import math
from dataclasses import replace
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common import timing
from common.progress import LoopProgress
from common.units import RYD_TO_EV
from file_io.restart_bundle import (
    PoleReader,
    open_pole_reader,
    validate_fit_store,
)
from gw.efermi import occupation_floor_reach_ry
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import (
    BandCountCube, SigmaOmegaResult, _residue_for_space, sigma_band_axis)
from gw.ppm_tau_kernel import (_get_sigma_kij_kernel,
                               get_shared_sigma_tau_kernel)
from gw.ppm_windows import branches_for_omega_grid
from gw import quadrature_log
from gw.sigma_box_plan import RUN_SCOPE, plan_sigma_windows, sigma_rule_scope
from gw.wavefunction_bundle import (
    parent_sigma_operands, sigma_face_kernel_kwargs)
from runtime.padding import combined_divisor, pad_to_axis, padded_axis

from .sigma_windows import (summarize_sigma_poles,
                            shared_pole_frequencies,
                            shared_pole_intervals,
                            summarize_shared_poles)


def _unfenced(name, *, sync_ranks=True):
    """Do not fence this τ band.

    ``timing.fence`` drains every live array and enters a global barrier so
    that a host band can be ATTRIBUTED; it is a profiling boundary, never
    physics (``common/timing.py``).  Production must not pay for it: this
    executor also serves the incumbent elementwise-MPA route, whose cost is
    not this campaign's to spend.  A measurement harness rebinds the module
    attribute ``_band_fence`` to ``timing.fence``, exactly as it already
    rebinds ``timing.section`` and the store reader, so band-level
    measurement stays available without a dial, an env var or a fast path.
    The incumbent route is unfenced whatever a harness installs.
    """


_band_fence = _unfenced


@jax.jit
def _shared_pole_weights(poles2, intervals, E_ref_B, t_node):
    """Causal residue weights exp[-i(Ω-Eref)τ]/(2Ω), DESIGN §3.4.

    Replicated ``poles2[parent,column]`` is in Ry², ``intervals[parent,2]``
    selects half-open active columns, Eref is Ry and signed complex τ is
    Ry^-1. Padding is made harmless before exponentiation, including at
    complex Laplace nodes. Evaluation η belongs to the existing rule only.
    """
    columns = jnp.arange(poles2.shape[1])[None, :]
    selected = ((columns >= intervals[:, :1])
                & (columns < intervals[:, 1:]))
    omega = jnp.sqrt(jnp.where(selected, poles2, 1.0))
    phase = jnp.where(selected, omega - E_ref_B, 0.0)
    return jnp.where(
        selected, jnp.exp(-1j * phase * t_node) / (2.0 * omega), 0.0j)


@jax.jit
def _shared_pole_omega0_weights(poles2, intervals, E_ref_B, t_node):
    """The causal branch at omega = 0: 1/(2 Omega (0 - Omega)) = -1/(2 Lambda).

    The frequency-domain partner of :func:`_shared_pole_weights` (the
    one-sided transform of (SP 4) is ``1/(2 Omega (omega - Omega))``),
    evaluated at the one frequency the static restart W0 needs. Same
    signature, so the synthesis binds either; Eref and tau are unused.
    """
    del E_ref_B, t_node
    columns = jnp.arange(poles2.shape[1])[None, :]
    selected = ((columns >= intervals[:, :1])
                & (columns < intervals[:, 1:]))
    return jnp.where(selected, -0.5 / jnp.where(selected, poles2, 1.0),
                     0.0).astype(jnp.complex128)


# Kernel lessons: the W(tau) synthesis b d(tau) b^dagger and its transposes (numbers: sandbox
# claim ids).
# Over plain JAX: none; it runs on G's GEMM route.  What paid was placement and communication:
#   tau-invariant factors placed once, Sigma tau -30% scalar Fe 4^3 (2732), -40% bispinor sectors
#   (2726); only each window's live pole columns contracted, Fe 8^3 map-0 Sigma tau -5.0% (2955);
#   transposes formed on the rank that needs them, Fe 8^3 Sigma tau -9.3% at P4, -24.5% at P16 (2958);
#   where the replicated columns do not fit, whole parents per rank (only W moves, no per-node
#   factor re-gather): Ni 20^3 P64-local tile synthesis 0.427 -> 0.131 s per tau node at P4,
#   0.81 -> 0.18 s at P16 (3100).  Where the replicated columns fit (Fe 4^3, Na 8^3 P4) they win:
#   the same local GEMM with no W exchange (whole parents there: +10-13% Sigma tau, 3100).
#   The photon sectors (sector_sigma.sector_synthesis) synthesize here on the same rule: at the
#   Fe 20^3 P36-local class tile the synthesis per tau node is CC 0.147 -> 0.064 s, TT 0.263 ->
#   0.099 s (axis), the node 2-9% faster; Fe 4^3 bispinor Sigma tau -19% per SC map (3116).
# Did not pay: W^T by a second GEMM, +10% at P4, +2% at P16 (2958), +5.7% on Fe 8^3 (2955); the 2-D
#   face layout for the Sigma residues, Sigma tau +25% Na 8^3, +160% Fe 4^3 (2955); the local
#   projector transpose costs Na 8^3 (group order 14.8) +6.4%, accepted (2958); a sync-free tau
#   loop, null (2829).  The latency-hiding scheduler: distrib_la.panel_matmul's lessons.
# Decides it: rank imbalance, not bytes.  transpose_xy copies on diagonal ranks and moves the tile on
#   the others, so diagonal ranks waited ~16 ms per node at the next collective (2954).  Left: this
#   transpose, 5.1 vs 1.5 ms per node at P16, ~7% of a node (2958).
def synthesize_shared_pole_parents(
    b_X, b_Y, poles2, intervals, E_ref_B, t_node, *, mesh_xy, gemm, layout="face",
    weights_fn=_shared_pole_weights, active_range=False, same_factor=True,
):
    """Synthesize both raw-parent orientations through the configured G service.

    The one W(τ) = b d(τ) b† owner: the scalar model and every photon sector
    (``gw.mpa.sector_sigma.sector_synthesis``) contract here.

    Parameters
    ----------
    b_X, b_Y : jax.Array
        Complex128 physical factors ``[parent,mu,spin,column]`` with
        face layouts, or replicated columns for the configured axis layout.
        The component axis is 1 for charge and 3 for current endpoints.
        ``layout='local'``: ``b_X`` is the one factor ``[Bp,mu,column]`` in
        ``distrib_la``'s batch layout (whole parents per rank, Bp the parents
        padded to the mesh) and ``b_Y`` is ``None`` (:func:`_shared_pole_contract_local`),
        or the right factor ``[Bp,nu,column]`` of a two-factor sector.
    same_factor : bool
        Both orientations hold the same physical b (the scalar model, a
        diagonal sector): the partner is the transpose of W.  ``False`` (a
        mixed CT/TC sector, ``b_X d b_Y†``): the partner ``conj(b_X) d b_Yᵀ``
        is contracted from the same operands (the face route's same panel
        exchange; the local route's same resident rows).
    poles2 : jax.Array
        Replicated float64 ``[parent,column]`` squared frequencies in Ry²
        (``layout='local'``: ``[Bp,column]`` in the batch layout).
    intervals : jax.Array
        Replicated integer ``[parent,2]`` active half-open column ranges,
        prepared from the sorted census for this window/panel.
    E_ref_B, t_node : scalar
        Window reference in Ry and signed complex time in Ry^-1.
    mesh_xy : jax.sharding.Mesh
        Existing mesh with named x/y axes.
    gemm : distrib_la.GemmPlan
        Eagerly planned N,N contraction for this parent and padded pole panel.
    active_range : bool
        ``gemm`` was planned with ``enable_active_range=True``: contract only
        each parent's ``intervals`` columns, each scaled by its weight.

    Returns
    -------
    Wplus, Wtranspose : jax.Array
        Parent ``P(None,'x','y')`` tiles ``(b_X d) b_Y†`` and
        ``(conj(b_X) d) b_Yᵀ`` at the SAME τ (DESIGN §3.4). Never conjugate
        Wplus to obtain its antiunitary partner: d must retain its phase.
    """
    if layout == "local":
        # The replicated intervals take the factor's padded batch rows (an
        # empty interval: zero weight) and its batch placement, a local slice.
        n_parent, rows = int(intervals.shape[0]), int(b_X.shape[0])
        intervals = jax.lax.with_sharding_constraint(
            jnp.pad(intervals, ((0, rows - n_parent), (0, 0))),
            NamedSharding(mesh_xy, P(("x", "y"), None)))
        weights = weights_fn(poles2, intervals, E_ref_B, t_node)
        plus = _shared_pole_contract_local(b_X, weights, intervals, mesh_xy=mesh_xy,
                                           n_parent=n_parent, right=b_Y,
                                           partner=not same_factor)
    else:
        if b_X.ndim != 4 or b_Y.ndim != 4:
            raise ValueError("shared-pole faces require [parent,mu,spin,column]")
        if b_X.shape[2] not in (1, 3) or b_Y.shape[2] not in (1, 3):
            raise ValueError("GATE shared_pole_components: expected charge=1 or current=3")
        weights = weights_fn(poles2, intervals, E_ref_B, t_node)
        plus = _shared_pole_contract(b_X, b_Y, weights, gemm=gemm, layout=layout,
                                     intervals=intervals if active_range else None,
                                     partner=not same_factor)
    if not same_factor:
        return plus
    # Both faces store the same physical b. Thus (b d b†)^T = b* d b^T
    # even for complex d: transpose the all-mesh operator, never conjugate
    # its causal phase or contract the same pole columns a second time.
    from common.collectives import transpose_xy
    transposed = jax.lax.with_sharding_constraint(
        transpose_xy(plus, mesh_xy), NamedSharding(mesh_xy, P(None, "x", "y")))
    return plus, transposed


def _shared_pole_factor_specs(layout):
    return ((P(None, "x", None, "y"), P(None, "y", None, "x"))
            if layout == "face" else
            (P(None, "x", None, None), P(None, "y", None, None)))


def _shared_pole_contract(b_X, b_Y, weights, *, gemm, layout="face", intervals=None,
                          partner=False):
    """W(τ) = b d b† through G's configured face or axis contraction.

    Factors [q,mu,spin,K] use G's face placement under ``layout='face'``;
    the axis layout keeps K replicated and divides each centroid
    endpoint over its assigned mesh axis.  The result always uses both axes.
    The causal weight [q,K] is separate and replicated. The permutations
    below are local axis views, giving exactly psi_mun and psi_nmu layouts.

    ``intervals`` ``[q,2]`` (the half-open pole columns outside which
    ``weights`` is zero) contracts only those columns, each scaled by its
    weight, as G's build does with its band interval (the face route scales
    each gathered panel slice; the axis route a weighted copy of the factor,
    as before); ``gemm`` must then be planned with ``enable_active_range=True``.

    ``partner``: also ``conj(b_X) d b_Yᵀ`` (two factors, where it is not the
    transpose): on the face route from the same panel exchange
    (``face_green_product(partner=True)``), on the axis route by the same
    local GEMM on the conjugate factors.
    """
    from gw.greens_function_kernel import _build_G_face, build_G

    # Components are operator-port labels, not Green-function spinors.
    # Merge them with their own centroid axis before entering build_G;
    # CT then has different row extents but the same unit spin axis.
    b_X = b_X.reshape(b_X.shape[0], b_X.shape[1] * b_X.shape[2], 1, b_X.shape[3])
    b_Y = b_Y.reshape(b_Y.shape[0], b_Y.shape[1] * b_Y.shape[2], 1, b_Y.shape[3])
    band_range = None if intervals is None else (intervals[:, 0], intervals[:, 1])

    def operands(conj):
        x, y = (jnp.conj(b_X), jnp.conj(b_Y)) if conj else (b_X, b_Y)
        return jnp.transpose(x, (0, 2, 1, 3)), jnp.transpose(y, (0, 3, 2, 1))
    if partner and getattr(gemm, "backend", "local") != "local":
        value, pair = _build_G_face(*operands(False), gemm=gemm, phases=weights,
                                    band_range=band_range, pair=True)
        return value[:, :, 0, :, 0], pair[:, :, 0, :, 0]
    value = build_G(*operands(False), phases=weights, layout=layout, gemm=gemm,
                    band_range=band_range)
    # build_G is centroid-major (q, mu, s, nu, s'); the unit spin axes are 2, 4.
    if not partner:
        return value[:, :, 0, :, 0]
    pair = build_G(*operands(True), phases=weights, layout=layout, gemm=gemm,
                   band_range=band_range)
    return value[:, :, 0, :, 0], pair[:, :, 0, :, 0]


def _shared_pole_contract_local(b, weights, intervals, *, mesh_xy, n_parent, right=None,
                                partner=False):
    """W(τ) = b d b† with whole parents per rank: one local GEMM each, one exchange out.

    ``b`` ``[Bp,mu,K]``, ``weights`` ``[Bp,K]`` and ``intervals`` ``[Bp,2]`` are
    in ``distrib_la``'s batch layout (``Bp``: the ``n_parent`` parents padded
    to the mesh, empty intervals).  Each rank contracts its own parents over
    their live pole columns (``distrib_la.batch_gram``, the local active-range
    GEMM of G's face panels) and only W moves, batch to face: per τ node a
    rank sends its W rows, not the K-complete factor panels that a face SUMMA
    re-gathers at every node (2·(μ/p)·K per parent; at Ni 20³ P64, 18.6 GB per
    rank per node across nodes, against 0.85 GB).  The factor is held once,
    not as two faces.  ``right``: a two-factor sector's right factor ``[Bp,nu,K]``
    (``b d right†``); ``partner`` adds ``conj(b) d rightᵀ`` from the same rows.
    """
    from distrib_la import batch_gram
    return batch_gram(b, weights, intervals, mesh=mesh_xy, nbatch=n_parent, right=right,
                      partner=partner)


def _shared_pole_fixed_q_policy(header):
    """Resolve the policy from the store's authenticated TRS/grid metadata.

    Resolved once per panel, in `_shared_pole_panel_tables`, and carried in its
    `policy` entry: the unfold and the routed synthesis both need it and both
    take those tables, so neither resolves a second copy of the same header.
    """
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store

    return qgrid_trs_policy_from_shared_pole_store(header, announce=False)


def _shared_pole_panel_tables(meta, header, q_span, *, mesh_xy):
    """Authenticate packed endpoint maps and one parent's child-row panel."""
    from gw.qgrid_symmetry import shared_pole_packed_action

    qt = header["qirr"]
    packed, wraps, certificates = shared_pole_packed_action(meta, header, mesh_xy=mesh_xy)
    lo, hi = map(int, q_span)
    parent_map = np.asarray(qt["irr_idx_q"], dtype=np.int32)
    rows = np.flatnonzero((parent_map >= lo) & (parent_map < hi)).astype(np.int32)
    # A finite tangential model need not preserve every spatial little-group
    # relation exactly. The common TRS policy makes q/-q use one spatial
    # realization, as for the ordinary W producer. This changes only small
    # row metadata; the endpoint routing and all-P operator tiles are intact.
    policy = _shared_pole_fixed_q_policy(header)
    return dict(parent_span=(lo, hi), rows=rows, parent_rows=parent_map[rows] - lo,
                sym_rows=policy.unfold_sym_idx[rows], policy=policy,
                q_frac=np.asarray(qt["q_irr_frac"], dtype=np.float64)[lo:hi],
                packed_perm=packed, wraps=wraps, certificates=certificates,
                n_sym_spatial=int(qt["n_sym_spatial"]))


def _shared_pole_panel_realizer(meta, header, q_span, *, mesh_xy, tables=None):
    """The physical parent pair of one panel: ``(plus, transposed) -> (W, Wt)`` on its parent rows.

    The magnetic little-group realization, then the fixed-q TRS projection,
    both on the parent tiles.  Nothing is unfolded: the Σ door reads the pair
    through the store's q-wedge load tables (:func:`_shared_pole_q_wedge`),
    ``Wt`` on the antiunitary rows.  Endpoint maps that cross a shard refuse
    (the fused load reads only this rank's parent tile).
    """
    from common.shard_map import shard_map
    from gw.qgrid_symmetry import shared_pole_operator_realizer

    if tables is None:
        tables = _shared_pole_panel_tables(meta, header, q_span, mesh_xy=mesh_xy)
    _require_local_maps(tables["certificates"])
    policy = tables["policy"]
    qids = np.asarray(header["q_irr_full_idx"])[slice(*q_span)]
    realize = shared_pole_operator_realizer(
        meta, header, q_full_idx=qids, mesh_xy=mesh_xy)

    def body(plus, transposed):
        projected, _ = policy.project_fixed_q(
            plus, qids, transposed_partner=transposed, measure=False)
        transposed, _ = policy.project_fixed_q(
            transposed, qids, transposed_partner=plus, measure=False)
        return projected, transposed

    spec = P(None, "x", "y")
    project_local = jax.jit(shard_map(
        body, mesh=mesh_xy, in_specs=(spec, spec), out_specs=(spec, spec), check_vma=False))

    @jax.jit
    def realize_pair(plus, transposed):
        return project_local(*realize(plus, transposed))

    return realize_pair


def _require_local_maps(certificates):
    if not all(certificates[axis]["is_local"] for axis in ("x", "y")):
        raise ValueError(
            "GATE shared_pole_w_parent_local: a packed endpoint map crosses a mesh shard; "
            "W is built on the irreducible q and unfolded on the Sigma load, which reads "
            "only this rank's parent tile (TASTE 97: no full-q W route)")


_W_LOADS = {}


def _shared_pole_q_wedge(meta, header, *, mesh_xy):
    """The store's q wedge as the Σ door reads W: ``(wedge, (particle, hole) device loads)``.

    ``wedge`` is a ``symmetry_maps.QirrOperator`` of tables (no values): the
    store's parent rows ``irr_idx_q`` with the TRS policy's unfold operations,
    the packed endpoint action (:func:`gw.qgrid_symmetry.shared_pole_packed_action`)
    and the pair-transpose rule, so an antiunitary q reads ``Wt``.  mathdx
    mode 9 (``_get_sigma_kij_kernel(q_wedge=)``) unfolds W on its load.  An
    ordered store's valence branch W_+(-q)^T reads the same parent pair
    through the q-negated tables (``sector_sigma.hole_tables``); a TRS store
    has none.  The device tables are placed once per mesh and wedge.
    """
    from symmetry_maps import QirrOperator, device_load_tables
    from gw.qgrid_symmetry import shared_pole_packed_action
    from .sector_sigma import hole_tables

    qt = header["qirr"]
    packed, wraps, certificates = shared_pole_packed_action(meta, header, mesh_xy=mesh_xy)
    _require_local_maps(certificates)
    wedge = QirrOperator(
        values=None, irr_idx=np.asarray(qt["irr_idx_q"], np.int32),
        sym_idx=np.asarray(_shared_pole_fixed_q_policy(header).unfold_sym_idx, np.int32),
        sym_perm=np.asarray(packed, np.int32), L_table=np.asarray(wraps),
        q_irr_frac=np.asarray(qt["q_irr_frac"], np.float64),
        n_sym_spatial=int(qt["n_sym_spatial"]),
        full_rows=np.asarray(header["q_irr_full_idx"], np.int32), trs_rule="pair_transpose")
    ordered = header.get("representation") == "scalar-ordered-ph"
    key = (tuple(d.id for d in np.asarray(mesh_xy.devices).flat), wedge.wedge_key(),
           tuple(int(v) for v in header["grid"]), ordered)
    loads = _W_LOADS.get(key)
    if loads is None:
        particle = wedge.load_tables(mesh_xy)
        loads = _W_LOADS[key] = (
            device_load_tables(particle, mesh_xy),
            device_load_tables(hole_tables(particle, header["grid"]), mesh_xy) if ordered else None)
    return wedge, loads


def _shared_pole_at_rows(meta, header, rows, *, mesh_xy):
    """``(W, Wt) -> W_+`` at the full-q rows ``rows`` only, for the W0 restart member.

    The BSE restart stores ``W0 = V + Wc(0)`` on V's q parents
    (:func:`shared_pole_static_wc`); the unfold tables are cut to ``rows``,
    so no full-q W is formed.  The Σ route never calls this.
    """
    from common.shard_map import shard_map
    from symmetry_maps import unfold_operator_local

    nq = int(header["n_q_irr"])
    tables = _shared_pole_panel_tables(meta, header, (0, nq), mesh_xy=mesh_xy)
    _require_local_maps(tables["certificates"])
    cert = tables["certificates"]
    rows = np.asarray(rows, np.int64).reshape(-1)
    if not np.array_equal(tables["rows"], np.arange(tables["rows"].size)):
        raise ValueError("GATE shared_pole_static_w: the whole parent span must "
                         "cover every full q in order")

    def body(W, Wt):
        return unfold_operator_local(
            W, irr_idx=tables["parent_rows"][rows], sym_idx=tables["sym_rows"][rows],
            q_irr_frac=tables["q_frac"],
            left_local_perm=cert["x"]["local_perm"], left_L_table=tables["wraps"],
            right_local_perm=cert["y"]["local_perm"], right_L_table=tables["wraps"],
            n_sym_spatial=tables["n_sym_spatial"],
            trs_rule="pair_transpose", transposed_parent_local=Wt)

    spec = P(None, "x", "y")
    return jax.jit(shard_map(body, mesh=mesh_xy, in_specs=(spec, spec), out_specs=spec,
                             check_vma=False))


_SYNTHESIS_PROGRAMS = {}


def _synthesis_program(key, build):
    """One scalar shared-pole synthesis program per static configuration, per process.

    Every SC map rebuilds the shared-pole model, not the programs that consume
    it: a later map with the same shapes dispatches the first map's jit object
    and compiled executable instead of recompiling them (P2-A, claim 2735).
    ``key`` names everything a program closes over (mesh, layout, panel span
    and extents, the FFI dials, the symmetry tables by content); factors,
    poles, intervals and τ always enter as arguments, never as constants.
    """
    entry = _SYNTHESIS_PROGRAMS.get(key)
    if entry is None:
        entry = _SYNTHESIS_PROGRAMS[key] = build()
    return entry


def _shared_pole_static_key(meta, header, tables, *, mesh_xy, layout):
    """Content key of the store symmetry, packed basis and panel tables."""
    from ffi import ffi_dial_key

    basis = meta.mu_basis
    return _static_key((
        mesh_xy, layout, ffi_dial_key(),
        {name: header.get(name) for name in (
            "representation", "grid", "q_order", "q_shift", "q_irr_full_idx",
            "n_q_irr", "n_q_full", "n_mu_logical", "nspinor", "qirr", "operations")},
        (header.get("recipe") or {}).get("operator_realization"),
        (int(basis.n_packed), getattr(basis, "n_logical", None),
         getattr(basis, "mesh_xy", None), np.asarray(basis.active_mask)),
        {name: value for name, value in tables.items() if name != "policy"}))


def _shared_pole_w_synthesis(io, meta, header, frequencies, schedule, *, mesh_xy, layout="face",
                             weights_fn=_shared_pole_weights, stage="sigma"):
    """Read the factors once and bind W(τ) on the irreducible q for the window executable.

    Returns a :class:`WSynthesis` whose ``q_wedge`` is the store's q wedge
    (:func:`_shared_pole_q_wedge`).  The parent faces are read once per Σ
    call and held for the sweep; ``w_kernel`` runs inside every window
    executable and, per τ node, synthesizes each parent panel of
    ``schedule``'s admitted capacity — W_parent = b d(τ) b† and its transpose
    through G's builder, little-group realization, fixed-q projection
    (:func:`_shared_pole_panel_realizer`) — into the parent pair ``(W, Wt)``
    ``[n_q_irr, m, m]``.  It returns ``(W, Wt, load)``: the Σ door unfolds
    the pair on its load through ``load`` (the particle tables, or on an
    ordered store's valence branch the q-negated ones), so no full-q W is
    formed (TASTE 97).  Parent panels are a static loop, sequenced so that
    one panel's temporaries live at a time; pole-column chunks of one static
    width are a device ``fori_loop`` over chunk-major slices, and a single
    chunk when the budget admits every column (TASTE 96).

    ``weights_fn`` is the per-column coefficient: the causal d(τ) for Σ, or
    :func:`_shared_pole_omega0_weights` for :func:`shared_pole_static_wc`.
    ``stage`` prefixes this call's capacity-ledger stage names, which are
    unique per map.
    """
    _band_fence('tau.synthesis_plan', sync_ranks=True)
    with timing.section('tau.synthesis_plan'):
        from file_io.shared_pole_store import face_width, read_shared_pole_faces
        from .sector_sigma import _placer, _zeros

        nq = int(header["n_q_irr"])
        kmax = int(header["Kmax"])
        m = int(meta.mu_basis.n_packed)
        bcap, ccap = int(schedule["parent_capacity"]), int(schedule["column_capacity"])
        if bcap < 1 or ccap < 1:
            raise ValueError("shared-pole panel capacities must be positive")
        # Ordered stores: conduction windows use W_+(q), valence windows W_+(-q)^T.
        ordered = header.get("representation") == "scalar-ordered-ph"
        q_wedge, loads = _shared_pole_q_wedge(meta, header, mesh_xy=mesh_xy)
        if kmax == 0:
            zero = _zeros(mesh_xy, (nq, m, m))
            synthesis = WSynthesis(
                lambda _f, _p, _i, load, _ref, _time, _hole: (zero(), zero(), load),
                lambda _space, _indices, _bounds: ((), (), None, loads), lambda: (),
                lambda _result=None: None, 0, ("zero", mesh_xy, nq, m), ordered=ordered)
            synthesis.q_wedge = q_wedge
            return synthesis
        local = layout == "local"
        if local and (bcap < nq or ccap < kmax):
            raise ValueError("shared-pole synthesis: whole parents per rank take one panel")
        factor_specs = None if local else _shared_pole_factor_specs(layout)
        # One static column width: the whole laddered K when every column fits
        # (one chunk), else the admitted chunk carrier, stepped by that width.
        chunked = ccap < kmax
        width = face_width(mesh_xy, kmax, (0, ccap) if chunked else None)
        n_chunks = -(-kmax // width)
        # Query the same distributed dense context used by G before warming
        # any matrix operands. The service accepts a resolved dense Plan for
        # a GEMM workspace query, as in the shared-pole constructor.
        from distrib_la import plan, workspace_bytes_per_rank
        workspace_plan = plan("eigh", mesh_xy, n=m, backend="distributed",
                              batched_route="auto")
        native_workspace = 0
        panels = []
        for lo in range(0, nq, bcap):
            hi = min(lo + bcap, nq)
            tables = _shared_pole_panel_tables(meta, header, (lo, hi), mesh_xy=mesh_xy)
            static = _shared_pole_static_key(meta, header, tables, mesh_xy=mesh_xy, layout=layout)
            count = hi - lo
            if not local:
                native_workspace = max(native_workspace, workspace_bytes_per_rank(
                    workspace_plan, "gemm", ((count, m, width), (count, width, m)),
                    np.complex128))

            def program(span=(lo, hi), tables=tables, count=count):
                from distrib_la import gemm_plan
                # As G's plan: pole columns outside a window's interval are
                # never contracted, and no warm-up (the plan runs inside the
                # window executable).  Whole parents per rank contract locally.
                gemm = None if local else gemm_plan(
                    mesh_xy, m=m, k=width, n=m, nq=count, dtype=np.complex128, layout=layout,
                    enable_active_range=True, warmup=False)
                realize_pair = _shared_pole_panel_realizer(
                    meta, header, span, mesh_xy=mesh_xy, tables=tables)

                def body(factors, poles2, ranges, e, t):
                    plus, transposed = synthesize_shared_pole_parents(
                        *factors, poles2, ranges, e, t, mesh_xy=mesh_xy, gemm=gemm,
                        layout=layout, weights_fn=weights_fn, active_range=True)
                    return realize_pair(plus, transposed)
                return jax.jit(body)
            kind = ("synthesis" if weights_fn is _shared_pole_weights
                    else "synthesis." + weights_fn.__name__)
            kernel = _synthesis_program((static, kind, count, m, width), program)
            panels.append(dict(span=(lo, hi), kernel=kernel, static=static, count=count))
        schedule["native_gemm_workspace_bytes_per_rank"] = native_workspace
    _band_fence('tau.factor_read', sync_ranks=True)
    with timing.section('tau.factor_read'):
        capacity = getattr(meta, "shared_pole_capacity", None)
        # The stages live beside this Σ call, as the admitted schedule named them.
        ambient = (tuple(schedule["capacity_receipt"]["concurrent_with"])
                   if "capacity_receipt" in schedule else None)
        x, y, poles, _counts = read_shared_pole_faces(
            io, (0, nq), meta=meta, header=header, column_span=(0, kmax) if chunked else None,
            orientations=("x",) if local else ("x", "y"))
        if local:
            # Both faces store the same physical b: one copy, whole parents
            # per rank (distrib_la.batch_layout), placed once per Σ call.
            from distrib_la import batch_layout
            x = jax.jit(lambda a: a[:, :, 0, :],
                        out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(x)
            x, y, poles = batch_layout(x, mesh_xy), None, batch_layout(poles, mesh_xy)
        else:
            x, y = (_placer(mesh_xy, spec)(a) for spec, a in zip(factor_specs, (x, y)))
        panel_factors, panel_poles = [], []
        for panel in panels:
            lo, hi = panel["span"]
            whole = (lo, hi) == (0, nq)
            fx = (x, y) if whole else (x[lo:hi], y[lo:hi])
            pp = poles if whole else poles[lo:hi]
            if chunked:
                fx = tuple(_chunk_major(mesh_xy, factor_specs[i % 2], n_chunks, width)(f)
                           for i, f in enumerate(fx))
                pp = _chunk_major(mesh_xy, P(), n_chunks, width)(pp)
            panel_factors.append(tuple(fx))
            panel_poles.append(pp)
        del x, y, poles
        panel_factors, panel_poles = tuple(panel_factors), tuple(panel_poles)
        jax.block_until_ready((panel_factors, panel_poles))
        if ambient is not None:
            resident = sum(int(a.addressable_shards[0].data.nbytes)
                           for a in jax.tree.leaves((panel_factors, panel_poles)))
            capacity.reserve(f"{stage}.synthesis.resident", resident_bytes_per_rank=resident,
                             workspace_bytes_per_rank=0, concurrent_with=ambient)
            capacity.live_stages = (*ambient, f"{stage}.synthesis.resident")

    spans = tuple((p["span"], p["kernel"]) for p in panels)

    def w_kernel(factors_by_panel, poles_by_panel, intervals, load, e_ref, t_node, hole):
        pair = None
        for ((lo, hi), kernel), factors, poles2 in zip(spans, factors_by_panel, poles_by_panel):
            ranges = intervals[lo:hi]
            if pair is not None:
                # One panel's temporaries at a time: the next panel's
                # synthesis waits for the running pair.
                pair, factors, poles2 = jax.lax.optimization_barrier((pair, factors, poles2))
            if not chunked:
                part = kernel(factors, poles2, jnp.clip(ranges, 0, width), e_ref, t_node)
                if pair is None and (lo, hi) == (0, nq):
                    # The one all-parent panel IS the parent pair.
                    pair = part
                else:
                    base = (_zeros(mesh_xy, (nq, m, m))(),) * 2 if pair is None else pair
                    pair = tuple(b.at[lo:hi].add(v) for b, v in zip(base, part))
                continue

            def chunk(j, acc, factors=factors, poles2=poles2, ranges=ranges, lo=lo, hi=hi,
                      kernel=kernel):
                selected = jnp.clip(ranges - j*width, 0, width)

                def add(acc):
                    # Chunk j is read from the counter once, behind a barrier
                    # (the R82 remat hazard, as the window runner's node reads).
                    faces, poles_j = jax.lax.optimization_barrier((
                        tuple(jax.lax.dynamic_index_in_dim(f, j, 0, keepdims=False)
                              for f in factors),
                        jax.lax.dynamic_index_in_dim(poles2, j, 0, keepdims=False)))
                    part = kernel(faces, poles_j, selected, e_ref, t_node)
                    return tuple(a.at[lo:hi].add(v) for a, v in zip(acc, part))
                # A chunk with no active column in this window adds nothing.
                return jax.lax.cond(jnp.any(selected[:, 1] > selected[:, 0]),
                                    add, lambda acc: acc, acc)
            pair = jax.lax.fori_loop(
                0, n_chunks, chunk,
                (_zeros(mesh_xy, (nq, m, m))(),) * 2 if pair is None else pair)
        # The valence branch of an ordered store reads W_+(-q)^T through the
        # q-negated tables: the same parent pair, another load.
        return (*pair, load[1] if hole else load[0])

    def window_operands(_space, indices, bounds):
        # Host intervals once per window; every τ node of the window reuses them.
        intervals = shared_pole_intervals(frequencies, np.asarray(indices), np.asarray(bounds))
        return (panel_factors, panel_poles,
                device_put_process_local(intervals, NamedSharding(mesh_xy, P())), loads)

    closed = False

    def close(result=None):
        nonlocal panel_factors, panel_poles, closed
        if closed:
            return
        try:
            jax.block_until_ready(result if result is not None
                                  else (panel_factors, panel_poles))
        finally:
            panel_factors = panel_poles = None
            if ambient is not None:
                capacity.live_stages = ambient
            closed = True

    key = ("scalar-parent", mesh_xy, nq, m, width, n_chunks, ordered, q_wedge.wedge_key(),
           tuple((p["span"], p["count"], p["static"]) for p in panels))
    if weights_fn is not _shared_pole_weights:
        key += (weights_fn.__name__,)
    synthesis = WSynthesis(w_kernel, window_operands, lambda: (panel_factors, panel_poles),
                           close, native_workspace, key, ordered=ordered,
                           tile_bytes=16 * nq * m * m // int(mesh_xy.size))
    synthesis.q_wedge = q_wedge
    return synthesis


def shared_pole_static_wc(handle, meta, *, mesh_xy, rows, layout="face"):
    """Wc(q, omega = 0) of the current scalar shared-pole model at the full-q rows ``rows``.

    The static screened correction the restart stores for BSE
    (``W0_qmunu = V + Wc(0)``). It is evaluated by the Σ synthesis above —
    the same factor read, little-group realization and fixed-q projection
    on the parents — with :func:`_shared_pole_omega0_weights` in place of
    d(τ), so the store keeps one evaluator; the parent pair is then unfolded
    at ``rows`` only (:func:`_shared_pole_at_rows`): the q parents of the
    run's V wedge, which the restart stores (BSE unfolds on load), so no
    full-q W is formed (TASTE 97). The two branches of the time-ordered W
    enter as the Σ consumer routes them: W_+(q) and, on an ordered store,
    W_+(-q)^T (SP 5), read at the rows -q; a TRS store's valence branch is
    W_+ itself, which gives ``-b Λ^-1 b†``
    (docs/architecture/shared_pole_model.md §7).

    Returns ``(len(rows), m, m)`` complex128 at ``P(None,'x','y')`` in the
    run's packed centroid order. Every array is an all-P tile. The synthesis
    admits its factors and panel workspace under ``w0.*`` ledger stages; the
    two outputs at ``rows`` (W_+ and the sum) are one more reservation of 2U.

    Validated (claim 2856, runs/CrI3/504_w0persist_20260926): on CrI3
    8x8x1 SOC (mu 1446, 10 IBZ q) ``V + Wc(0)`` matches the GN-PPM Dyson
    W(0) to 1.47e-5 max-relative and 1.99e-5 Frobenius in W0, 2.3e-5
    Frobenius in Wc (4e-6 at Gamma), with identical V and heads.  The BSE
    E_1..E_5 on the two restarts agree within 0.07 meV.
    """
    from file_io.shared_pole_store import (open_shared_pole_model,
                                           validate_shared_pole_model)
    from symmetry_maps import q_negation_index

    capacity = meta.shared_pole_capacity
    header = validate_shared_pole_model(
        handle["path"], expected_identity=handle["identity"], mesh_xy=mesh_xy,
        capacity=capacity)
    if header["digest"] != handle["digest"]:
        raise ValueError("GATE shared_pole_identity: W0 handle digest differs from model")
    if header.get("representation") not in ("scalar-trs-even-s", "scalar-ordered-ph"):
        raise ValueError("GATE shared_pole_static_w: a scalar charge store is required; "
                         f"got representation {header.get('representation')!r}")
    ordered = header["representation"] == "scalar-ordered-ph"
    Q, m = int(header["n_q_full"]), int(meta.mu_basis.n_packed)
    rows = np.asarray(rows, np.int64).reshape(-1)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= Q:
        raise ValueError(f"GATE shared_pole_static_w: rows must be full-q rows in [0, {Q})")
    ambient = capacity.live_stages
    tile = -(-16 * rows.size * m * m // int(mesh_xy.size))
    capacity.reserve("w0.static_output", resident_bytes_per_rank=2 * tile,
                     workspace_bytes_per_rank=0, concurrent_with=ambient)
    capacity.live_stages = (*ambient, "w0.static_output")
    try:
        schedule = _shared_pole_memory_schedule(meta, header, mesh_xy=mesh_xy,
                                                layout=layout, stage="w0")
        counts = np.asarray(header["K"], np.int64)
        intervals = device_put_process_local(
            np.stack([np.zeros_like(counts), counts], axis=1),
            NamedSharding(mesh_xy, P()))
        minus_q = np.asarray(q_negation_index(tuple(int(v) for v in header["grid"])))
        with open_shared_pole_model(handle["path"], mesh_xy=mesh_xy) as reader:
            synthesis = _shared_pole_w_synthesis(
                reader, meta, header, None, schedule, mesh_xy=mesh_xy,
                layout=schedule.get("factor_layout", layout),
                weights_fn=_shared_pole_omega0_weights, stage="w0")
        from common.collectives import transpose_xy
        plus_at = _shared_pole_at_rows(meta, header, rows, mesh_xy=mesh_xy)
        # W_-(q) = W_+(-q)^T: W_+ at the rows -q, transposed in its endpoint faces.
        minus_at = (_shared_pole_at_rows(meta, header, minus_q[rows], mesh_xy=mesh_xy)
                    if ordered else None)
        face = NamedSharding(mesh_xy, P(None, "x", "y"))

        @partial(jax.jit, out_shardings=face)
        def static(factors, poles, intervals):
            W, Wt, _ = synthesis.w_kernel(factors, poles, intervals, (None, None),
                                          0.0, 0.0, False)
            plus = plus_at(W, Wt)
            return plus + (transpose_xy(minus_at(W, Wt), mesh_xy) if ordered else plus)
        wc = None
        try:
            wc = static(*synthesis.resident_operands(), intervals)
        finally:
            synthesis.close(wc)
    finally:
        capacity.live_stages = ambient
    return wc


@lru_cache(maxsize=None)
def _chunk_major(mesh_xy, spec, n_chunks, width):
    """``[..., K] -> [n_chunks, ..., width]`` so a chunk is a local leading-axis index.

    Pads the pole columns to ``n_chunks*width`` (zero factors, unit poles:
    zero weight) and keeps the given face or replicated placement per chunk.
    """
    def split(a):
        pad = n_chunks*width - a.shape[-1]
        if pad > 0:
            a = jnp.pad(a, [(0, 0)]*(a.ndim-1) + [(0, pad)],
                        constant_values=1.0 if a.dtype == jnp.float64 else 0.0)
        a = a[..., :n_chunks*width].reshape(*a.shape[:-1], n_chunks, width)
        return jnp.moveaxis(a, -2, 0)
    return jax.jit(split, out_shardings=NamedSharding(mesh_xy, P(None, *spec)))


def _w_extents(meta, extents):
    """``(rows, cols)`` of one W(τ) parent tile, components merged with their centroids:
    the scalar store's ``(μ, μ)``, or a photon sector's ``(m·n_A, n·n_B)``."""
    if extents is None:
        m = int(meta.mu_basis.n_packed)
        return m, m
    return tuple(int(e) for e in extents)


def _shared_pole_panel_cost(meta, header, b, c, *, mesh_xy, layout="face", extents=None):
    """Price the new buffers of one b-parent x c-column synthesis panel.

    Coordinator ruling12 separates unchanged spatial/ψ/Σ peak regression
    from this three-U admission.  W stays on the parent rows (the Σ door
    unfolds it on its load), so no child W tile is priced.  ``extents``: a
    photon sector's ``(rows, cols)`` (:func:`_w_extents`).
    """
    # Factors and W tiles are mu x mu charge operators (factor spin axis 1)
    # on scalar and two-component decks; G alone carries the spinor axes.
    rows, cols = _w_extents(meta, extents)
    nq = int(header["n_q_irr"])
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    tile = 16 * rows*cols // (px*py)
    multiple = combined_divisor(px,py)
    c = padded_axis(c,multiple,name="shared_pole_K_chunk").carrier
    faces = 16 * b * c*(rows+cols) / (px*py)
    # Parent, partner and fixed-size group accumulators coexist. The
    # compiled reservation below measures actual aliases and exchange
    # scratch; this bound also informs the panel-size search.
    peak = 6*b*tile + 8*b + 40*b*c*(rows+cols)/(px*py) + 64*b*c
    if layout == "axis":
        # (The square form is the scalar's own float expression.)
        axis_faces = (16*b*rows*c*(1/px+1/py) if rows == cols
                      else 16*b*c*(rows/px + cols/py))
        # Retained axis factors, the reader's face carrier, and weighted
        # GEMM operands coexist.
        peak += axis_faces + axis_faces
        faces = axis_faces
    return dict(resident_bytes_per_rank=int(np.ceil(faces)),
                workspace_bytes_per_rank=int(np.ceil(peak-faces)))


def _shared_pole_resident_bytes(meta, header, *, mesh_xy, layout, extents=None):
    """Per-rank bytes of the factors the synthesis holds for a whole Σ call.

    All n_q_irr parent faces at the store's whole-K carrier K̄, both
    orientations: 32·n·μ·K̄/P on the face layout, 16·n·μ·K̄·(1/Px+1/Py) with
    the pole columns replicated (axis layout), plus the replicated poles
    8·n·K̄ (a sector: μ·K̄ and ν·K̄ for its two endpoints, ``extents``).
    """
    from file_io.shared_pole_store import face_width

    rows, cols = _w_extents(meta, extents)
    nq = int(header["n_q_irr"])
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    k = face_width(mesh_xy, int(header["Kmax"]))
    pair = (16*(rows+cols)*k/(px*py) if layout == "face" else
            16*rows*k*(1/px + 1/py) if rows == cols else 16*k*(rows/px + cols/py))
    return int(np.ceil(nq*pair + 8*nq*k))


def _shared_pole_local_price(meta, header, *, mesh_xy, extents=None, factors=1):
    """Per-rank ``(resident, workspace)`` bytes of the synthesis with whole parents per rank.

    Resident: the one factor ``[ceil(nq/P), μ, K̄]`` (a two-factor sector,
    ``factors=2``: both, ``μ + ν``) and its poles.  Workspace: one parent's two
    GEMM operands, the rank's W rows (and a two-factor sector's partner rows)
    before the exchange, and the parent pair on the face with its realization
    temporaries (as :func:`_shared_pole_panel_cost`'s ``6·tile``).
    """
    from file_io.shared_pole_store import face_width

    m, n = _w_extents(meta, extents)
    nq = int(header["n_q_irr"])
    ranks = int(mesh_xy.size)
    k = face_width(mesh_xy, int(header["Kmax"]))
    rows = -(-nq // ranks)
    resident = 16 * rows * (m if factors == 1 else m + n) * k + 8 * rows * k
    workspace = (16 * (m + n) * k + 2 * factors * 16 * rows * m * n
                 + 6 * -(-16 * nq * m * n // ranks))
    return int(resident), int(workspace)


def _shared_pole_memory_schedule(meta, header, *, mesh_xy, layout="face", stage="sigma",
                                 linalg=None, extents=None, factors=1):
    """Price the resident factors, size the τ panels from what is left, admit.

    The factors are read once per Σ call and stay resident
    (:func:`_shared_pole_resident_bytes`); only the synthesis workspace of one
    parent panel × pole-column chunk depends on (b, c), and the chunk count
    degenerates to one when everything fits (TASTE 96).  Caller-bound
    live_stages charge other NEW shared-pole objects. Per coordinator
    ruling12, the unchanged spatial/ψ/Σ footprint and its one full-q W are
    reported separately against the incumbent (<=1.05x).

    Placement, in order: the replicated pole columns (``axis``) when the panel
    search admits them (no exchange per node); else, on a ``linalg = local``
    deck (the resolved ``linalg``) whose whole parents fit per rank
    (``shared_pole_execution.whole_parent_execution``, the bank's rule), whole
    parents (``factor_layout='local'``, :func:`_shared_pole_contract_local`:
    only W(τ) moves); else the face SUMMA panels.

    A photon sector (``gw.mpa.sector_sigma.sector_synthesis``) takes the same
    rule with its tile ``extents`` (rows, cols) and ``factors`` (2: a mixed
    CT/TC sector's two factors): its synthesis is one panel of every parent
    and pole column, so only that panel is admissible (the face route always
    is: its SUMMA panels are bounded by one W tile from the shapes).
    """
    capacity = getattr(meta, "shared_pole_capacity", None)
    if capacity is None:
        raise ValueError("GATE shared_pole_capacity: missing current-map CapacityLedger")
    concurrent = capacity.live_stages
    accepted = {row["stage"]: row for row in capacity.entries}
    caller_bytes = sum(accepted[name][key] for name in concurrent for key in
                       ("resident_bytes_per_rank", "workspace_bytes_per_rank"))
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    n, spin, Q = (int(header[key]) for key in
                  ("n_mu_logical", "nspinor", "n_q_full"))
    nq, kmax = int(header["n_q_irr"]), int(header["Kmax"])
    # Compare the geometry itself, in integers. The derived unit is
    # 16*Q*(spin*mu)^2 before the mesh divides it; that numerator passes 2**53
    # at the sizes LORRAX targets, where float equality stops being exact.
    sector = extents is not None
    if not sector and tuple(int(capacity.geometry[key]) for key in
             ("nq", "nspinor", "nmu", "px", "py")) != (Q, spin, n, px, py):
        raise ValueError("GATE shared_pole_capacity: store/current-map geometry mismatch")
    U = capacity.U_bytes_per_rank
    if kmax == 0:
        receipt = capacity.reserve(f"{stage}.synthesis", resident_bytes_per_rank=0,
                                   workspace_bytes_per_rank=0, concurrent_with=concurrent)
        return dict(status=receipt["status"],parent_capacity=nq,column_capacity=1,
                    capacity_receipt=receipt,route="empty")
    if not sector:
        tables = _shared_pole_panel_tables(meta, header, (0,nq), mesh_xy=mesh_xy)
        _require_local_maps(tables["certificates"])
    rows_w, cols_w = _w_extents(meta, extents)
    # The ledger owns the hardware limit (ruling24); 3U is a scaling
    # receipt. A zero-byte planning reservation prices the existing ambient set.
    admission = capacity.reserve(
        f"{stage}.panel_budget", resident_bytes_per_rank=0,
        workspace_bytes_per_rank=0, concurrent_with=concurrent)
    budget = math.floor(admission["available_device_bytes_per_rank"]
                        - admission["aggregate_bytes_per_rank"])
    multiple = combined_divisor(px,py)

    def workspace(b, c, layout):
        return _shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh_xy,
                                       layout=layout,extents=extents)["workspace_bytes_per_rank"]

    def search(layout):
        best = None
        if sector:
            left = budget - _shared_pole_resident_bytes(
                meta, header, mesh_xy=mesh_xy, layout=layout, extents=extents)
            return (1, -nq*kmax, nq, kmax) if workspace(nq, kmax, layout) <= left else None
        for b in range(1,nq+1):
            left = budget - _shared_pole_resident_bytes(
                meta, header, mesh_xy=mesh_xy, layout=layout)
            # The physical logical-U bound applies to every NEW projector
            # matrix, even when orbit packing pads the endpoint carrier. The
            # pre-existing full-q Sigma output is accounted separately above.
            if 16*b*meta.mu_basis.n_packed**2/(px*py) > U:
                continue
            # workspace(j column multiples) = intercept + j*slope; both ends are
            # ceilinged byte counts, so a non-positive slope carries no width
            # information. Integer floor division keeps the sizing exact.
            p1, p2 = workspace(b, multiple, layout), workspace(b, 2*multiple, layout)
            slope, intercept = p2 - p1, 2*p1 - p2
            if slope <= 0:
                continue
            c = min(kmax, multiple*((left - intercept)//slope))
            if c < 1:
                continue
            cost = ((nq+b-1)//b)*((kmax+c-1)//c)
            candidate = (cost,-b*c,b,c)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return best

    best = search(layout)
    # The factors do not depend on tau; only the causal weight does. Face
    # factors make every tau's GEMM a per-q distributed SUMMA that
    # re-broadcasts the same panels; pole columns replicated (axis
    # orientation) make it local. Take that placement whenever it needs no
    # more panel passes than the face schedule (the sector route measured
    # Fe 4^3 Sigma tau 69.4 -> 42.0 s).
    if layout == "face":
        replicated = search("axis")
        if replicated is not None and (best is None or replicated[0] <= best[0]):
            layout, best = "axis", replicated
    # Where the replicated columns do not fit, whole parents per rank keep
    # the contraction local and move only W (one tile per node), against the
    # face SUMMA's per-node re-gather of every factor panel.
    if layout != "axis" and linalg == "local" and (
            sector or int(header.get("factor_components", 1)) == 1):
        from gw.shared_pole_execution import whole_parent_execution
        execution, whole_resident, whole_workspace = whole_parent_execution(
            lambda e: (_shared_pole_local_price(meta, header, mesh_xy=mesh_xy, extents=extents,
                                                factors=factors) if e == "local"
                       else (_shared_pole_resident_bytes(meta, header, mesh_xy=mesh_xy,
                                                         layout=layout, extents=extents), 0)),
            ledger=capacity)
        if execution == "local":
            receipt = capacity.reserve(
                f"{stage}.synthesis", resident_bytes_per_rank=whole_resident,
                workspace_bytes_per_rank=whole_workspace, concurrent_with=concurrent)
            return dict(status=receipt["device_budget_status"], unit_bytes=U,
                        factor_layout="local",
                        peak_live_bytes_per_rank=receipt["aggregate_bytes_per_rank"],
                        peak_in_U=receipt["aggregate_bytes_per_rank"]/U,
                        parent_capacity=nq, column_capacity=kmax,
                        resident_factor_bytes_per_rank=whole_resident,
                        caller_live_bytes_per_rank=caller_bytes, capacity_receipt=receipt,
                        route="local_parent",
                        inherited_sigma_peak_status="NOT_MEASURED",
                        projection_matrix_bytes_per_rank=int(16*nq*rows_w*cols_w
                                                             / (px*py)))
    b,c = (nq,kmax) if sector else (1,multiple) if best is None else best[2:]
    footprint = _shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh_xy,layout=layout,
                                        extents=extents)
    resident = _shared_pole_resident_bytes(meta, header, mesh_xy=mesh_xy, layout=layout,
                                           extents=extents)
    projection_bytes = 16*b*rows_w*cols_w/(px*py)
    if not sector and projection_bytes > U:
        from common.gpu_utils import warn_over_budget
        warn_over_budget(f"{stage} parent star (all-P logical bound)", projection_bytes, U)
    receipt = capacity.reserve(
        f"{stage}.synthesis", resident_bytes_per_rank=resident,
        workspace_bytes_per_rank=footprint["workspace_bytes_per_rank"],
        concurrent_with=concurrent)
    return dict(status=receipt["device_budget_status"], unit_bytes=U,
                factor_layout=layout,
                peak_live_bytes_per_rank=receipt["aggregate_bytes_per_rank"],
                peak_in_U=receipt["aggregate_bytes_per_rank"]/U,
                parent_capacity=b,column_capacity=c,
                resident_factor_bytes_per_rank=resident,
                caller_live_bytes_per_rank=caller_bytes,capacity_receipt=receipt,
                route="local_parent",
                inherited_sigma_peak_status="NOT_MEASURED",
                projection_matrix_bytes_per_rank=int(projection_bytes))


def _geometry_residue(B, B_odd):
    """A nonzero witness for either ordered residue, or incumbent ``B``."""
    if B_odd is None:
        return B
    plus = B + B_odd
    minus = B - B_odd
    return jnp.where(jnp.abs(plus) > 0.0, plus, minus)


def _refuse_nonfinite_pole_slab(lo, Omega, B, B_odd=None):
    """Finite-reduce one streamed pole slab before it reaches any planner."""
    arrays = [("Omega_p", Omega), ("B_p", B)]
    if B_odd is not None:
        arrays.append(("B_odd_p", B_odd))
    finite = jnp.stack([jnp.all(jnp.isfinite(value))
                        for _, value in arrays])
    flags = np.asarray(jax.device_get(finite), dtype=bool)
    if np.all(flags):
        return
    bad = [name for (name, _), ok in zip(arrays, flags) if not ok]
    width = int(Omega.shape[0])
    raise ValueError(
        "MPA streamed pole slab contains non-finite payload before window "
        f"planning/execution: pole_range=[{int(lo)},{int(lo) + width}), "
        f"datasets={bad}")


def _bounded_pole_batch_size(value):
    size = int(value)
    if not 1 <= size <= 8:
        raise ValueError("MPA pole_batch_size must be in [1, 8]")
    return size


def _batch_rows(row, batch):
    """Relocalize one window's pole ranges into a fixed batch-width carrier.

    The tau kernel's executable signature must not depend on how many poles a
    particular pane/product window selects.  Inactive rows therefore occupy
    the remaining batch slots with an impossible ``a`` interval.  All windows
    over a resident batch then call the same jitted callable with identical
    shapes, dtypes, and shardings; only the selector values change. The
    returned int32 count bounds the occupied prefix so empty slots do not
    execute the pole-field arithmetic.
    """
    batch = tuple(int(p) for p in batch)
    local = {int(p): i for i, p in enumerate(batch)}
    keep = [i for i, p in enumerate(row.pole_indices) if int(p) in local]
    if not keep:
        return None
    capacity = len(batch)
    if len(keep) > capacity:
        raise ValueError(
            "one MPA window selects a resident pole more than once; "
            f"{len(keep)} selector rows exceed batch width {capacity}")
    count = len(keep)
    pole_indices = np.zeros(capacity, dtype=np.int32)
    pole_indices[:count] = [local[int(row.pole_indices[i])] for i in keep]
    # ``a > +inf`` is false for every finite pole.  Keeping all six values
    # finite-or-infinite (never NaN) also makes the inactive path harmless
    # under XLA predicate motion.
    bounds = np.broadcast_to(
        np.asarray((np.inf, -np.inf, np.inf, np.inf, -np.inf, -np.inf),
                   np.float64),
        (capacity, 6),
    ).copy()
    bounds[:count] = np.asarray(row.bounds[keep], np.float64)
    phase_real = np.zeros(capacity, dtype=bool)
    phase_real[:count] = np.asarray(row.phase_real[keep], bool)
    return (
        pole_indices,
        bounds,
        phase_real,
        np.int32(count),
    )


def _admit(compiled, meta, stage, *, native=0, resident=0, counted=0, peak=None):
    """Reserve a compiled executable's peak; ``counted`` argument bytes are charged elsewhere.
    ``peak`` passes an ``aot_kernel_peak_bytes`` breakdown the caller already read."""
    from runtime.aot_memory import aot_kernel_peak_bytes
    if peak is None:
        peak = aot_kernel_peak_bytes(compiled)
    if not peak.cufft_measured:
        raise ValueError(f"GATE shared_pole_capacity: {stage} FFT workspace unavailable")
    meta.shared_pole_capacity.reserve(
        stage, resident_bytes_per_rank=resident,
        workspace_bytes_per_rank=max(0, peak.total - counted) + native,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    return compiled


def _numeric_table(value):
    """A nested list or tuple of Python numbers as one array, else ``None``.

    ``None`` too when the array would not keep every value exactly: a float or
    complex table with a magnitude at or above 2**53, where an integer
    promoted to float64 can merge with its neighbour (``[0.5, 2**53+1]`` and
    ``[0.5, 2**53]``), so the caller keys it element by element.
    """
    if not value or not all(isinstance(v, (bool, int, float, complex, np.number, list, tuple))
                            for v in value):
        return None
    try:
        table = np.asarray(value)
    except (ValueError, OverflowError):  # ragged, or an integer outside every dtype
        return None
    if table.dtype.kind not in "biufc":
        return None
    if table.dtype.kind in "fc" and table.size and not np.abs(table).max() < 2.0**53:
        return None
    return table


def _static_key(value):
    """Hashable content key of a small table tree (arrays by bytes digest).

    A store header carries its symmetry tables as nested lists (qirr's
    permutations, wraps and orbits); each is digested as one array, not
    element by element (0.35 s per SC map on Na 8^3, 48 operations).
    """
    import hashlib
    if isinstance(value, dict):
        return tuple(sorted((k, _static_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        table = _numeric_table(value)
        if table is None:
            return tuple(_static_key(v) for v in value)
        value = table
    if isinstance(value, (np.ndarray, jax.Array)):
        a = np.asarray(value)
        return ('array', a.shape, a.dtype.str,
                hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return value


class WSynthesis:
    """A shared-pole model's bound W(τ) synthesis: kernel, per-window operands, lifetime.

    ``w_kernel(*window_operands(space, indices, bounds), ref, time, hole)`` is
    W(τ) in its consumer's operand layout and is traceable: it runs inside the
    window executable.  ``window_operands`` does the host work once per window
    (the pole intervals).  ``resident_operands()`` are the device factors its
    stage already charged, ``native`` its GEMM's native workspace, ``close``
    releases them, ``key`` is the static configuration ``w_kernel`` closes
    over, and ``ordered`` says whether the valence branch reads -q (``hole``).
    ``tile_bytes`` is one W(τ) parent tile's bytes per rank.
    """

    def __init__(self, w_kernel, window_operands, resident_operands, close, native, key,
                 *, ordered, tile_bytes=0):
        self.key = key
        self.tile_bytes = int(tile_bytes)
        self.w_kernel = w_kernel
        self.window_operands = window_operands
        self.resident_operands = resident_operands
        self.close = close
        self.native = int(native)
        self.ordered = bool(ordered)


_SYNTHESIS_TAU = {}


class SynthesisTau:
    """One τ body for the window executable: W(τ) synthesis and Σ(τ) in one program.

    Serves the scalar shared-pole route and every photon sector.
    ``window_kernel(space)`` is the traceable ``fn(*arguments, t, active_count)``
    of :meth:`DeviceOmegaAccumulator.integrate_window`, one per branch hole on
    an ordered model; ``window_arguments`` swaps in the right endpoint's
    operands and the synthesis's per-window operands.  Neither closes over a
    device buffer, so the accumulator's runner cache retains no factors.
    ``door`` (the Σ door's placed load tables, ``ppm_tau_kernel.sigma_door_tables``)
    rides the window arguments too and reaches ``spatial`` as its last
    argument, so no window program holds table constants.  ``overlap``: the
    window runs two nodes per loop trip (``gw.ppm_accumulators.WINDOW_OVERLAP``).
    """

    def __init__(self, spatial, synthesis, right_yr, right_proj, native, stage, meta, key, plans,
                 door=None, overlap=False):
        self._spatial, self._synthesis = spatial, synthesis
        self.overlap = bool(overlap)
        self._right = (right_yr, right_proj)
        self._native, self._stage, self._meta = native, stage, meta
        self._key, self._plans = key, plans
        self._door = door
        self._admitted = False

    def window_kernel(self, space):
        """The τ body for ``space``, one function object per static configuration.

        The window runner is cached on this object, so returning the first
        map's body for an equal configuration (same shapes, mesh, layout,
        parent plans and W synthesis) lets every later SC map dispatch the
        compiled window executable instead of recompiling it.
        """
        hole = space == 'val' and self._synthesis.ordered
        key = (self._key, self._synthesis.key, hole)
        if key not in _SYNTHESIS_TAU:
            spatial, w_kernel = self._spatial, self._synthesis.w_kernel

            def tau(xn, yr, xr, yn, energies, weight, w_operands, e_ref_a, e_ref_b, door, t,
                    _active):
                interactions = w_kernel(*w_operands, e_ref_b, t, hole)
                if door is None:
                    return spatial(xn, yr, xr, yn, energies, weight, e_ref_a, t, interactions)
                return spatial(xn, yr, xr, yn, energies, weight, e_ref_a, t, interactions, door)
            # The plans ride along so the ids in the key cannot be reused.
            _SYNTHESIS_TAU[key] = (self._plans, tau)
        return _SYNTHESIS_TAU[key][1]

    def window_arguments(self, xn, xr, energies, weight, e_ref_a, e_ref_b, space, indices, bounds):
        w_operands = self._synthesis.window_operands(space, indices, bounds)
        return (xn, self._right[0], xr, self._right[1], energies, weight, w_operands,
                e_ref_a, e_ref_b, self._door)

    def fits(self, compiled):
        """Whether ``compiled`` fits the budget beside the live stages.

        :meth:`admit`'s ledger row, priced by ``CapacityLedger.preview`` and
        not recorded: shapes and the ledger, so every rank decides alike.
        """
        from runtime.aot_memory import aot_kernel_peak_bytes
        counted = sum(int(x.addressable_shards[0].data.nbytes)
                      for x in jax.tree.leaves(self._synthesis.resident_operands()))
        peak = aot_kernel_peak_bytes(compiled)
        ledger = self._meta.shared_pole_capacity
        row = ledger.preview(resident_bytes_per_rank=0,
                             workspace_bytes_per_rank=max(0, peak.total - counted) + self._native,
                             concurrent_with=ledger.live_stages)
        return row["device_budget_status"] == "PASS"

    def admit(self, compiled, arguments):
        """Reserve the first window executable; the resident factors are the synthesis's stage."""
        if self._admitted:
            return
        counted = sum(int(x.addressable_shards[0].data.nbytes)
                      for x in jax.tree.leaves(self._synthesis.resident_operands()))
        from runtime.aot_memory import aot_kernel_peak_bytes
        with timing.section('tau.memcheck'):
            peak = aot_kernel_peak_bytes(compiled)
        _admit(compiled, self._meta, self._stage, native=self._native, counted=counted,
               peak=peak)
        self._admitted = True
        from gw.ppm_tau_kernel import sigma_pass_price
        plan = sigma_pass_price(self._spatial)
        if plan is not None:
            # The priced pass (sigma_spin_block's x blocks, or the sub-tile
            # kernel's row passes), from the window executable
            # (runtime.aot_memory): its new bytes beside the synthesis's resident
            # operands plus the synthesis GEMM's native workspace.
            from common.gpu_utils import record_stage_price
            got = int(peak.resident_increment) + int(self._native)
            shape = (f"{plan['passes']} row pass(es)" if "passes" in plan
                     else f"d={plan['d']}/{plan['ns']}")
            # Overlapped: the second node's W pair and its synthesis output
            # are live beside the first node's passes.
            second = 3 * self._synthesis.tile_bytes if self.overlap else 0
            if second:
                shape += ", two nodes per trip"
            record_stage_price(f"Sigma tau, compiled window {shape}",
                               counted + max(plan["new"] + second, got), section="sigma.tau_sweep")


def _integrate_sigma_batches(
    wfns,
    batches,
    n_poles,
    plan,
    omega_grid_ry,
    meta,
    mesh_xy,
    *,
    pole_batch_size,
    brackets=None,
    band_counts=None,
    odd_residue_off=False,
    w_synthesis=None,
    tau_kernel_factory=None,
    q_wedge=None,
    tau_capacity=None,
    print_fn,
):
    """One spatial executor for streamed fit slabs: one executable per window.

    Each planned window runs its τ nodes in one device ``fori_loop``
    (:meth:`DeviceOmegaAccumulator.integrate_window`).  A shared-pole model
    (``w_synthesis``, a :class:`WSynthesis`) synthesizes W(τ) inside that
    body through :class:`SynthesisTau`: the scalar route over the shared
    ``sigma_kij``, a photon sector over its own spatial door
    (``tau_kernel_factory``).

    ``q_wedge``: the slabs are on the q wedge (GN-PPM, owner 2026-09-25);
    W(τ) is built on the wedge and read by the Sigma convolution through the
    wedge's unfold tables (mathdx mode 9), whose device tables ride the
    residue argument."""
    # Band fences are profiling boundaries (see ``_unfenced``).  This executor
    # is shared with the incumbent elementwise-MPA route (``w_synthesis is
    # None``), which is never fenced whatever a harness has installed.
    synthesis = w_synthesis is not None
    fence = _band_fence if synthesis else _unfenced
    fence('tau.setup', sync_ranks=True)
    with timing.section('tau.setup'):
        omega = np.asarray(omega_grid_ry, np.float64)
        if omega.ndim != 1 or not omega.size:
            raise ValueError("omega_grid_ry must be a nonempty vector")
        s = wfns.slices
        sigma_axis = sigma_band_axis(
            int(s.nb_sigma), mesh_xy, ansatz="dynamic")
        bracketed = brackets is not None
        if bracketed:
            brackets = tuple(
                (int(lo), None if hi is None else int(hi))
                for lo, hi in brackets)
            if not brackets:
                raise ValueError("MPA Sigma band-bracket plan must be nonempty")
        face_kwargs = sigma_face_kernel_kwargs(wfns)
        k_unfold_plan = wfns.green_parent.plan
        (psi_coh_xn, psi_coh_yr,
         psi_proj_xr, psi_proj_yn, _, _) = parent_sigma_operands(wfns)
        psi_proj_xr = pad_to_axis(
            psi_proj_xr, sigma_axis, axis=1)
        psi_proj_yn = pad_to_axis(
            psi_proj_yn, sigma_axis, axis=3)
        if synthesis or (tau_kernel_factory is None and q_wedge is not None):
            # The q-wedge Σ kernel and the photon sectors' τ body run row passes
            # from band-complete ψ (gw.ppm_tau_kernel._sigma_subtile_kernel,
            # gw.mpa.sector_sigma.sector_tau_factory): placed here, once per Σ
            # call, so no τ node exchanges ψ.
            from gw.ppm_tau_kernel import sigma_subtile_operands
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn = sigma_subtile_operands(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, mesh_xy=mesh_xy)
        spatial_shape = (
            int(k_unfold_plan.n_parent), sigma_axis.carrier, sigma_axis.carrier)
        face_kwargs["face_band_extent"] = sigma_axis.carrier
        shape = (omega.size, *spatial_shape)
        output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
        reduce = None
        if bracketed:
            # The sweep folds only what the band counts keep: the matrix of
            # the first and the last count and the diagonals between.
            reduce = _band_count_reduce(mesh_xy, len(brackets))
            n_between = max(len(brackets) - 2, 0)
            shape = ((shape,) * (len(brackets) - n_between),
                     ((omega.size, spatial_shape[0], sigma_axis.carrier,
                       int(mesh_xy.shape["y"])),) * n_between)
        accumulator = DeviceOmegaAccumulator(
            omega, shape=shape, sharding=output_sharding, omega_axis=0,
            reduce=reduce)
        kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

        if tau_kernel_factory is not None:
            tau_kernel = tau_kernel_factory(w_synthesis, sigma_axis)
        elif synthesis:
            # The scalar shared-pole route: the shared sigma_kij reads the
            # parent pair its synthesis builds inside the same τ body, and
            # unfolds it on the transform's load (mathdx mode 9, the q wedge).
            from gw.ppm_tau_kernel import sigma_door_tables, sigma_pass_price
            sigma_kij = _get_sigma_kij_kernel(
                mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True, brackets=brackets,
                q_wedge=w_synthesis.q_wedge, **face_kwargs)

            def scalar_spatial(xn, yr, xr, yn, energies, weight, e_ref, t, interactions,
                               g_load, sigma_kij=sigma_kij):
                return sigma_kij(xn, yr, xr, yn, energies, weight, e_ref, t, *interactions,
                                 g_load)
            scalar_spatial.price = sigma_pass_price(sigma_kij)
            spatial_key = ("scalar-parent", mesh_xy, kgrid, brackets,
                           w_synthesis.q_wedge.wedge_key(),
                           tuple(sorted(face_kwargs.items(), key=lambda kv: kv[0])))
            # The Green door's tables, placed once per run and plan: a window argument.
            # Two nodes per loop trip (gw.ppm_accumulators.WINDOW_OVERLAP): one
            # node's W exchange runs beside the other's k-convolutions, when
            # the paired window fits (SynthesisTau.fits, at the first compile).
            tau_kernel = SynthesisTau(
                scalar_spatial, w_synthesis, psi_coh_yr, psi_proj_yn, w_synthesis.native,
                "sigma.synthesis.window", meta, spatial_key, (k_unfold_plan,),
                door=sigma_door_tables(mesh_xy, k_unfold_plan), overlap=True)
        else:
            tau_kernel = get_shared_sigma_tau_kernel(
                mesh_xy=mesh_xy, kgrid=kgrid, brackets=brackets,
                q_wedge=q_wedge, **face_kwargs)
        # The residues' carrier on the wedge: the pair-transpose tables with
        # their device load, placed once per run (see ppm_tau_kernel).
        q_pair = (None if q_wedge is None else dataclasses.replace(
            q_wedge, values=None, load=None, trs_rule="pair_transpose").with_load(mesh_xy))
        # The Green door's tables on the wedge, placed once per run and plan
        # (ppm_tau_kernel.sigma_door_tables): a τ argument, never a constant.
        g_load = None
        if q_wedge is not None and tau_kernel_factory is None and not synthesis:
            from gw.ppm_tau_kernel import sigma_door_tables
            g_load = sigma_door_tables(mesh_xy, k_unfold_plan)
        small = NamedSharding(mesh_xy, P())
        # A held SC plan's executables keep one node capacity (the session's
        # largest, never lowered), so a refit recompiles them only if it
        # raises it; a one-shot sizes it from its own windows.
        tau_capacity = max(max((len(row.window.nodes.t) for row in plan), default=0),
                           int(tau_capacity or 0))

        n_sweeps = n_tau = 0
        batch_size = int(pole_batch_size)
        sweep_started = False
        max_b = max_d = 0.0
        # The same bar as the zeta fit and the W roles (common.progress); the
        # step count is exact because window/batch membership is a pure function
        # of the pole ranges the sweep will visit.  Owner request 2026-09-03.
        total_tau = 0
        if synthesis:
            # A synthesis finishes W inside each τ node: its q/K panels are
            # storage work, never additional state-pole product windows.
            total_tau = sum(len(np.asarray(row.window.nodes.t)) for row in plan)
        else:
            for _lo in range(0, int(n_poles), batch_size):
                _batch = tuple(range(_lo, min(_lo + batch_size, int(n_poles))))
                for _row in plan:
                    if _batch_rows(_row, _batch) is not None:
                        total_tau += len(np.asarray(_row.window.nodes.t))
        progress = LoopProgress(
            max(1, total_tau), print_fn, title="Sigma tau sweep",
            item_name="tau node", max_updates=20)
        progress.start()
    for lo, Omega, B, B_odd in batches:
        if not synthesis and getattr(meta, 'mu_basis', None) is not None:
            # The pole store keeps the canonical centroid order; the run
            # computes in its packed order.  Pack every operator-shaped pole
            # field once per batch at this read seam.  GN-PPM's pole frequency
            # is per (q, mu, nu) and follows the residues; MPA's scalar poles
            # are left alone.
            _basis = meta.mu_basis
            _mu_can = int(_basis.n_canonical)
            B = _basis.pack_operator(B)
            if B_odd is not None:
                B_odd = _basis.pack_operator(B_odd)
            if Omega.ndim >= 2 and tuple(Omega.shape[-2:]) == (_mu_can, _mu_can):
                Omega = _basis.pack_operator(Omega)
        if B_odd is not None:
            max_b = max(max_b, float(jax.device_get(jnp.max(jnp.abs(B)))))
            max_d = max(
                max_d, float(jax.device_get(jnp.max(jnp.abs(B_odd)))))
            if odd_residue_off:
                B_odd = jnp.zeros_like(B_odd)
        width = 0 if synthesis else int(Omega.shape[0])
        batch = tuple(range(int(lo), int(lo) + width))
        for row in plan:
            selected = (
                (row.pole_indices, row.bounds, row.phase_real,
                 np.int32(len(row.pole_indices)))
                if synthesis else _batch_rows(row, batch))
            if selected is None:
                continue
            fence('tau.window_arguments', sync_ranks=True)
            with timing.section('tau.window_arguments'):
                pole_indices, bounds, phase_real, active_count = selected
                active_count = device_put_process_local(active_count, small)
                win = row.window
                weight = getattr(row, "band_weight", None)
                if weight is None:
                    selector = jnp.asarray(win.mask_A)
                else:
                    selector = (jnp.asarray(win.mask_A, jnp.float64)
                                * jnp.reshape(
                                    jnp.asarray(weight, jnp.float64),
                                    np.asarray(win.mask_A).shape))
                E_A_call = k_unfold_plan.parent_rows(row.E_A)
                selector = k_unfold_plan.parent_rows(
                    jnp.reshape(selector, np.shape(row.E_A)))
                # These arrays do not change between time nodes in this planned
                # window, so they are built once per window rather than per node.
                if synthesis:
                    # The synthesis reads the right endpoint's faces and its
                    # window operands (host intervals, once per window).
                    row_kernel = tau_kernel.window_kernel(row.space)
                    tau_arguments = tau_kernel.window_arguments(
                        psi_coh_xn, psi_proj_xr, E_A_call, selector,
                        jnp.asarray(win.E_ref_A), jnp.asarray(win.E_ref_B),
                        row.space, row.pole_indices, row.bounds)
                else:
                    pole_indices, bounds, phase_real = (
                        device_put_process_local(x, small)
                        for x in (pole_indices, bounds, phase_real))
                    row_kernel = tau_kernel
                    B_branch = _residue_for_space(row.space, B, B_odd)
                    if q_wedge is not None:
                        B_branch = dataclasses.replace(
                            q_pair, values=B_branch)
                    tau_arguments = (
                        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                        E_A_call, selector, B_branch, Omega,
                        pole_indices, bounds, phase_real,
                        jnp.asarray(win.E_ref_A), jnp.asarray(win.E_ref_B), g_load)
            window_options = dict(
                active_count=active_count, capacity=tau_capacity,
                omega_sign=win.omega_sign, prefactor=win.prefactor,
                e_ref_sum=win.E_ref_A + win.E_ref_B,
                antihermitian=(win.project_code == 1),
                omega_indices=row.omega_idx, omega_values=row.omega_abs,
                overlap=getattr(tau_kernel, "overlap", False))
            if not sweep_started:
                fence('tau.initial_compile_and_probe', sync_ranks=True)
                with timing.section('tau.initial_compile_and_probe'):
                    compiled = accumulator.integrate_window(
                        row_kernel, tau_arguments, win.nodes.t,
                        win.nodes.alpha, n_active=len(win.nodes.t),
                        compile_only=True, **window_options)
                    if window_options["overlap"] and not tau_kernel.fits(compiled):
                        # Two nodes in flight do not fit beside the live
                        # stages: one node per trip, the default schedule (the
                        # stage table's window price then names no pairing).
                        tau_kernel.overlap = window_options["overlap"] = False
                        compiled = accumulator.integrate_window(
                            row_kernel, tau_arguments, win.nodes.t,
                            win.nodes.alpha, n_active=len(win.nodes.t),
                            compile_only=True, **window_options)
                    if synthesis:
                        tau_kernel.admit(compiled, tau_arguments)
                    print_fn(
                        "  MPA Sigma sweep begin: one executable per window "
                        "prewarmed")
                    sweep_started = True
            fence('tau.window_setup', sync_ranks=True)
            with timing.section('tau.window_setup'):
                t_nodes = np.asarray(
                    jax.device_get(win.nodes.t), np.complex128)
                alpha_nodes = np.asarray(
                    jax.device_get(win.nodes.alpha), np.complex128)
            # One executable per window: the node loop runs on device.
            total = accumulator.integrate_window(
                row_kernel, tau_arguments, t_nodes, alpha_nodes,
                n_active=len(t_nodes), **window_options)
            for _ in t_nodes:
                progress.step(wait=total)
            n_tau += len(t_nodes)
            n_sweeps += 1
        del B, B_odd, Omega
        gc.collect()
    progress.finish()

    fence('tau.finalize', sync_ranks=True)
    with timing.section('tau.finalize'):
        sigma = accumulator.finalize()
        if bracketed:
            # The counts stay on the wedge; each is unfolded when it is read.
            matrices, diagonals = sigma
            sym = k_unfold_plan.sym
            full = _wedge_is_full_bz(sym)
            sigma = BandCountCube(
                matrices=list(matrices), diagonals=list(diagonals),
                unfold=((lambda value: value) if full else
                        _unfold_sigma_cube_fn(sym, 1, output_sharding)),
                unfold_rows=((lambda slots: slots) if full else
                             _band_count_rows_fn(sym, mesh_xy)),
                nk=int(sym.nk_tot), fresh=not full, mesh_xy=mesh_xy)
            if band_counts is None:
                band_counts = tuple(
                    int(s.nb_sigma_sum) if hi is None else int(hi)
                    for _lo, hi in brackets)
            else:
                band_counts = tuple(int(count) for count in band_counts)
            if len(band_counts) != len(brackets):
                raise ValueError(
                    "MPA Sigma band_counts must align with band brackets")
        else:
            sigma = _unfold_sigma_cube(
                sigma, k_unfold_plan.sym, k_axis=1, sharding=output_sharding)
        # Format read by the sandbox parser (tools/parse_lorrax_sigma_run.py);
        # every node now runs inside its window's executable, so none is
        # left undispatched.
        print_fn(
            f"  MPA Sigma: {n_tau} tau dispatches in {n_sweeps} sweeps "
            f"({n_poles} poles, batches of {batch_size}); "
            f"0 undispatched logical tau; "
            f"panes and product windows used one shared tau kernel")
        ratio = None
        if max_b or max_d:
            ratio = max_d / max_b if max_b else np.inf
            state = "D=0 twin" if odd_residue_off else "enabled"
            print_fn(
                f"  MPA odd Sigma: measured-broken-TR ordered residues; {state}; "
                f"max|D|/max|B|={ratio:.12e}")
        return SigmaOmegaResult(
            omega_ry=omega,
            omega_ev=np.asarray(omega * RYD_TO_EV, np.float64),
            sigma_c_kij=sigma,
            band_axis=sigma_axis,
            band_counts=(() if band_counts is None else tuple(band_counts)),
            odd_even_residue_ratio=ratio)


def _unfold_sigma_cube(sigma, sym, *, k_axis, sharding):
    """FILE wedge -> full BZ on the k axis of the Sigma(omega) cube, band axes kept sharded.

    Transpose is complex-linear, so transport follows the complete omega fold
    without conjugating quadrature coefficients or storing tau tiles.  The
    output sharding is PINNED: unpinned, XLA replicated the full-BZ cube on
    every rank (CrI3 16x16, 3 x 65 x 256 x 184^2 c128 = 27 GB/rank plus a
    57 GB temp at P64 -- the map-0 OOM; KNOWN_LORRAX_ISSUES 2026-09-23).
    """
    return _unfold_sigma_cube_fn(sym, int(k_axis), sharding)(sigma)


# The two finalize executables are built once per (symmetry table, layout),
# not once per call: a fresh ``jax.jit(lambda ...)`` is a new cache entry, so
# every SC map re-traced and re-compiled both (QUALITY_PATTERNS §5,
# compiled-object lifetime).  Same jaxpr, so the values are bit-identical.
@lru_cache(maxsize=8)
def _unfold_sigma_cube_fn(sym, k_axis, sharding):
    from symmetry_maps import unfold_file_wedge_band_operator
    return jax.jit(lambda value: jnp.moveaxis(
        unfold_file_wedge_band_operator(
            sym, jnp.moveaxis(value, k_axis, 0),
            trs_rule="transpose"), 0, k_axis),
        out_shardings=sharding)


def _wedge_is_full_bz(sym) -> bool:
    """The FILE wedge is the full BZ in its own order (a WFN without symmetry)."""
    from symmetry_maps import star_tables_of
    irr, sidx, n_spatial = star_tables_of(sym)
    return (int(sym.nk_red) == irr.size
            and np.array_equal(irr, np.arange(irr.size))
            and not np.any(np.asarray(sidx) >= int(n_spatial)))


@lru_cache(maxsize=8)
def _band_count_reduce(mesh_xy, n_brackets):
    """``sigma(t)`` on the disjoint brackets -> what the band counts keep.

    ``(matrices, diagonals)``: the cumulative matrix at the first and at the
    last count, and the band diagonal of each count between them.  A diagonal
    stays on the band tiles that own it, ``(k, nb, p_y)`` at
    ``P(None, 'x', 'y')`` with one nonzero slot per band, so the fold is
    rank-local.
    """
    from gw.ppm_sigma import band_diagonal_slots
    diag_slots = band_diagonal_slots(mesh_xy, 3) if n_brackets > 2 else None

    def reduce(sigma):
        running = sigma[0]
        matrices, diagonals = [running], []
        for b in range(1, n_brackets):
            running = running + sigma[b]
            if b < n_brackets - 1:
                diagonals.append(diag_slots(running))
        if n_brackets > 1:
            matrices.append(running)
        return tuple(matrices), tuple(diagonals)

    return reduce


@lru_cache(maxsize=8)
def _band_count_rows_fn(sym, mesh_xy):
    """Diagonal slots on the FILE wedge -> full BZ (a row gather; the
    antiunitary transpose leaves a diagonal unchanged)."""
    from symmetry_maps import star_tables_of
    rows = np.asarray(star_tables_of(sym)[0], dtype=np.int32)
    return jax.jit(lambda slots: slots[:, jnp.asarray(rows)],
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def _attach_ordered_odd_sigma(total, even):
    """Attach the exact ordered-residue MPA contribution to ``total``.

    Both inputs must be executions of the same fitted poles, planner and tau
    grid; only the second execution has ``D=0``.  Keeping the subtraction at
    this seam makes ``sigC_odd`` a diagnostic of the production contraction,
    not a separately approximated formula.
    """
    if not np.array_equal(total.omega_ry, even.omega_ry):
        raise ValueError(
            "GATE mpa_odd_sigma_reference: total and D=0 MPA Sigma used "
            "different omega grids")
    if tuple(total.sigma_c_kij.shape) != tuple(even.sigma_c_kij.shape):
        raise ValueError(
            "GATE mpa_odd_sigma_reference: total and D=0 MPA Sigma shapes "
            f"differ: {total.sigma_c_kij.shape} versus "
            f"{even.sigma_c_kij.shape}")
    return replace(
        total,
        sigma_c_odd_kij=total.sigma_c_kij - even.sigma_c_kij)


@lru_cache(maxsize=8)
def _logical_mu_fn(sharding, n_mu):
    """Zero both trailing (mu, nu) axes past the logical extent, in place of layout."""
    def keep(x):
        live = jnp.arange(x.shape[-1]) < n_mu
        return jnp.where(live[:, None] & live[None, :], x, jnp.zeros((), x.dtype))
    return jax.jit(keep, out_shardings=sharding)


class MemoryPoleSource:
    """Fitted poles handed to the Sigma executor in memory: PoleReader's batch interface, no store.

    Fields are ``(n_p, n_q, n_mu_pad, n_mu_pad)`` in Ry on ``P(None, None,
    'x', 'y')``, with the full-BZ q axis.  Entries past ``n_mu_logical`` are
    zeroed once here; that is what the store round trip gave, since the
    store keeps only the logical extent and its reads zero-fill the padding.
    Batches are device slices of the resident fields.  Nothing is gathered,
    copied to host, or written.

    ``axis``: a ``runtime.padding.PaddedAxis`` naming the operator extent
    when it is not the centroid axis -- the plane-wave response sphere
    (``gw.plane_wave_screening.SphereScreening.axis``).
    ``q_wedge``: the carrier the executor wraps each residue batch in
    (``dataclasses.replace(q_wedge, values=None, load=None,
    trs_rule="pair_transpose").with_load(mesh)``, then ``values=B``): the ISDF
    ``QirrOperator``, or the plane-wave ``SphereResidues`` whose load is the
    pair convolution's tables (its τ body reads W on the q-IBZ through them).
    """

    def __init__(self, Omega_p, B_p, B_odd_p=None, *, n_mu_logical=None, mesh_xy,
                 provenance, q_wedge=None, axis=None):
        from runtime.padding import padded_mu_extent
        shape = tuple(int(n) for n in Omega_p.shape)
        n_mu = int(axis.logical if n_mu_logical is None else n_mu_logical)
        if len(shape) != 4 or tuple(B_p.shape) != shape or (
                B_odd_p is not None and tuple(B_odd_p.shape) != shape):
            raise ValueError(
                "MemoryPoleSource: Omega/B/B_odd must share one "
                f"(n_p, n_q, mu, nu) shape; got {shape}, {tuple(B_p.shape)}, "
                f"{None if B_odd_p is None else tuple(B_odd_p.shape)}")
        n_pad = int(padded_mu_extent(n_mu, mesh_xy) if axis is None else axis.carrier)
        if shape[2:] != (n_pad, n_pad):
            raise ValueError(
                f"MemoryPoleSource: mu extent {shape[2:]} is not the store's "
                f"read extent ({n_pad}, {n_pad}) for n_mu = {n_mu}")
        keep = _logical_mu_fn(
            NamedSharding(mesh_xy, P(None, None, "x", "y")), n_mu)
        self.Omega, self.B = keep(Omega_p), keep(B_p)
        self.B_odd = None if B_odd_p is None else keep(B_odd_p)
        from file_io.mpa_store import refuse_bad_pole_fields
        refuse_bad_pole_fields(
            self.Omega, self.B, self.B_odd, where="MemoryPoleSource")
        self.n_poles = shape[0]
        # The q wedge the fields live on (the GN fit's), or None: full zone.
        self.q_wedge = q_wedge
        self.ledger = {
            "n_p": shape[0], "n_q": shape[1], "n_mu": n_mu,
            "ordered_residues": B_odd_p is not None,
            "q_storage": "full" if q_wedge is None and axis is None else "ibz",
            "energy_unit": "Ry",
            "provenance": dict(provenance),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, pole_slice=None, *, unfold=False, return_sharded=False,
             to_unit=None, include_odd=False):
        from file_io.mpa_store import _pole_range
        if not return_sharded or to_unit not in (None, "Ry"):
            raise ValueError(
                "MemoryPoleSource serves sharded Ry batches only; got "
                f"return_sharded={return_sharded}, to_unit={to_unit!r}")
        if unfold and self.q_wedge is not None:
            raise ValueError(
                "MemoryPoleSource: the fields are on the q wedge; the Sigma "
                "executor reads them there (unfold=False) and unfolds on load")
        lo, hi = _pole_range(self.ledger, pole_slice, "MemoryPoleSource.read")
        take = ((lambda x: x) if (lo, hi) == (0, self.n_poles)
                else (lambda x: x[lo:hi]))
        B_odd = (take(self.B_odd)
                 if include_odd and self.B_odd is not None else None)
        return take(self.Omega), take(self.B), B_odd


def integrate_sigma_store(
    wfns,
    fit_src,
    n_poles,
    plan,
    omega_grid_ry,
    meta,
    mesh_xy,
    *,
    pole_batch_size=4,
    brackets=None,
    band_counts=None,
    odd_residue_off=False,
    tau_capacity=None,
    tau_kernel_factory=None,
    print_fn=print,
):
    """Read, unfold, consume, and release one pole range at a time.

    ``fit_src`` is a store path, or a live
    :class:`~file_io.restart_bundle.PoleReader` whose collective handle the
    caller owns — which is what
    :func:`compute_sigma_c_mpa_omega_grid` passes, so the census walk and
    this executor walk share ONE open handle for the whole iteration
    instead of opening the store once per pole batch (audit A1).  Given a
    path, this function owns a reader for the length of its own walk.

    ``brackets`` optionally partitions the intermediate-state band sum into
    disjoint slices.  The spatial kernel then returns a leading bracket axis;
    this executor folds the cumulative first and last counts and the
    diagonals between (``_band_count_reduce``) on the FILE wedge and returns
    a ``BandCountCube``, which unfolds one count at a time.  ``None``
    preserves the ordinary MPA rank-4 result.

    ``tau_kernel_factory`` replaces the resident pole route's τ body (the
    plane-wave path's ``get_shared_sigma_tau_kernel(_sigma_kij=...)``); see
    :func:`_integrate_sigma_batches`.
    """
    batch_size = _bounded_pole_batch_size(pole_batch_size)

    q_wedge = getattr(fit_src, "q_wedge", None)

    def batches(reader):
        for lo in range(0, int(n_poles), batch_size):
            hi = min(lo + batch_size, int(n_poles))
            Omega, B, B_odd = reader.read(
                slice(lo, hi), unfold=q_wedge is None, return_sharded=True,
                to_unit="Ry", include_odd=True)
            yield lo, Omega, B, B_odd
            del Omega, B, B_odd
            gc.collect()

    def run(reader):
        return _integrate_sigma_batches(
            wfns, batches(reader), int(n_poles), plan, omega_grid_ry, meta,
            mesh_xy, pole_batch_size=batch_size, brackets=brackets,
            band_counts=band_counts, odd_residue_off=odd_residue_off,
            q_wedge=q_wedge, tau_capacity=tau_capacity,
            tau_kernel_factory=tau_kernel_factory, print_fn=print_fn)

    if isinstance(fit_src, (PoleReader, MemoryPoleSource)):
        return run(fit_src)
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        return run(reader)


def _branches(wfns, omega, efermi_ry, occupation_state=None):
    """The four causal branches, with occupation and energy kept separate.

    Band axis is ``slices.sigma_sum`` -- the Sigma band count, not the
    chi one -- for the same reason as the psi slices above: these index
    the SAME band axis the causal branches sum over.  That choice is
    orthogonal to the occupation one below: the slice says WHICH bands
    are summed, the weights say with what amplitude.

    ``occupation_state=None`` is the incumbent insulating semantics,
    bit-exact: bool occ>0.5 masks, distances signed against ``efermi_ry``.
    With a state (duck-typed: ``.f_kn``, ``.mu_ry``), the branches carry the
    fractional supports and weights: the val branch sums every band whose
    weight f is in its float64 support at weight f, the cond branch every
    band whose weight 1−f is at weight 1−f.  Nothing is clipped; the
    Fermi-Dirac weights metals use lie in [0, 1], and an MP overshoot
    (f<0 or f>1) would ride through unchanged
    (docs/theory/finite-occupation-screening.md).

    ``branches_for_omega_grid`` applies the one support predicate
    (``gw.efermi.band_in_occupation_window``).  Applying it here rather
    than only in the planner keeps ONE support: ``sigma_windows._a_space``
    re-applies the same predicate to the same weights, so the two agree by
    construction instead of by review.
    """
    if occupation_state is None:
        energy = wfns.enk[:, wfns.slices.sigma_sum] - float(efermi_ry)
        occupied = wfns.occ[:, wfns.slices.sigma_sum] > 0.5
        # Do not clip these distances at zero.  In a small-gap or inverted
        # system an unoccupied state may sit below E_F (or an occupied state
        # above it); occupation still chooses the band sum.  A cell whose
        # rectangle then crosses zero is rerouted through the crossing core
        # by the planner's excursion-deepened edge (sigma_windows._geometry).
        return branches_for_omega_grid(
            omega, E_cond=energy, H_val=-energy,
            cond_mask=~occupied, val_mask=occupied)
    mu = float(occupation_state.mu_ry)
    if abs(float(efermi_ry) - mu) > 1.0e-12:
        raise ValueError(
            "MPA Sigma got efermi_ry inconsistent with its occupation "
            f"state: efermi_ry={float(efermi_ry):.12g} Ry vs "
            f"occupation_state.mu_ry={mu:.12g} Ry.  One chemical potential "
            "per iteration — pass the state's own mu.")
    f = jnp.reshape(jnp.asarray(occupation_state.f_kn),
                    wfns.enk.shape)[:, wfns.slices.sigma_sum]
    energy = wfns.enk[:, wfns.slices.sigma_sum] - mu
    return branches_for_omega_grid(
        omega, E_cond=energy, H_val=-energy,
        cond_mask=(f != 1.0), val_mask=(f != 0.0),
        cond_weight=1.0 - f, val_weight=f)


def compute_sigma_c_mpa_omega_grid(
    wfns,
    fit_src,
    meta,
    mesh_xy,
    *,
    omega_grid_ry,
    efermi_ry,
    regularization_width_ry,
    edge_factor=1.5,
    quadrature_eps,
    omega_grid_step_ry,
    pole_batch_size=4,
    fit_identity=None,
    fit_digest=None,
    expected_screening_diagrams=None,
    occupation_state=None,
    sigma_branches=None,
    band_brackets=None,
    band_counts=None,
    fixed_quadrature_session=None,
    material_class=None,
    sigma_w_model="mpa",
    analytic_line=False,
    sector_context=None,
    odd_reference=True,
    tau_kernel_factory=None,
    omega_eta_ry=None,
    omega_group=None,
    group_fixed=None,
    linalg=None,
    print_fn=print,
):
    """Read a fitted MPA store, derive its windows, and compute Sigma_c.

    ``omega_eta_ry``/``omega_group``/``group_fixed``: the SC coarse windows'
    per-frequency broadening and window labels (``sigma_box_plan.plan_sigma_windows``).
    ``linalg``: the deck's resolved dense layout, which places a shared-pole
    model's W(τ) synthesis (:func:`_shared_pole_memory_schedule`).

    ``occupation_state`` (duck-typed ``gw.efermi.OccupationState``): None is
    the incumbent insulating semantics, bit-exact.  With a state, the causal
    branches carry exact fractional supports and (f, 1−f) weights, and
    ``efermi_ry`` must equal ``occupation_state.mu_ry``.

    The branch build and both planner entry points read the one support
    predicate (``gw.efermi.band_in_occupation_window``), which keeps the
    branch supports, the pole census and the window build on one support.

    Pole tensors are read collectively in their native sharding.  The box
    planner retains only exact live extrema from each configured pole batch;
    the pane control consumes the same bounded census.  The spatial executor
    then rereads and releases the pole ranges through one shared tau kernel.
    ``sigma_branches`` is an optional already-resolved set of causal branches;
    it lets another pole model retain its established occupation/Fermi policy
    while sharing this planner and executor exactly.  ``band_brackets`` and
    ``band_counts`` similarly carry the optional disjoint band-convergence
    partition.  They change neither pole interpretation nor window planning.
    ``analytic_line`` is the PPM request for its real-pole crossing windows;
    the planner validates the pole and denominator geometry before using it.
    ``tau_kernel_factory`` (the pole route only): the τ body another spatial
    basis supplies, ``factory(w_synthesis, sigma_axis) -> tau_kernel`` with
    the resident route's signature -- the plane-wave path
    (``gw.plane_wave_pipeline``) passes ``get_shared_sigma_tau_kernel`` over
    its own ``_sigma_kij``.  The planner, the windows and the accumulator
    are unchanged.
    """
    if sigma_w_model not in ("mpa", "shared_pole"):
        raise ValueError(f"sigma_w_model must be mpa or shared_pole; got {sigma_w_model!r}")
    shared_pole = sigma_w_model == "shared_pole"
    if shared_pole and tau_kernel_factory is not None:
        raise ValueError(
            "compute_sigma_c_mpa_omega_grid: tau_kernel_factory serves the pole "
            "route; a shared-pole model passes its sector kernel through sector_context")
    fixed_pole_support_ry = None
    if shared_pole:
        from file_io.shared_pole_store import open_shared_pole_model, validate_shared_pole_model
        with timing.section("sigma.model_validate"):
            ledger = validate_shared_pole_model(
                fit_src, expected_identity=fit_identity, mesh_xy=mesh_xy,
                capacity=meta.shared_pole_capacity)
        if fit_digest is not None and ledger["digest"] != fit_digest:
            raise ValueError("GATE shared_pole_identity: screening handle digest differs from model")
        recipe = meta.shared_pole_recipe
        if not np.isclose(regularization_width_ry * RYD_TO_EV,
                          recipe["eta_ev"], rtol=0, atol=1e-12):
            raise ValueError("GATE shared_pole_eta: Sigma and current recipe eta differ")
        if fixed_quadrature_session is not None:
            fixed_pole_support_ry = (
                recipe.get("sector_pole_treatment") or {}).get("ceiling_ry")
        n_poles = int(ledger["n_q_irr"])
        ordered_residues = False
        with timing.section("sigma.capacity"):
            schedule = (_shared_pole_memory_schedule(meta, ledger, mesh_xy=mesh_xy, layout=wfns.layout,
                                                     linalg=linalg)
                        if sector_context is None else sector_context["schedule"](ledger))
        print_fn(f"  shared-pole Sigma capacity: {schedule}")
    elif isinstance(fit_src, MemoryPoleSource):
        # In-memory GN/HL poles: no store, so no file identity to check.
        ledger = fit_src.ledger
        got = ledger["provenance"].get("screening_diagrams")
        want = (None if expected_screening_diagrams is None else str(getattr(
            expected_screening_diagrams, "value", expected_screening_diagrams)))
        if fit_identity is not None or (want is not None and got != want):
            raise ValueError(
                "GATE memory_pole_source_identity: in-memory poles carry "
                f"screening_diagrams={got!r} (want {want!r}) and no file "
                f"identity (fit_identity={fit_identity!r} requested)")
        n_poles = int(ledger["n_p"])
        ordered_residues = bool(ledger["ordered_residues"])
    else:
        ledger = validate_fit_store(
            fit_src, expected_identity=fit_identity,
            expected_screening_diagrams=expected_screening_diagrams)
        n_poles = int(ledger["n_p"])
        ordered_residues = bool(ledger["ordered_residues"])
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    with timing.section("sigma.branches"):
        branches = (_branches(
            wfns, omega_grid_ry, efermi_ry,
            occupation_state=occupation_state)
            if sigma_branches is None else tuple(sigma_branches))
    # ONE collective handle for the census walk, the planner, and the
    # executor walk — the whole Σ stage of this iteration.  The reader
    # does its h5py reads (ledger, unfold tables) before that handle
    # exists and none after, so no serial-h5py open on this store
    # overlaps or interleaves with the FFI one anywhere inside a Σ stage
    # (audit A1; hdf5_owner enforces it).  The context manager is the
    # release path: a refusal from the planner or the executor must still
    # close the handle on every rank.
    with (open_shared_pole_model(fit_src, mesh_xy=mesh_xy) if shared_pole else
          fit_src if isinstance(fit_src, MemoryPoleSource) else
          open_pole_reader(fit_src, mesh_xy=mesh_xy)) as reader:
        # One bounded extrema census serves both routes.  In particular, the
        # production route does not read residues into a host histogram and
        # never constructs a sampled state-pole lattice.
        summaries = []
        certificate = None
        # A route without a pole census reuses rules across the run's plans.
        rule_scope = RUN_SCOPE
        if shared_pole:
            from file_io.shared_pole_store import read_shared_pole_census
            with timing.section("sigma.census"):
                poles_device, counts_device = read_shared_pole_census(
                    reader, header=ledger, capacity=meta.shared_pole_capacity)
                poles2, counts = map(np.asarray, jax.device_get((poles_device, counts_device)))
                del poles_device, counts_device
                # A sector call scopes its rules by the map's union census
                # (compute_sector_sigma), so sectors reuse each other's fits.
                scope = (sector_context or {}).get("rule_census")
                rule_scope = sigma_rule_scope(
                    ledger["identity"],
                    *((poles2, counts) if scope is None else scope),
                    eta=regularization_width_ry, eps=quadrature_eps)
                frequencies = shared_pole_frequencies(poles2, counts)
                summaries = summarize_shared_poles(
                    poles2, counts, branches,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor)
                if scope is not None:
                    # The certificate boxes come from the same union, so the
                    # map's sector calls request one box set: the first fits
                    # it and the others hit (Fe 4^3: 3.3 s per later sector).
                    union_rows, union_counts = scope
                    union2 = np.ones((len(union_rows), max(1, max(map(len, union_rows)))))
                    for q, row in enumerate(union_rows):
                        union2[q, :len(row)] = np.sort(row)
                    certificate = summarize_shared_poles(
                        union2, np.asarray(union_counts, np.int64), branches,
                        regularization_width_ry=regularization_width_ry,
                        edge_factor=edge_factor)
        for lo in (() if shared_pole else range(0, n_poles, int(pole_batch_size))):
            hi = min(lo + int(pole_batch_size), n_poles)
            Omega, B, B_odd = reader.read(
                slice(lo, hi), unfold=getattr(reader, "q_wedge", None) is None,
                return_sharded=True, to_unit="Ry", include_odd=True)
            _refuse_nonfinite_pole_slab(lo, Omega, B, B_odd)
            summaries.extend(summarize_sigma_poles(
                Omega, _geometry_residue(B, B_odd), branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor, pole_offset=lo))
            del Omega, B, B_odd
            gc.collect()
        # Rule fitting is its own timing row: on the Si b80/c504 deck the
        # cold fits took ~180 s of a 194 s "Sigma" stage while the tau
        # sweep took 6 s (2026-09-03, runs/DEV/122), and the table
        # could not tell them apart.
        with timing.section("sigma.rule_plan"):
            plan, geometry = plan_sigma_windows(
                summaries, branches, omega_grid_ry,
                regularization_width_ry,
                eps=quadrature_eps,
                scope=rule_scope,
                print_fn=print_fn, edge_factor=edge_factor,
                fixed_rule_session=fixed_quadrature_session,
                analytic_line=bool(analytic_line),
                material_class=material_class,
                fixed_pole_support_ry=fixed_pole_support_ry,
                certificate_pole_summaries=certificate,
                occupation_reach_ry=occupation_floor_reach_ry(occupation_state),
                omega_eta_ry=omega_eta_ry, omega_group=omega_group,
                group_fixed=group_fixed)
        quadrature_log.record_sigma_plan(geometry)
        print_fn(
            f"  MPA windows [box]: "
            f"eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
            f"eps={geometry['eps']:.3g}, "
            f"certificate={geometry['rule_eps']:.3g}, "
            f"{geometry['n_windows']} logical windows, "
            f"{geometry['window_tau_pairs']} (window,tau) pairs, "
            f"{geometry['distinct_tau_count']} branch-distinct tau, "
            f"rule_scope={(geometry['rule_scope'] or 'none')[:24]}, "
            f"rules_built={geometry['rules_built']}")
        if geometry["sc_fixed_quadrature"]:
            reasons = geometry.get("sc_fixed_recompute_reasons") or {}
            print_fn(
                "  SC fixed quadrature: "
                f"iteration={geometry['sc_fixed_iteration']}, "
                f"rules={geometry['sc_rule_mode']}, "
                f"initialized={geometry['sc_fixed_initialized']}, "
                f"frozen={geometry['sc_rule_mode'] == 'frozen' and not reasons and geometry['sc_fixed_rebuilds_this_iteration'] == 0}, "
                f"escaped={geometry['sc_fixed_escaped_windows']}/"
                f"{geometry['n_windows']}, "
                f"rebuilds_this_iteration="
                f"{geometry['sc_fixed_rebuilds_this_iteration']}, "
                f"rebuilds_total="
                f"{geometry['sc_fixed_total_rebuild_count']}, "
                f"material_class={geometry.get('sc_fixed_material_class')}, "
                f"pair_cost={geometry['window_tau_pairs']}, "
                f"initial_pair_cost="
                f"{geometry['sc_fixed_initial_window_tau_pairs']}, "
                f"plan={geometry['sc_plan_event']}, "
                f"escape_maps_total={geometry['sc_fixed_escape_maps_total']}, "
                f"state_pad=outer max({geometry['sc_state_edge_padding_ev']:.2f} eV, "
                f"{100.0 * geometry['sc_state_edge_padding_fraction']:.0f}%|E-mu|), "
                f"crossing inner {geometry['sc_inner_state_padding_eta']:g} eta"
                + ("" if geometry['sc_occupation_reach_ry'] is None else
                   f" clipped at -{geometry['sc_occupation_reach_ry'] * RYD_TO_EV:.4f} eV")
                + ", "
                f"pole_pad="
                f"{100.0 * geometry['sc_pole_extent_padding_fraction']:.1f}% "
                f"(far x{geometry['sc_far_pole_factor']:g}), "
                f"tau_capacity={geometry['sc_tau_capacity']}")
            for name, reason in sorted(reasons.items()):
                print_fn(
                    f"    SC fixed quadrature recompute: {name!r} "
                    f"({reason})")
        for branch in geometry["branches"]:
            for window in branch["windows"]:
                prefix = (
                    "    SC fixed window: "
                    if window["sc_fixed_rule"] else "    ")
                box = tuple(window["box_ry"])
                padded = (
                    "" if not window["sc_fixed_rule"] else
                    f"padded_box="
                    f"{tuple(value * RYD_TO_EV for value in window['sc_fixed_padded_box_ry'])} "
                    "eV, ")
                if window["sc_fixed_rule"]:
                    box = tuple(value * RYD_TO_EV for value in box)
                print_fn(
                    f"{prefix}{window['name']}: "
                    f"n_tau={window['node_count']}, "
                    f"nodes={window['node_digest']}, "
                    f"rule={window['rule_source']}, "
                    f"box={box} "
                    f"{'eV' if window['sc_fixed_rule'] else 'Ry'}, "
                    f"{padded}"
                    f"sup={window['sup_error']:.6g}/"
                    f"{window['eps']:.6g} ({window['criterion']}), "
                    f"kappa_max={window['kappa_max']:.6g}, "
                    f"noise={window['runtime_noise_bound']:.6g}/"
                    f"{window['runtime_noise_budget']:.6g}")
        with timing.section("sigma.tau_sweep"):
            if shared_pole:
                _band_fence('tau.synthesis_setup', sync_ranks=True)
                with timing.section('tau.synthesis_setup'):
                    synthesis = (_shared_pole_w_synthesis(
                        reader, meta, ledger, frequencies, schedule, mesh_xy=mesh_xy,
                        layout=schedule.get("factor_layout", wfns.layout))
                        if sector_context is None else sector_context["synthesis"](
                            reader, ledger, frequencies, schedule))
                # A sector's caller owns its synthesis lifetime
                # (compute_sector_sigma); the scalar model's ends here.
                owned = sector_context is None
                try:
                    total = _integrate_sigma_batches(
                        wfns, ((0, None, None, None),), n_poles, plan,
                        omega_grid_ry, meta, mesh_xy, pole_batch_size=n_poles,
                        brackets=band_brackets, band_counts=band_counts,
                        w_synthesis=synthesis,
                        tau_kernel_factory=(None if owned else
                                            sector_context["tau_kernel"]),
                        tau_capacity=geometry.get("sc_tau_capacity"), print_fn=print_fn)
                except BaseException:
                    if owned:
                        synthesis.close()
                    raise
                if owned:
                    synthesis.close(total.sigma_c_kij)
                del synthesis
            else:
                total = integrate_sigma_store(
                    wfns, reader, n_poles, plan, omega_grid_ry, meta, mesh_xy,
                    pole_batch_size=pole_batch_size, brackets=band_brackets,
                    band_counts=band_counts,
                    tau_capacity=geometry.get("sc_tau_capacity"),
                    tau_kernel_factory=tau_kernel_factory,
                    print_fn=print_fn)
        # odd_reference=False: the caller builds its own D=0 reference (the GN
        # arm does, in ppm_pipeline), so a second twin here would be a whole
        # extra sweep whose sigma_c_odd_kij nobody reads.
        if not ordered_residues or not odd_reference:
            return total
        # Exact observability twin, shared in algebra with the GN arm in
        # ppm_pipeline: Sigma is linear in the fitted residues, so the same
        # plan and compiled contraction with D=0 isolates the ordered term.
        # A debug-off arm deliberately emits a zero twin, letting the public
        # sigC_odd column check its own A/B.
        even = integrate_sigma_store(
            wfns, reader, n_poles, plan, omega_grid_ry, meta, mesh_xy,
            pole_batch_size=pole_batch_size, brackets=band_brackets,
            band_counts=band_counts, odd_residue_off=True,
            tau_kernel_factory=tau_kernel_factory,
            print_fn=lambda *args, **kwargs: None)
        return _attach_ordered_odd_sigma(total, even)


def assert_head_body_occupation_match(
        head_attrs, occupation_state, *, compatible_occ_hashes=()):
    """Refuse when the head fit and the body Sigma disagree about occupations.

    ``head_attrs`` is the stamp dict a head-fit reader returns.  One
    occupation state per iteration is the rule (ARCHITECTURE W2.d/W3); the
    head's stamped ``occ_hash``/``mu_ry`` must equal the body's.  A legacy
    hash may match only when the caller reproduced it from the live table by
    exact-zero padding and supplies it in ``compatible_occ_hashes``.  A metal
    run with an UNSTAMPED head fit refuses too — an unverifiable stamp is not
    a pass.  Insulating runs (state None) skip the check.
    """
    if occupation_state is None:
        return
    stamped_hash = head_attrs.get("occ_hash")
    stamped_mu = head_attrs.get("mu_ry")
    if stamped_hash is None or stamped_mu is None:
        raise ValueError(
            "metallic MPA Sigma requires an occupation-stamped head fit "
            "(occ_hash + mu_ry attrs); this store has "
            f"occ_hash={stamped_hash!r}, mu_ry={stamped_mu!r}.  Refit the "
            "head with the current iteration's occupation state.")
    hash_exact = str(stamped_hash) == str(occupation_state.occ_hash)
    hash_compatible = str(stamped_hash) in {
        str(value) for value in compatible_occ_hashes}
    if (not (hash_exact or hash_compatible)
            or abs(float(stamped_mu) - float(occupation_state.mu_ry))
            > 1.0e-12):
        raise ValueError(
            "head fit and Sigma body carry different occupation states: "
            f"head (occ_hash={stamped_hash}, mu={float(stamped_mu):.12g}) "
            f"vs body (occ_hash={occupation_state.occ_hash}, "
            f"mu={float(occupation_state.mu_ry):.12g}).  One state per "
            "iteration; rebuild the stale artifact.")
    return "exact" if hash_exact else "legacy_zero_pad"


__all__ = [
    "assert_head_body_occupation_match",
    "compute_sigma_c_mpa_omega_grid",
    "integrate_sigma_store",
    "_attach_ordered_odd_sigma",
]
