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
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, _residue_for_space, sigma_band_axis
from gw.ppm_tau_kernel import (_get_sigma_kij_kernel,
                               get_shared_sigma_tau_kernel)
from gw.ppm_windows import branches_for_omega_grid
from gw import quadrature_log
from gw.sigma_box_plan import plan_sigma_windows, sigma_rule_request_cache
from gw.wavefunction_bundle import (
    parent_sigma_operands, sigma_face_kernel_kwargs)
from runtime.padding import combined_divisor, pad_to_axis, padded_axis

from .sigma_windows import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                            summarize_sigma_poles,
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


def synthesize_shared_pole_parents(
    b_X, b_Y, poles2, intervals, E_ref_B, t_node, *, mesh_xy, gemm, layout="face",
):
    """Synthesize both raw-parent orientations through the configured G service.

    Parameters
    ----------
    b_X, b_Y : jax.Array
        Complex128 physical factors ``[parent,mu,spin,column]`` with
        face layouts, or replicated columns for the configured axis layout.
        The component axis is 1 for charge and 3 for current endpoints.
    poles2 : jax.Array
        Replicated float64 ``[parent,column]`` squared frequencies in Ry².
    intervals : jax.Array
        Replicated integer ``[parent,2]`` active half-open column ranges,
        prepared from the sorted census for this window/panel.
    E_ref_B, t_node : scalar
        Window reference in Ry and signed complex time in Ry^-1.
    mesh_xy : jax.sharding.Mesh
        Existing mesh with named x/y axes.
    gemm : distrib_la.GemmPlan
        Eagerly planned N,N contraction for this parent and padded pole panel.

    Returns
    -------
    Wplus, Wtranspose : jax.Array
        Parent ``P(None,'x','y')`` tiles ``(b_X d) b_Y†`` and
        ``(conj(b_X) d) b_Yᵀ`` at the SAME τ (DESIGN §3.4). Never conjugate
        Wplus to obtain its antiunitary partner: d must retain its phase.
    """
    if b_X.ndim != 4 or b_Y.ndim != 4:
        raise ValueError("shared-pole faces require [parent,mu,spin,column]")
    if b_X.shape[2] not in (1, 3) or b_Y.shape[2] not in (1, 3):
        raise ValueError("GATE shared_pole_components: expected charge=1 or current=3")
    weights = _shared_pole_weights(poles2, intervals, E_ref_B, t_node)
    plus = _shared_pole_contract(b_X, b_Y, weights, gemm=gemm, layout=layout)
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


def _shared_pole_contract(b_X, b_Y, weights, *, gemm, layout="face"):
    """W(τ) = b d b† through G's configured face or axis contraction.

    Factors [q,mu,spin,K] use G's face placement under ``layout='face'``;
    the axis layout keeps K replicated and divides each centroid
    endpoint over its assigned mesh axis.  The result always uses both axes.
    The causal weight [q,K] is separate and replicated. The permutations
    below are local axis views, giving exactly psi_mun and psi_nmu layouts.
    """
    from gw.greens_function_kernel import build_G

    # Components are operator-port labels, not Green-function spinors.
    # Merge them with their own centroid axis before entering build_G;
    # CT then has different row extents but the same unit spin axis.
    b_X = b_X.reshape(b_X.shape[0], b_X.shape[1] * b_X.shape[2], 1, b_X.shape[3])
    b_Y = b_Y.reshape(b_Y.shape[0], b_Y.shape[1] * b_Y.shape[2], 1, b_Y.shape[3])
    value = build_G(jnp.transpose(b_X, (0, 2, 1, 3)),
                    jnp.transpose(b_Y, (0, 3, 2, 1)),
                    phases=weights, layout=layout, gemm=gemm)
    # build_G is centroid-major (q, mu, s, nu, s'); the unit spin axes are 2, 4.
    return value[:, :, 0, :, 0]


@lru_cache(maxsize=None)
def shared_pole_hole_kernel(mesh_xy):
    """Compile the valence-branch W of an ordered (time-reversal-broken) store.

    An ordered store keeps each parent's positive poles. The occupied branch
    evolves R_-(q) = R_+(-q)^T, so its W at one tau is the complete full-q
    W_+ gathered at -q on the replicated q axis and transposed in its endpoint
    faces. No residue contraction is repeated; q = -q rows give W_+(q)^T.
    """
    from common.collectives import transpose_xy
    return jax.jit(lambda w_full, minus_q: transpose_xy(w_full[minus_q], mesh_xy),
                   out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))


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


def _shared_pole_panel_unfold(meta, header, q_span, *, mesh_xy, tables=None):
    """Realize each parent and apply its local child operation.

    Returns explicit full-q row IDs and a compiled pair-transpose unfold.
    Nonlocal maps refuse here; the caller routes bounded endpoint factors
    through the symmetry service before contraction for those maps.
    """
    from common.shard_map import shard_map
    from gw.qgrid_symmetry import shared_pole_operator_realizer
    from symmetry_maps import unfold_operator_local

    if tables is None:
        tables = _shared_pole_panel_tables(meta, header, q_span, mesh_xy=mesh_xy)
    cert = tables["certificates"]
    if not all(cert[axis]["is_local"] for axis in ("x", "y")):
        raise ValueError("shared-pole nonlocal maps require routed endpoint panels")

    policy = tables["policy"]
    qids = np.asarray(header["q_irr_full_idx"])[slice(*q_span)]
    realize = shared_pole_operator_realizer(
        meta, header, q_full_idx=qids, mesh_xy=mesh_xy)

    def body(plus, transposed):
        projected, _ = policy.project_fixed_q(
            plus, qids, transposed_partner=transposed, measure=False)
        transposed, _ = policy.project_fixed_q(
            transposed, qids, transposed_partner=plus, measure=False)
        return unfold_operator_local(
            projected, irr_idx=tables["parent_rows"], sym_idx=tables["sym_rows"],
            q_irr_frac=tables["q_frac"],
            left_local_perm=cert["x"]["local_perm"], left_L_table=tables["wraps"],
            right_local_perm=cert["y"]["local_perm"], right_L_table=tables["wraps"],
            n_sym_spatial=tables["n_sym_spatial"],
            trs_rule="pair_transpose", transposed_parent_local=transposed)

    unfold_local = jax.jit(shard_map(
        body, mesh=mesh_xy,
        in_specs=(P(None, "x", "y"), P(None, "x", "y")),
        out_specs=P(None, "x", "y"), check_vma=False))

    @jax.jit
    def unfold(plus, transposed):
        return unfold_local(*realize(plus, transposed))

    return tables["rows"], unfold


def _shared_pole_routed_children(meta, header, tables, *, endpoint_budgets, mesh_xy, layout):
    """The τ-invariant half of the routed synthesis, as one program per panel.

    Returns ``(route, count)`` with ``route(b_X, b_Y) -> factors``, a tuple of
    ``count`` faces: each endpoint face routed to the
    panel's child rows by the common wavefunction symmetry owner (antiunitary
    children conjugate the factor, never the causal time weight), followed by
    the conjugate-face partners when a child is self-negative, all placed in
    the contraction's factor layout.  Only d(τ) depends on the node, so a
    panel's factors are routed once per read — once per Σ call for a resident
    panel — not once per τ node (P2-E hoist, claim 2726).  The retained child
    faces replace the parent faces for the read's lifetime: 2(1+f)·16·N_child
    ·μ·K/P bytes per rank on the face layout (f = 1 with a self-negative
    child), inside the endpoint budgets ``_shared_pole_panel_cost`` prices.
    """
    from symmetry_maps import unfold_endpoint_panel

    operations = header["operations"]
    spin = (np.asarray(operations["spin_real"])
            + 1j * np.asarray(operations["spin_imag"]))[tables["sym_rows"]]
    fixed = tables["policy"].self_negative_q[_shared_pole_child_ids(header, tables)]
    specs = _shared_pole_factor_specs(layout)
    count = 4 if np.any(fixed) else 2

    def route(b_X, b_Y):
        children, partners = [], []
        for axis, face in (("x", b_X), ("y", b_Y)):
            kwargs = dict(
                irr_idx=tables["parent_rows"], sym_idx=tables["sym_rows"],
                q_irr_frac=tables["q_frac"], source_perm=tables["packed_perm"],
                L_table=tables["wraps"], spin_action_full=spin,
                n_sym_spatial=tables["n_sym_spatial"], active_mask=meta.mu_basis.active_mask,
                mesh=mesh_xy, mesh_axis=axis, max_live_bytes=endpoint_budgets[axis])
            children.append(unfold_endpoint_panel(face, **kwargs)[0])
            if count == 4:
                partners.append(unfold_endpoint_panel(face.conj(), **kwargs)[0])
        return (*children, *partners)

    return jax.jit(route, out_shardings=tuple(
        NamedSharding(mesh_xy, specs[i % 2]) for i in range(count))), count


def _shared_pole_child_ids(header, tables):
    lo, hi = tables["parent_span"]
    return np.asarray(header["q_irr_full_idx"])[lo:hi][tables["parent_rows"]]


def _shared_pole_routed_synthesis(
    factors, poles2, intervals, E_ref_B, t_node, *, header, tables,
    realize, mesh_xy, gemm, layout="face",
):
    """Synthesize W from routed child factors, DESIGN §3.4 fallback.

    ``factors`` is ``_shared_pole_routed_children``'s output: the two child
    endpoint faces, then their conjugate partners when a child is
    self-negative. No all-star factor cache is retained.
    """
    policy = tables["policy"]
    child_ids = _shared_pole_child_ids(header, tables)
    children, partners = factors[:2], factors[2:]
    weights = _shared_pole_weights(poles2, intervals, E_ref_B, t_node)
    child_weights = weights[tables["parent_rows"]]
    plus = _shared_pole_contract(*children, child_weights, gemm=gemm, layout=layout)
    if partners:
        transposed = _shared_pole_contract(*partners, child_weights, gemm=gemm, layout=layout)
        plus, _ = policy.project_fixed_q(
            plus, child_ids, transposed_partner=transposed, measure=False)
    # Conjugacy of stabilizers makes child-space averaging equivalent to
    # averaging the parent before unfolding. This avoids enlarging the
    # routed factors by a symmetry axis. Both operator orientations remain
    # distributed over the complete mesh, including the transpose exchange.
    from common.collectives import transpose_xy
    transposed = jax.lax.with_sharding_constraint(
        transpose_xy(plus, mesh_xy), NamedSharding(mesh_xy, P(None, "x", "y")))
    return realize(plus, transposed)[0]


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


def _shared_pole_w_synthesis(io, meta, header, frequencies, schedule, *, mesh_xy, layout="face"):
    """Read the factors once and bind the complete full-q W(τ) for the window executable.

    Returns a :class:`WSynthesis`.  The parent faces are read once per Σ call
    (a routed store's are routed to its children once, too) and held for the
    sweep; ``w_kernel`` runs inside every window executable and, per τ node,
    synthesizes each parent panel of ``schedule``'s admitted capacity —
    W_parent = b d(τ) b† → fixed-q projection → little-group realization →
    unfold to its children — and scatters the children into the full-q W.
    Parent panels are a static loop, sequenced so that one panel's temporaries
    live at a time; pole-column chunks of one static width are a device
    ``fori_loop`` over chunk-major slices, and a single chunk when the budget
    admits every column (TASTE 96).  The summation order is the panel-by-panel,
    chunk-by-chunk order of the admitted schedule.
    """
    _band_fence('tau.synthesis_plan', sync_ranks=True)
    with timing.section('tau.synthesis_plan'):
        from file_io.shared_pole_store import face_width, read_shared_pole_faces
        from .sector_sigma import _placer, _zeros

        if schedule["status"] != "PASS":
            raise ValueError("GATE shared_pole_capacity: an admitted schedule is required")
        nq = int(header["n_q_irr"])
        kmax = int(header["Kmax"])
        Q, m = int(header["n_q_full"]), int(meta.mu_basis.n_packed)
        bcap, ccap = int(schedule["parent_capacity"]), int(schedule["column_capacity"])
        if bcap < 1 or ccap < 1:
            raise ValueError("shared-pole panel capacities must be positive")
        # Ordered stores: conduction windows use W_+(q), valence windows W_+(-q)^T.
        ordered = header.get("representation") == "scalar-ordered-ph"
        if kmax == 0:
            zero = _zeros(mesh_xy, (Q, m, m))
            return WSynthesis(lambda _ref, _time, _hole: zero(),
                              lambda _space, _indices, _bounds: (), lambda: (),
                              lambda _result=None: None, 0, ("zero", mesh_xy, Q, m),
                              ordered=ordered)
        factor_specs = _shared_pole_factor_specs(layout)
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
            local = all(c["is_local"] for c in tables["certificates"].values())
            static = _shared_pole_static_key(meta, header, tables, mesh_xy=mesh_xy, layout=layout)
            route = None
            if not local:
                # The panel's faces are routed to its child rows by one bound
                # program: at the read for a single panel (τ-invariant), in
                # the τ body for one panel at a time otherwise.
                budgets = tuple(sorted(schedule["endpoint_budgets"].items()))
                route, _n_faces = _synthesis_program((static, "route", budgets), lambda: (
                    _shared_pole_routed_children(
                        meta, header, tables, endpoint_budgets=schedule["endpoint_budgets"],
                        mesh_xy=mesh_xy, layout=layout)))
            count = hi - lo if local else len(tables["rows"])
            native_workspace = max(native_workspace, workspace_bytes_per_rank(
                workspace_plan, "gemm", ((count, m, width), (count, width, m)),
                np.complex128))
            if "capacity_receipt" in schedule:
                # Include both asynchronous eager warm calls and their
                # throwaway A/B/C operands before the actual factor read.
                # Span-qualified: a ledger stage is an identity, and two panels
                # of equal (count, width) are two reservations.
                factor_bytes = (2*m*width/int(mesh_xy.size) if layout == "face" else
                                m*width*(1/mesh_xy.shape["x"]+1/mesh_xy.shape["y"]))
                warm_bytes = int(16*count*(factor_bytes+m*m/int(mesh_xy.size)))
                meta.shared_pole_capacity.reserve(
                    f"sigma.gemm_warm.{lo}.{hi}.{count}.{width}",
                    resident_bytes_per_rank=0,
                    workspace_bytes_per_rank=2*warm_bytes+native_workspace,
                    concurrent_with=tuple(schedule["capacity_receipt"]["concurrent_with"]))

            def program(span=(lo, hi), tables=tables, local=local, count=count):
                from distrib_la import gemm_plan
                gemm = gemm_plan(mesh_xy, m=m, k=width, n=m, nq=count,
                                 dtype=np.complex128, layout=layout)
                if local:
                    _rows, unfold = _shared_pole_panel_unfold(
                        meta, header, span, mesh_xy=mesh_xy, tables=tables)

                    def body(factors, poles2, ranges, e, t):
                        plus, transposed = synthesize_shared_pole_parents(
                            *factors, poles2, ranges, e, t, mesh_xy=mesh_xy, gemm=gemm,
                            layout=layout)
                        return unfold(plus, transposed)
                else:
                    # The realization is the magnetic little-group average the
                    # store's policy authenticates; on the local branch it is
                    # already inside ``unfold``.
                    from gw.qgrid_symmetry import shared_pole_operator_realizer
                    body = partial(
                        _shared_pole_routed_synthesis, header=header, tables=tables,
                        realize=shared_pole_operator_realizer(
                            meta, header, q_full_idx=tables["rows"], mesh_xy=mesh_xy),
                        mesh_xy=mesh_xy, gemm=gemm, layout=layout)
                return dict(kernel=jax.jit(body))
            kernel = _synthesis_program((static, "synthesis", count, m, width), program)["kernel"]
            panels.append(dict(span=(lo, hi), rows=np.asarray(tables["rows"], np.int32),
                               kernel=kernel, route=route, static=static, count=count))
        schedule["native_gemm_workspace_bytes_per_rank"] = native_workspace
        from symmetry_maps import q_negation_index
        minus_q = np.asarray(q_negation_index(tuple(int(v) for v in header["grid"])))
        hole_kernel = shared_pole_hole_kernel(mesh_xy)
    _band_fence('tau.factor_read', sync_ranks=True)
    with timing.section('tau.factor_read'):
        capacity = getattr(meta, "shared_pole_capacity", None)
        # The stages live beside this Σ call, as the admitted schedule named them.
        ambient = (tuple(schedule["capacity_receipt"]["concurrent_with"])
                   if "capacity_receipt" in schedule else None)
        x, y, poles, _counts = read_shared_pole_faces(
            io, (0, nq), meta=meta, header=header, column_span=(0, kmax) if chunked else None)
        x, y = (_placer(mesh_xy, spec)(a) for spec, a in zip(factor_specs, (x, y)))
        panel_factors, panel_poles = [], []
        for panel in panels:
            lo, hi = panel["span"]
            whole = (lo, hi) == (0, nq)
            fx = (x, y) if whole else (x[lo:hi], y[lo:hi])
            if panel["route"] is not None and whole:
                # One panel: route its children once per Σ call (τ-invariant).
                # Several panels route one panel at a time inside the τ body,
                # so one panel's children are live, as the schedule priced.
                fx = panel["route"](*fx)
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
            capacity.reserve("sigma.synthesis.resident", resident_bytes_per_rank=resident,
                             workspace_bytes_per_rank=0, concurrent_with=ambient)
            capacity.live_stages = (*ambient, "sigma.synthesis.resident")

    spans = tuple((p["span"], p["rows"], p["kernel"],
                   None if p["span"] == (0, nq) else p["route"]) for p in panels)

    def w_kernel(factors_by_panel, poles_by_panel, intervals, e_ref, t_node, hole):
        total = None
        for ((lo, hi), rows, kernel, route), factors, poles2 in zip(
                spans, factors_by_panel, poles_by_panel):
            ranges = intervals[lo:hi]
            if total is not None:
                # One panel's temporaries at a time: the next panel's
                # synthesis waits for the running total.
                total, factors, poles2 = jax.lax.optimization_barrier((total, factors, poles2))
            if not chunked:
                if route is not None:
                    factors = route(*factors)
                child = kernel(factors, poles2, jnp.clip(ranges, 0, width), e_ref, t_node)
                if total is None and (lo, hi) == (0, nq):
                    # All children are in canonical full-q order: the one
                    # all-parent panel IS the full-q W.
                    total = child
                else:
                    base = _zeros(mesh_xy, (Q, m, m))() if total is None else total
                    total = base.at[rows].add(child, indices_are_sorted=True,
                                              unique_indices=True)
                continue

            def chunk(j, acc, factors=factors, poles2=poles2, ranges=ranges,
                      rows=rows, kernel=kernel, route=route):
                selected = jnp.clip(ranges - j*width, 0, width)

                def add(acc):
                    faces = tuple(jax.lax.dynamic_index_in_dim(f, j, 0, keepdims=False)
                                  for f in factors)
                    child = kernel(
                        faces if route is None else route(*faces),
                        jax.lax.dynamic_index_in_dim(poles2, j, 0, keepdims=False),
                        selected, e_ref, t_node)
                    return acc.at[rows].add(child, indices_are_sorted=True,
                                            unique_indices=True)
                # A chunk with no active column in this window adds nothing.
                return jax.lax.cond(jnp.any(selected[:, 1] > selected[:, 0]),
                                    add, lambda acc: acc, acc)
            total = jax.lax.fori_loop(
                0, n_chunks, chunk, _zeros(mesh_xy, (Q, m, m))() if total is None else total)
        if hole:
            return hole_kernel(total, jnp.asarray(minus_q))
        return total

    def window_operands(_space, indices, bounds):
        # Host intervals once per window; every τ node of the window reuses them.
        intervals = shared_pole_intervals(frequencies, np.asarray(indices), np.asarray(bounds))
        return (panel_factors, panel_poles,
                device_put_process_local(intervals, NamedSharding(mesh_xy, P())))

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

    key = ("scalar", mesh_xy, Q, m, width, n_chunks, ordered,
           tuple((p["span"], p["count"], p["static"]) for p in panels))
    return WSynthesis(w_kernel, window_operands, lambda: (panel_factors, panel_poles),
                      close, native_workspace, key, ordered=ordered)


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


def _shared_pole_panel_cost(meta, header, b, c, *, mesh_xy, local, layout="face"):
    """Price actual new E buffers; the incumbent's one full W is inherited.

    Coordinator ruling12 separates unchanged spatial/ψ/Σ peak regression
    from this three-U admission. A child W tile is priced as new even when
    an all-parent first panel can reuse it as the inherited full W.
    """
    from symmetry_maps import endpoint_panel_cost

    # Factors and W tiles are mu x mu charge operators (factor spin axis 1)
    # on scalar and two-component decks; G alone carries the spinor axes.
    m, spin = int(meta.mu_basis.n_packed), 1
    nq = int(header["n_q_irr"])
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    parents = np.asarray(header["qirr"]["irr_idx_q"], dtype=np.int32)
    children = max(int(np.count_nonzero((parents >= lo)
                   & (parents < min(lo+b, nq)))) for lo in range(0, nq, b))
    tile = 16 * (spin*m)**2 // (px*py)
    multiple = combined_divisor(px,py)
    c = padded_axis(c,multiple,name="shared_pole_K_chunk").carrier
    faces = 32 * b * spin*m*c / (px*py)
    endpoint_budgets = {}
    traffic = 0
    if local:
        # Parent, partner and fixed-size group accumulators coexist. The
        # compiled reservation below measures actual aliases and exchange
        # scratch; this bound also informs the panel-size search.
        peak = ((6*b+children)*tile + 8*b
                + 80*b*spin*m*c/(px*py) + 64*b*c)
    else:
        costs = {axis: endpoint_panel_cost((b,m,spin,c), children,
                 mesh=mesh_xy, mesh_axis=axis, dtype=np.complex128)
                 for axis in ("x", "y")}
        endpoint_budgets = {axis: row["estimated_live_bytes_per_rank"]
                            for axis, row in costs.items()}
        traffic = sum(row["ring_bytes_per_rank"] for row in costs.values())
        # The service bounds include input, output, rotating and phase
        # scratch. Extra weighted child faces and child W coexist at GEMM.
        peak = (sum(endpoint_budgets.values()) + 5*children*tile
                + 16*children*spin*m*c/(px*py) + 64*(b+children)*c)
    if layout == "axis":
        axis_faces = 16*b*spin*m*c*(1/px+1/py)
        # Retained axis factors, the reader's face carrier, and weighted
        # GEMM operands coexist; nonlocal routes also keep their child faces.
        peak += axis_faces + 16*(b if local else children)*spin*m*c*(1/px+1/py)
        faces = axis_faces
    return dict(resident_bytes_per_rank=int(np.ceil(faces)),
                workspace_bytes_per_rank=int(np.ceil(peak-faces)),
                endpoint_budgets=endpoint_budgets,
                routed_bytes_per_panel_per_rank=int(traffic), children=children)


def _shared_pole_resident_bytes(meta, header, *, mesh_xy, local, layout, whole=True):
    """Per-rank bytes of the factors the synthesis holds for a whole Σ call.

    All n_q_irr parent faces at the store's whole-K carrier K̄, both
    orientations: 32·n·μ·K̄/P on the face layout, 16·n·μ·K̄·(1/Px+1/Py) with
    the pole columns replicated (axis layout), plus the replicated poles
    8·n·K̄.  A nonlocal (routed) store scheduled as one ``whole`` panel also
    keeps every child face and its conjugate partner (routed once per call),
    the same pair bytes over all Q children; with several panels each
    panel's children are routed inside the τ body and priced as workspace.
    """
    from file_io.shared_pole_store import face_width

    m, nq, Q = (int(meta.mu_basis.n_packed), int(header["n_q_irr"]),
                int(header["n_q_full"]))
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    k = face_width(mesh_xy, int(header["Kmax"]))
    pair = (32*m*k/(px*py) if layout == "face" else 16*m*k*(1/px + 1/py))
    rows = nq if local or not whole else nq + 2*Q
    return int(np.ceil(rows*pair + 8*nq*k))


def _shared_pole_memory_schedule(meta, header, *, mesh_xy, layout="face"):
    """Price the resident factors, size the τ panels from what is left, admit.

    The factors are read once per Σ call and stay resident
    (:func:`_shared_pole_resident_bytes`); only the synthesis workspace of one
    parent panel × pole-column chunk depends on (b, c), and the chunk count
    degenerates to one when everything fits (TASTE 96).  Caller-bound
    live_stages charge other NEW shared-pole objects. Per coordinator
    ruling12, the unchanged spatial/ψ/Σ footprint and its one full-q W are
    reported separately against the incumbent (<=1.05x).
    """
    capacity = getattr(meta, "shared_pole_capacity", None)
    if capacity is None:
        raise ValueError("GATE shared_pole_capacity: missing current-map CapacityLedger")
    concurrent = capacity.live_stages
    accepted = {row["stage"]: row for row in capacity.entries
                if row["device_budget_status"] == "PASS"}
    caller_bytes = sum(accepted[name][key] for name in concurrent for key in
                       ("resident_bytes_per_rank", "workspace_bytes_per_rank"))
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    n, spin, Q = (int(header[key]) for key in
                  ("n_mu_logical", "nspinor", "n_q_full"))
    nq, kmax = int(header["n_q_irr"]), int(header["Kmax"])
    # Compare the geometry itself, in integers. The derived unit is
    # 16*Q*(spin*mu)^2 before the mesh divides it; that numerator passes 2**53
    # at the sizes LORRAX targets, where float equality stops being exact.
    if tuple(int(capacity.geometry[key]) for key in
             ("nq", "nspinor", "nmu", "px", "py")) != (Q, spin, n, px, py):
        raise ValueError("GATE shared_pole_capacity: store/current-map geometry mismatch")
    U = capacity.U_bytes_per_rank
    if kmax == 0:
        receipt = capacity.reserve("sigma.synthesis", resident_bytes_per_rank=0,
                                   workspace_bytes_per_rank=0, concurrent_with=concurrent)
        return dict(status=receipt["status"],parent_capacity=nq,column_capacity=1,
                    capacity_receipt=receipt,route="empty")
    tables = _shared_pole_panel_tables(meta, header, (0,nq), mesh_xy=mesh_xy)
    local = all(c["is_local"] for c in tables["certificates"].values())
    # The ledger owns the hardware limit (ruling24); 3U is a scaling
    # receipt. A zero-byte planning reservation prices the existing ambient set.
    admission = capacity.reserve(
        "sigma.panel_budget", resident_bytes_per_rank=0,
        workspace_bytes_per_rank=0, concurrent_with=concurrent)
    budget = math.floor(admission["available_device_bytes_per_rank"]
                        - admission["aggregate_bytes_per_rank"])
    multiple = combined_divisor(px,py)

    def workspace(b, c, layout):
        return _shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh_xy,local=local,
                                       layout=layout)["workspace_bytes_per_rank"]

    def search(layout):
        best = None
        for b in range(1,nq+1):
            left = budget - _shared_pole_resident_bytes(
                meta, header, mesh_xy=mesh_xy, local=local, layout=layout, whole=b >= nq)
            projection_rows = b if local else _shared_pole_panel_cost(
                meta,header,b,multiple,mesh_xy=mesh_xy,local=local,layout=layout)["children"]
            # The physical logical-U bound applies to every NEW projector
            # matrix, even when orbit packing pads the endpoint carrier. The
            # pre-existing full-q Sigma output is accounted separately above.
            if 16*projection_rows*meta.mu_basis.n_packed**2/(px*py) > U:
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
    b,c = (1,multiple) if best is None else best[2:]
    footprint = _shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh_xy,local=local,layout=layout)
    resident = _shared_pole_resident_bytes(meta, header, mesh_xy=mesh_xy, local=local,
                                           layout=layout, whole=b >= nq)
    projection_rows = b if local else footprint["children"]
    projection_bytes = 16*projection_rows*meta.mu_basis.n_packed**2/(px*py)
    if projection_bytes > U:
        raise ValueError("GATE shared_pole_capacity: one parent star exceeds the all-P logical matrix bound")
    try:
        receipt = capacity.reserve(
            "sigma.synthesis", resident_bytes_per_rank=resident,
            workspace_bytes_per_rank=footprint["workspace_bytes_per_rank"],
            concurrent_with=concurrent)
    except MemoryError as exc:
        # Resident factors and one panel's workspace are per-rank tiles of the
        # all-P operators: every term scales as 1/P on the face layout.
        need = resident + footprint["workspace_bytes_per_rank"]
        side = math.isqrt(max(1, -(-need*px*py // max(1, budget)))-1) + 1
        raise MemoryError(
            f"{exc}; resident shared-pole factors need {resident} B/rank "
            f"(n_q_irr={nq}, Kmax={kmax}, mu={meta.mu_basis.n_packed}, {layout} layout) "
            f"plus {footprint['workspace_bytes_per_rank']} B/rank for one "
            f"{b}-parent x {c}-column synthesis panel, against {budget} B/rank "
            f"available beside the live stages; the smallest square mesh that fits "
            f"them is P >= {side*side} ({side}x{side}; every term scales as 1/P, "
            f"live stages held fixed)") from exc
    return dict(status=receipt["device_budget_status"], unit_bytes=U,
                factor_layout=layout,
                peak_live_bytes_per_rank=receipt["aggregate_bytes_per_rank"],
                peak_in_U=receipt["aggregate_bytes_per_rank"]/U,
                parent_capacity=b,column_capacity=c,
                resident_factor_bytes_per_rank=resident,
                caller_live_bytes_per_rank=caller_bytes,capacity_receipt=receipt,
                route="local_parent" if local else "routed_child",
                endpoint_budgets=footprint["endpoint_budgets"],
                routed_bytes_per_panel_per_rank=footprint["routed_bytes_per_panel_per_rank"],
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


def _admit(compiled, meta, stage, *, native=0, resident=0, counted=0):
    """Reserve a compiled executable's peak; ``counted`` argument bytes are charged elsewhere."""
    from runtime.aot_memory import aot_kernel_peak_bytes
    peak = aot_kernel_peak_bytes(compiled)
    if not peak.cufft_measured:
        raise ValueError(f"GATE shared_pole_capacity: {stage} FFT workspace unavailable")
    meta.shared_pole_capacity.reserve(
        stage, resident_bytes_per_rank=resident,
        workspace_bytes_per_rank=max(0, peak.total - counted) + native,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    return compiled


def _static_key(value):
    """Hashable content key of a small table tree (arrays by bytes digest)."""
    import hashlib
    if isinstance(value, dict):
        return tuple(sorted((k, _static_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_static_key(v) for v in value)
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
    """

    def __init__(self, w_kernel, window_operands, resident_operands, close, native, key,
                 *, ordered):
        self.key = key
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
    ``window_kernel(space, real_phases)`` is the traceable ``fn(*arguments, t, active_count)``
    of :meth:`DeviceOmegaAccumulator.integrate_window`, one per branch hole on
    an ordered model; ``window_arguments`` swaps in the right endpoint's
    operands and the synthesis's per-window operands.  Neither closes over a
    device buffer, so the accumulator's runner cache retains no factors.
    """

    def __init__(self, spatial, synthesis, right_yr, right_proj, native, stage, meta, key, plans):
        self._spatial, self._synthesis = spatial, synthesis
        self._right = (right_yr, right_proj)
        self._native, self._stage, self._meta = native, stage, meta
        self._key, self._plans = key, plans
        self._admitted = False

    def window_kernel(self, space, real_phases):
        """The τ body for ``space``, one function object per static configuration.

        The window runner is cached on this object, so returning the first
        map's body for an equal configuration (same shapes, mesh, layout,
        parent plans and W synthesis) lets every later SC map dispatch the
        compiled window executable instead of recompiling it.
        ``real_phases``: every node time of the window has ``Re t == 0``
        (a Laplace window), so the G phases are real and the antiunitary
        partner is ``conj(G)``; decided on the host once per window, it is
        static in the body, never a runtime predicate.
        """
        hole = space == 'val' and self._synthesis.ordered
        real_phases = bool(real_phases)
        key = (self._key, self._synthesis.key, hole, real_phases)
        if key not in _SYNTHESIS_TAU:
            spatial, w_kernel = self._spatial, self._synthesis.w_kernel

            def tau(xn, yr, xr, yn, energies, weight, w_operands, e_ref_a, e_ref_b, t, _active):
                interactions = w_kernel(*w_operands, e_ref_b, t, hole)
                return spatial(xn, yr, xr, yn, energies, weight, e_ref_a, t, interactions,
                               real_phases=real_phases)
            # The plans ride along so the ids in the key cannot be reused.
            _SYNTHESIS_TAU[key] = (self._plans, tau)
        return _SYNTHESIS_TAU[key][1]

    def window_arguments(self, xn, xr, energies, weight, e_ref_a, e_ref_b, space, indices, bounds):
        w_operands = self._synthesis.window_operands(space, indices, bounds)
        return (xn, self._right[0], xr, self._right[1], energies, weight, w_operands,
                e_ref_a, e_ref_b)

    def admit(self, compiled, arguments):
        """Reserve the first window executable; the resident factors are the synthesis's stage."""
        if self._admitted:
            return
        counted = sum(int(x.addressable_shards[0].data.nbytes)
                      for x in jax.tree.leaves(self._synthesis.resident_operands()))
        _admit(compiled, self._meta, self._stage, native=self._native, counted=counted)
        self._admitted = True


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
        spatial_shape = (
            int(k_unfold_plan.n_parent), sigma_axis.carrier, sigma_axis.carrier)
        face_kwargs["face_band_extent"] = sigma_axis.carrier
        if bracketed:
            shape = (len(brackets), omega.size, *spatial_shape)
            output_sharding = NamedSharding(
                mesh_xy, P(None, None, None, "x", "y"))
        else:
            shape = (omega.size, *spatial_shape)
            output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
        accumulator = DeviceOmegaAccumulator(
            omega, shape=shape, sharding=output_sharding,
            omega_axis=1 if bracketed else 0)
        kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

        if tau_kernel_factory is not None:
            tau_kernel = tau_kernel_factory(w_synthesis, sigma_axis)
        elif synthesis:
            # The scalar shared-pole route: the shared sigma_kij consumes the
            # full-q W its synthesis builds inside the same τ body.
            sigma_kij = _get_sigma_kij_kernel(
                mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True, brackets=brackets,
                **face_kwargs)
            spatial_key = ("scalar", mesh_xy, kgrid, brackets,
                           tuple(sorted(face_kwargs.items(), key=lambda kv: kv[0])))
            tau_kernel = SynthesisTau(
                sigma_kij, w_synthesis, psi_coh_yr, psi_proj_yn, w_synthesis.native,
                "sigma.synthesis.window", meta, spatial_key, (k_unfold_plan,))
        else:
            tau_kernel = get_shared_sigma_tau_kernel(
                mesh_xy=mesh_xy, kgrid=kgrid, brackets=brackets,
                q_wedge=q_wedge, **face_kwargs)
        # The residues' carrier on the wedge: the pair-transpose tables with
        # their device load, placed once per run (see ppm_tau_kernel).
        q_pair = (None if q_wedge is None else dataclasses.replace(
            q_wedge, values=None, load=None, trs_rule="pair_transpose").with_load(mesh_xy))
        small = NamedSharding(mesh_xy, P())
        tau_capacity = max((len(row.window.nodes.t) for row in plan), default=0)

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
                    # G's phases are exp(-i t (E - E_ref)): real exactly
                    # when every node time is imaginary-axis (Re t == 0).
                    real_phases = bool(np.all(np.real(np.asarray(
                        jax.device_get(win.nodes.t), np.complex128)) == 0))
                    row_kernel = tau_kernel.window_kernel(row.space, real_phases)
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
                        jnp.asarray(win.E_ref_A), jnp.asarray(win.E_ref_B))
            window_options = dict(
                active_count=active_count, capacity=tau_capacity,
                omega_sign=win.omega_sign, prefactor=win.prefactor,
                e_ref_sum=win.E_ref_A + win.E_ref_B,
                antihermitian=(win.project_code == 1),
                omega_indices=row.omega_idx, omega_values=row.omega_abs)
            if not sweep_started:
                fence('tau.initial_compile_and_probe', sync_ranks=True)
                with timing.section('tau.initial_compile_and_probe'):
                    # Admit the conjugate-build body (real_phases=False): its
                    # partner G tile makes it the larger of the two variants.
                    compiled = accumulator.integrate_window(
                        tau_kernel.window_kernel(row.space, False) if synthesis
                        else row_kernel, tau_arguments, win.nodes.t,
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
        sigma = _unfold_sigma_cube(
            accumulator.finalize(), k_unfold_plan.sym,
            k_axis=2 if bracketed else 1, sharding=output_sharding)
        if bracketed:
            sigma = _bracket_cumsum_fn(sigma.sharding)(sigma)
            if band_counts is None:
                band_counts = tuple(
                    int(s.nb_sigma_sum) if hi is None else int(hi)
                    for _lo, hi in brackets)
            else:
                band_counts = tuple(int(count) for count in band_counts)
            if len(band_counts) != len(brackets):
                raise ValueError(
                    "MPA Sigma band_counts must align with band brackets")
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


@lru_cache(maxsize=8)
def _bracket_cumsum_fn(sharding):
    """Cumulative band-count sum over the leading bracket axis."""
    return jax.jit(lambda values: jnp.cumsum(values, axis=0),
                   out_shardings=sharding)


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
    """

    def __init__(self, Omega_p, B_p, B_odd_p=None, *, n_mu_logical, mesh_xy,
                 provenance, q_wedge=None):
        from runtime.padding import padded_mu_extent
        shape = tuple(int(n) for n in Omega_p.shape)
        n_mu = int(n_mu_logical)
        if len(shape) != 4 or tuple(B_p.shape) != shape or (
                B_odd_p is not None and tuple(B_odd_p.shape) != shape):
            raise ValueError(
                "MemoryPoleSource: Omega/B/B_odd must share one "
                f"(n_p, n_q, mu, nu) shape; got {shape}, {tuple(B_p.shape)}, "
                f"{None if B_odd_p is None else tuple(B_odd_p.shape)}")
        n_pad = int(padded_mu_extent(n_mu, mesh_xy))
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
            "q_storage": "full" if q_wedge is None else "ibz", "energy_unit": "Ry",
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
    this executor inserts omega behind it and cumulatively sums the brackets
    before returning.  ``None`` preserves the ordinary MPA rank-4 result.
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
            q_wedge=q_wedge, print_fn=print_fn)

    if isinstance(fit_src, (PoleReader, MemoryPoleSource)):
        return run(fit_src)
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        return run(reader)


def _branches(wfns, omega, efermi_ry, occupation_state=None,
              occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
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
    weight f clears the occupancy window at weight f, the cond branch every
    band whose weight 1−f clears it at weight 1−f.  Nothing is clipped; the
    Fermi-Dirac weights metals use lie in [0, 1], and an MP overshoot
    (f<0 or f>1) would ride through unchanged
    (docs/theory/finite-occupation-screening.md).

    ``occupation_window_threshold`` sets that window;
    ``branches_for_omega_grid`` applies it.  1.0 restores the historical
    ``f != 1`` / ``f != 0`` supports bit-for-bit.  Applying it here rather
    than only in the planner keeps ONE support: ``sigma_windows._a_space``
    re-applies the same floor to the same weights, so the two agree by
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
        cond_weight=1.0 - f, val_weight=f,
        occupation_window_threshold=occupation_window_threshold)


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
    quadrature_cache_dir,
    omega_grid_step_ry,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
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
    print_fn=print,
):
    """Read a fitted MPA store, derive its windows, and compute Sigma_c.

    ``occupation_state`` (duck-typed ``gw.efermi.OccupationState``): None is
    the incumbent insulating semantics, bit-exact.  With a state, the causal
    branches carry exact fractional supports and (f, 1−f) weights, and
    ``efermi_ry`` must equal ``occupation_state.mu_ry``.

    ``occupation_window_threshold`` is the OCCUPANCY below which a band is
    still counted in a branch; the cut is ``|weight| > 1 - it``.  It is
    forwarded from this one value to the BRANCH BUILD and to BOTH planner
    entry points, which is what keeps the branch supports, the pole census
    and the window build on one support.

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
    """
    if sigma_w_model not in ("mpa", "shared_pole"):
        raise ValueError(f"sigma_w_model must be mpa or shared_pole; got {sigma_w_model!r}")
    shared_pole = sigma_w_model == "shared_pole"
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
        quadrature_eps = float(recipe["sigma_tolerance"])
        if fixed_quadrature_session is not None:
            fixed_pole_support_ry = (
                recipe.get("sector_pole_treatment") or {}).get("ceiling_ry")
        n_poles = int(ledger["n_q_irr"])
        ordered_residues = False
        with timing.section("sigma.capacity"):
            schedule = (_shared_pole_memory_schedule(meta, ledger, mesh_xy=mesh_xy, layout=wfns.layout)
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
            occupation_state=occupation_state,
            occupation_window_threshold=occupation_window_threshold)
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
                quadrature_cache_dir = sigma_rule_request_cache(
                    quadrature_cache_dir, ledger["identity"],
                    *((poles2, counts) if scope is None else scope),
                    eta=regularization_width_ry, eps=quadrature_eps)
                frequencies = shared_pole_frequencies(poles2, counts)
                summaries = summarize_shared_poles(
                    poles2, counts, branches,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor,
                    occupation_window_threshold=occupation_window_threshold)
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
                        edge_factor=edge_factor,
                        occupation_window_threshold=occupation_window_threshold)
        for lo in (() if shared_pole else range(0, n_poles, int(pole_batch_size))):
            hi = min(lo + int(pole_batch_size), n_poles)
            Omega, B, B_odd = reader.read(
                slice(lo, hi), unfold=getattr(reader, "q_wedge", None) is None,
                return_sharded=True, to_unit="Ry", include_odd=True)
            _refuse_nonfinite_pole_slab(lo, Omega, B, B_odd)
            summaries.extend(summarize_sigma_poles(
                Omega, _geometry_residue(B, B_odd), branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor, pole_offset=lo,
                occupation_window_threshold=occupation_window_threshold))
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
                cache_dir=quadrature_cache_dir,
                print_fn=print_fn, edge_factor=edge_factor,
                fixed_rule_session=fixed_quadrature_session,
                analytic_line=bool(analytic_line),
                material_class=material_class,
                fixed_pole_support_ry=fixed_pole_support_ry,
                certificate_pole_summaries=certificate)
        quadrature_log.record_sigma_plan(geometry)
        print_fn(
            f"  MPA windows [box]: "
            f"eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
            f"eps={geometry['eps']:.3g}, "
            f"certificate={geometry['rule_eps']:.3g}, "
            f"{geometry['n_windows']} logical windows, "
            f"{geometry['window_tau_pairs']} (window,tau) pairs, "
            f"{geometry['distinct_tau_count']} branch-distinct tau, "
            f"cache={geometry['cache_dir'] or 'off'}")
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
                f"max_state_pad={geometry['sc_state_edge_padding_ev']:.3f} eV (energy-proportional), "
                f"pole_pad="
                f"{100.0 * geometry['sc_pole_extent_padding_fraction']:.1f}%")
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
                    f"cache={window['cache_status']}, "
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
                                            sector_context["tau_kernel"]), print_fn=print_fn)
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
