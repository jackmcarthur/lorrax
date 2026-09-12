"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common import timing
from common.progress import LoopProgress
from common.units import RYD_TO_EV
from file_io.mpa_store import PoleReader, open_pole_reader, validate_fit_store
from gw.gw_config import DynamicSigmaConfig
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, _residue_for_space, sigma_band_axis
from gw.ppm_tau_kernel import (TAU_KERNEL_PROFILE_PHASES,
                               get_shared_sigma_tau_kernel)
from gw.ppm_windows import branches_for_omega_grid
from gw.sigma_box_plan import plan_sigma_windows, sigma_rule_request_cache
from gw.sigma_plan import resolve_sigma_plan
from gw.wavefunction_bundle import (
    parent_sigma_operands, sigma_face_kernel_kwargs)
from runtime.env_flags import env_bool
from runtime.padding import combined_divisor, pad_to_axis, round_up

from .sigma_windows import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                            CROSSING_NODE_FLOOR,
                            build_shared_sigma_windows,
                            summarize_sigma_poles,
                            shared_pole_frequencies,
                            shared_pole_intervals,
                            summarize_shared_poles)


# The pane route is an immutable comparison instrument, not a production
# accuracy policy.  Freezing its historical target here lets old/new box-rule
# comparisons keep the same control while retiring the measured-sector deck
# dial from the production path.
_PANE_CONTROL_TARGET_ERROR = 6.5e-4
# The pane CONTROL's own rank cap.  There is no deck pair ceiling any more
# (owner ruling 2026-09-02); the control keeps a generous fixed cap only
# because its legacy planner needs one to size its tables.
_PANE_CONTROL_MAX_RANK = 4096


_DEBUG_GN_ODD_RESIDUE_OFF_ENV = "LORRAX_DEBUG_GN_ODD_RESIDUE_OFF"
_DEBUG_MAX_TAU_DISPATCHES_ENV = "LORRAX_DEBUG_SIGMA_MAX_TAU_DISPATCHES"


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


class _UntimedBand:
    """An unprofiled tau band: entered, never watched.

    Watching is what costs.  ``TimingSection.__exit__`` runs every watcher
    (``common/timing.py``), so a per-tau-node section with a watched result
    is a host synchronization per node -- on the incumbent elementwise-MPA
    route too, which never asked for the measurement.  The tau profile is
    requested with ``LORRAX_SIGMA_TAU_TIMING``; without it the sweep enters
    this object instead and the dispatch stays asynchronous.
    """

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def watch(self, *_values):
        """Accept the band's result and do not block on it."""


_UNTIMED_BAND = _UntimedBand()


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
    b_X, b_Y, poles2, intervals, E_ref_B, t_node, *, mesh_xy, gemm,
):
    """Synthesize both raw-parent orientations through the face service.

    Parameters
    ----------
    b_X, b_Y : jax.Array
        Complex128 physical factors ``[parent,mu,spin,column]`` with
        ``P(None,'x',None,'y')`` / ``P(None,'y',None,'x')`` layouts.
        Only spin=1 is currently supported; endpoints merge in the service.
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
    if b_X.shape[2] != 1 or b_Y.shape[2] != 1:
        raise ValueError("GATE shared_pole_scalar: shared-pole Sigma requires spin=1")
    weights = _shared_pole_weights(poles2, intervals, E_ref_B, t_node)
    return (_shared_pole_contract(b_X, b_Y, weights, gemm=gemm),
            _shared_pole_contract(b_X.conj(), b_Y.conj(), weights, gemm=gemm))


def _shared_pole_contract(b_X, b_Y, weights, *, gemm):
    """W(τ) = b d b† through G's existing two-axis face contraction.

    Factors [q,mu,spin,K] tile (mu,K) over (x,y)/(y,x).
    The causal weight [q,K] is separate and replicated. The permutations
    below are local axis views, giving exactly psi_mun and psi_nmu layouts.
    """
    from gw.greens_function_kernel import build_G

    value = build_G(jnp.transpose(b_X, (0, 2, 1, 3)),
                    jnp.transpose(b_Y, (0, 3, 2, 1)),
                    phases=weights, layout="face", gemm=gemm)
    return value[:, 0, :, 0, :]


def _shared_pole_fixed_q_policy(header):
    """Resolve the policy from the store's authenticated TRS/grid metadata."""
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store

    return qgrid_trs_policy_from_shared_pole_store(header, announce=False)


def _shared_pole_panel_tables(meta, header, q_span, *, mesh_xy):
    """Authenticate packed endpoint maps and one parent's child-row panel."""
    from symmetry_maps import certify_endpoint_locality

    basis, qt = meta.mu_basis, header["qirr"]
    packed = basis.layout.axis.pack_permutations_host(
        np.asarray(qt["sym_perm"], dtype=np.int32), require_local=False)
    certificates = {axis: certify_endpoint_locality(
        packed, mesh=mesh_xy, mesh_axis=axis, active_mask=basis.active_mask)
        for axis in ("x", "y")}
    wraps = basis.layout.axis.pack_host(
        np.asarray(qt["L_table"], dtype=np.int32), axis=1, fill_value=0)
    lo, hi = map(int, q_span)
    parent_map = np.asarray(qt["irr_idx_q"], dtype=np.int32)
    rows = np.flatnonzero((parent_map >= lo) & (parent_map < hi)).astype(np.int32)
    # A finite tangential model need not preserve every spatial little-group
    # relation exactly. The common TRS policy makes q/-q use one spatial
    # realization, as for the ordinary W producer. This changes only small
    # row metadata; the endpoint routing and all-P operator tiles are intact.
    policy = _shared_pole_fixed_q_policy(header)
    return dict(parent_span=(lo, hi), rows=rows, parent_rows=parent_map[rows] - lo,
                sym_rows=policy.unfold_sym_idx[rows],
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

    policy = _shared_pole_fixed_q_policy(header)
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


def _shared_pole_routed_synthesis(
    b_X, b_Y, poles2, intervals, E_ref_B, t_node, *, meta, header, tables,
    endpoint_budgets, realize, mesh_xy, gemm,
):
    """Synthesize W after bounded child-factor routing, DESIGN §3.4 fallback.

    Both endpoint actions use the common wavefunction symmetry owner; this
    conjugates factors for antiunitary children without conjugating the
    causal time weight. No all-star factor cache is retained.
    """
    from symmetry_maps import unfold_endpoint_panel

    operations = header["operations"]
    spin = (np.asarray(operations["spin_real"])
            + 1j * np.asarray(operations["spin_imag"]))[tables["sym_rows"]]
    policy = _shared_pole_fixed_q_policy(header)
    parent_ids = np.asarray(header["q_irr_full_idx"])[tables["parent_span"][0]:tables["parent_span"][1]]
    child_ids = parent_ids[tables["parent_rows"]]
    fixed = policy.self_negative_q[child_ids]
    children = []
    partners = []
    for axis, face in (("x", b_X), ("y", b_Y)):
        child, _ = unfold_endpoint_panel(
            face, irr_idx=tables["parent_rows"], sym_idx=tables["sym_rows"],
            q_irr_frac=tables["q_frac"], source_perm=tables["packed_perm"],
            L_table=tables["wraps"], spin_action_full=spin,
            n_sym_spatial=tables["n_sym_spatial"], active_mask=meta.mu_basis.active_mask,
            mesh=mesh_xy, mesh_axis=axis,
            max_live_bytes=endpoint_budgets[axis])
        if np.any(fixed):
            partner, _ = unfold_endpoint_panel(
                face.conj(), irr_idx=tables["parent_rows"], sym_idx=tables["sym_rows"],
                q_irr_frac=tables["q_frac"], source_perm=tables["packed_perm"],
                L_table=tables["wraps"], spin_action_full=spin,
                n_sym_spatial=tables["n_sym_spatial"], active_mask=meta.mu_basis.active_mask,
                mesh=mesh_xy, mesh_axis=axis, max_live_bytes=endpoint_budgets[axis])
            partners.append(partner)
        children.append(child)
    weights = _shared_pole_weights(poles2, intervals, E_ref_B, t_node)
    child_weights = weights[tables["parent_rows"]]
    plus = _shared_pole_contract(*children, child_weights, gemm=gemm)
    if partners:
        transposed = _shared_pole_contract(*partners, child_weights, gemm=gemm)
        plus, _ = policy.project_fixed_q(
            plus, child_ids, transposed_partner=transposed, measure=False)
    # Conjugacy of stabilizers makes child-space averaging equivalent to
    # averaging the parent before unfolding. This avoids enlarging the
    # routed factors by a symmetry axis. Both operator orientations remain
    # distributed over the complete mesh, including the transpose exchange.
    transposed = jax.lax.with_sharding_constraint(
        jnp.swapaxes(plus, -1, -2), NamedSharding(mesh_xy, P(None, "x", "y")))
    return realize(plus, transposed)[0]


def _shared_pole_w_synthesis(io, meta, header, frequencies, schedule, *, mesh_xy):
    """Resolve bounded face reads → parent synthesis → complete full-q W.

    ``schedule`` is the capacity planner's admitted parent/column capacities
    and receipt. A resident all-parent schedule reads once. Otherwise the
    same store reader supplies bounded panels per τ; panels change storage
    and summation order, never the number of spatial calls. Factor arrays
    live only in this stage closure, never in a global executable cache.
    """
    _band_fence('tau.synthesis_plan', sync_ranks=True)
    with timing.section('tau.synthesis_plan'):
        from functools import partial
        from file_io.shared_pole_store import read_shared_pole_faces

        if schedule["status"] != "PASS":
            raise ValueError("GATE shared_pole_capacity: an admitted schedule is required")
        nq = int(header["n_q_irr"])
        kmax = int(header["Kmax"])
        bcap, ccap = int(schedule["parent_capacity"]), int(schedule["column_capacity"])
        if bcap < 1 or ccap < 1:
            raise ValueError("shared-pole panel capacities must be positive")
        if kmax == 0:
            shape = (int(header["n_q_full"]), meta.mu_basis.n_packed, meta.mu_basis.n_packed)
            zero = jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                           out_shardings=NamedSharding(mesh_xy, P(None,"x","y")))
            return lambda *_args: zero()
        # Query the same distributed dense context used by G before warming
        # any matrix operands. The service accepts a resolved dense Plan for
        # a GEMM workspace query, as in the shared-pole constructor.
        from distrib_la import plan, workspace_bytes_per_rank
        workspace_plan = plan("eigh", mesh_xy, n=meta.mu_basis.n_packed,
                              backend="distributed", batched_route="auto")
        native_workspace = 0
        panels = []
        for lo in range(0, nq, bcap):
            hi = min(lo + bcap, nq)
            tables = _shared_pole_panel_tables(meta, header, (lo, hi), mesh_xy=mesh_xy)
            local = all(c["is_local"] for c in tables["certificates"].values())
            if local:
                rows, unfold = _shared_pole_panel_unfold(
                    meta, header, (lo, hi), mesh_xy=mesh_xy, tables=tables)
            else:
                rows, unfold = tables["rows"], None
            # The realization is the magnetic little-group average the store's
            # policy authenticates.  On the local branch it is already inside
            # ``unfold``; on the routed branch the service accepts traced
            # faces, so bind the immutable map once per current-map panel and
            # reuse its executable at every tau.
            realize = None
            if not local:
                from gw.qgrid_symmetry import shared_pole_operator_realizer
                realize = shared_pole_operator_realizer(
                    meta, header, q_full_idx=rows, mesh_xy=mesh_xy)

            def make_kernel(width, *, span=(lo, hi), tables=tables,
                            unfold=unfold, local=local, realize=realize):
                nonlocal native_workspace
                from distrib_la import gemm_plan

                count = (tables["parent_span"][1]-tables["parent_span"][0]
                         if local else len(tables["rows"]))
                m = meta.mu_basis.n_packed
                native_workspace = max(native_workspace, workspace_bytes_per_rank(
                    workspace_plan, "gemm", ((count,m,width),(count,width,m)),
                    np.complex128))
                schedule["native_gemm_workspace_bytes_per_rank"] = native_workspace
                # Include both asynchronous eager warm calls and their
                # throwaway A/B/C operands before the actual factor read.
                warm_bytes = 16*count*(2*m*width+m*m)//int(mesh_xy.size)
                if "capacity_receipt" in schedule:
                    # Span-qualified like its sigma.synthesis.compiled
                    # sibling below: a ledger stage is an identity, and two
                    # panels of equal (count, width) are two reservations.
                    meta.shared_pole_capacity.reserve(
                        f"sigma.gemm_warm.{span[0]}.{span[1]}.{count}.{width}",
                        resident_bytes_per_rank=0,
                        workspace_bytes_per_rank=2*warm_bytes+native_workspace,
                        concurrent_with=tuple(schedule["capacity_receipt"]["concurrent_with"]))
                gemm = gemm_plan(mesh_xy, m=m, k=width, n=m, nq=count,
                                 dtype=np.complex128)
                if local:
                    def body(b_X,b_Y,poles2,ranges,e,t):
                        plus,transposed = synthesize_shared_pole_parents(
                            b_X,b_Y,poles2,ranges,e,t,mesh_xy=mesh_xy,gemm=gemm)
                        return unfold(plus,transposed)
                else:
                    body = partial(
                        _shared_pole_routed_synthesis, meta=meta, header=header,
                        tables=tables, endpoint_budgets=schedule["endpoint_budgets"],
                        realize=realize, mesh_xy=mesh_xy, gemm=gemm)
                return jax.jit(body)

            kernels = {}
            multiple = combined_divisor(mesh_xy.shape["x"],mesh_xy.shape["y"])
            widths = sorted({round_up(min(ccap,kmax-c0),multiple)
                             for c0 in range(0,kmax,ccap)})
            for width in widths:
                kernel = kernels[width] = make_kernel(width)
                from runtime.aot_memory import aot_kernel_peak_bytes
                def abstract(shape,dtype,spec):
                    return jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh_xy,spec))
                shape = (hi-lo,meta.mu_basis.n_packed,int(header["nspinor"]),width)
                compiled = kernel.lower(
                    abstract(shape,np.complex128,P(None,"x",None,"y")),
                    abstract(shape,np.complex128,P(None,"y",None,"x")),
                    abstract((hi-lo,width),np.float64,P()),
                    abstract((hi-lo,2),np.int32,P()),
                    abstract((),np.float64,P()),abstract((),np.complex128,P())).compile()
                peak = aot_kernel_peak_bytes(compiled)
                row = dict(parent_span=[lo,hi],column_width=width,
                           compiled_bytes_per_rank=peak.total,
                           output_bytes_per_rank=compiled.memory_analysis().output_size_in_bytes,
                           cufft_measured=peak.cufft_measured)
                schedule.setdefault("compiled_panels",[]).append(row)
                if "capacity_receipt" in schedule:
                    if not peak.cufft_measured:
                        raise ValueError("shared-pole synthesis native FFT workspace query unavailable")
                    meta.shared_pole_capacity.reserve(
                        f"sigma.synthesis.compiled.{lo}.{hi}.{width}",
                        resident_bytes_per_rank=0,
                        workspace_bytes_per_rank=peak.total+native_workspace,
                        concurrent_with=tuple(schedule["capacity_receipt"]["concurrent_with"]))
            schedule["compiled_peak_status"] = "PASS"
            panels.append((lo, hi, device_put_process_local(
                rows, NamedSharding(mesh_xy, P())), kernels, make_kernel))
    _band_fence('tau.factor_read', sync_ranks=True)
    with timing.section('tau.factor_read'):
        resident = None
        if bcap >= nq and ccap >= kmax:
            resident = read_shared_pole_faces(io, (0, nq), meta=meta, header=header)
    shape = (int(header["n_q_full"]), meta.mu_basis.n_packed, meta.mu_basis.n_packed)
    sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    zeros = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=sharding)

    @partial(jax.jit, donate_argnums=(0,), out_shardings=sharding)
    def add_panel(total, rows, values):
        # Every full-q child occurs once in a parent panel, in sorted order.
        # Expose that fact so complex scatter need not use atomic updates.
        return total.at[rows].add(values, indices_are_sorted=True, unique_indices=True)

    cached_indices = cached_bounds = cached_intervals = None

    def build(_residues, _omega_fields, indices, bounds, _phase_real, E_ref_B, t_node):
        nonlocal cached_indices, cached_bounds, cached_intervals
        _band_fence("tau.W_synthesis")
        with timing.section("tau.W_synthesis"):
            # Fixed-q projection remains in the symmetry owner. The extra tau=1
            # diagnostic replay is covered by the fixed-q acceptance tests.
            if indices is not cached_indices or bounds is not cached_bounds:
                cached_intervals = shared_pole_intervals(
                    frequencies, np.asarray(jax.device_get(indices)),
                    np.asarray(jax.device_get(bounds)))
                cached_indices, cached_bounds = indices, bounds
            intervals = cached_intervals
            total = None
            for lo, hi, rows, kernels, make_kernel in panels:
                for c0 in range(0, kmax, ccap):
                    c1 = min(c0 + ccap, kmax)
                    selected = np.clip(intervals[lo:hi] - c0, 0, c1 - c0)
                    if not np.any(selected[:, 1] > selected[:, 0]):
                        continue
                    faces = resident if resident is not None else read_shared_pole_faces(
                        io, (lo, hi), meta=meta, header=header, column_span=(c0, c1))
                    b_X, b_Y, poles2, _counts = faces
                    ranges = device_put_process_local(selected, NamedSharding(mesh_xy, P()))
                    # Preserve the admitted K tiling for the entire panel.
                    # Window-dependent slicing can redistribute the pole axis
                    # and recreate a one-axis factor temporary. The separate
                    # causal weights mask inactive columns without moving b.
                    width = b_X.shape[-1]
                    if width not in kernels:
                        kernels[width] = make_kernel(width)
                    kernel = kernels[width]
                    child = kernel(b_X,b_Y,poles2,ranges,E_ref_B,t_node)
                    if resident is None:
                        child.block_until_ready()
                    del faces, b_X, b_Y, poles2, _counts, ranges
                    if total is None and lo == 0 and hi == nq:
                        # All children are in canonical full-q order. Avoid a
                        # redundant zero buffer in the resident all-parent case.
                        total = child
                    else:
                        if total is None:
                            total = zeros()
                        total = add_panel(total, rows, child)
                    # Complete this panel before the next collective read can
                    # allocate another face carrier (the admitted live set is one).
                    if resident is None:
                        total.block_until_ready()
                    del child
            result = zeros() if total is None else total
            jax.block_until_ready(result)
            return result

    return build


def _shared_pole_panel_cost(meta, header, b, c, *, mesh_xy, local):
    """Price actual new E buffers; the incumbent's one full W is inherited.

    Coordinator ruling12 separates unchanged spatial/ψ/Σ peak regression
    from this three-U admission. A child W tile is priced as new even when
    an all-parent first panel can reuse it as the inherited full W.
    """
    from symmetry_maps import endpoint_panel_cost

    m, spin = int(meta.mu_basis.n_packed), int(header["nspinor"])
    nq = int(header["n_q_irr"])
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    parents = np.asarray(header["qirr"]["irr_idx_q"], dtype=np.int32)
    children = max(int(np.count_nonzero((parents >= lo)
                   & (parents < min(lo+b, nq)))) for lo in range(0, nq, b))
    tile = 16 * (spin*m)**2 // (px*py)
    multiple = combined_divisor(px,py)
    c = round_up(c,multiple)
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
    return dict(resident_bytes_per_rank=int(np.ceil(faces)),
                workspace_bytes_per_rank=int(np.ceil(peak-faces)),
                endpoint_budgets=endpoint_budgets,
                routed_bytes_per_panel_per_rank=int(traffic), children=children)


def _shared_pole_memory_schedule(meta, header, *, mesh_xy):
    """Choose bounded panels, then admit through the canonical map ledger.

    Caller-bound live_stages charge other NEW shared-pole objects. Per
    coordinator ruling12, the unchanged spatial/ψ/Σ footprint and its one
    full-q W are reported separately against the incumbent (<=1.05x).
    This schedule never certifies that inherited no-regression gate.
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
                    capacity_receipt=receipt,route="empty",compiled_peak_status="NOT_APPLICABLE")
    tables = _shared_pole_panel_tables(meta, header, (0,nq), mesh_xy=mesh_xy)
    local = all(c["is_local"] for c in tables["certificates"].values())
    # The ledger owns the hardware limit (ruling24); 3U is a scaling
    # receipt, not a reason to reread resident factors at every tau node.
    # A zero-byte planning reservation prices the existing ambient set.
    admission = capacity.reserve(
        "sigma.panel_budget", resident_bytes_per_rank=0,
        workspace_bytes_per_rank=0, concurrent_with=concurrent)
    budget = math.floor(admission["available_device_bytes_per_rank"]
                        - admission["aggregate_bytes_per_rank"])
    best = None
    for b in range(1,nq+1):
        # Byte counts are affine in the column width. Price through the
        # SAME routine used for the admitted row, including routed scratch.
        multiple = combined_divisor(px,py)
        one = _shared_pole_panel_cost(meta,header,b,multiple,mesh_xy=mesh_xy,local=local)
        projection_rows = b if local else one["children"]
        # The physical logical-U bound applies to every NEW projector
        # matrix, even when orbit packing pads the endpoint carrier. The
        # pre-existing full-q Sigma output is accounted separately above.
        projection_bytes = 16*projection_rows*(spin*meta.mu_basis.n_packed)**2/(px*py)
        if projection_bytes > U:
            continue
        two = _shared_pole_panel_cost(meta,header,b,2*multiple,mesh_xy=mesh_xy,local=local)
        keys = ("resident_bytes_per_rank", "workspace_bytes_per_rank")
        p1, p2 = sum(one[k] for k in keys), sum(two[k] for k in keys)
        # price(j column multiples) = intercept + j*slope. Both ends are
        # ceilinged byte counts, so on a small enough panel they can land on
        # the same integer: a non-positive slope carries no width information
        # and must not size c (nor divide by zero). Integer floor division
        # keeps the sizing exact past 2**53.
        slope, intercept = p2 - p1, 2*p1 - p2
        if slope <= 0:
            continue
        c = min(kmax, multiple*((budget - intercept)//slope))
        if c < 1:
            continue
        cost = ((nq+b-1)//b)*((kmax+c-1)//c)
        candidate = (cost,-b*c,b,c)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    b,c = (1,1) if best is None else best[2:]
    footprint = _shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh_xy,local=local)
    projection_rows = b if local else footprint["children"]
    projection_bytes = 16*projection_rows*(spin*meta.mu_basis.n_packed)**2/(px*py)
    if projection_bytes > U:
        raise ValueError("GATE shared_pole_capacity: one parent star exceeds the all-P logical matrix bound")
    receipt = capacity.reserve(
        "sigma.synthesis", resident_bytes_per_rank=footprint["resident_bytes_per_rank"],
        workspace_bytes_per_rank=footprint["workspace_bytes_per_rank"],
        concurrent_with=concurrent)
    return dict(status=receipt["device_budget_status"], unit_bytes=U,
                peak_live_bytes_per_rank=receipt["aggregate_bytes_per_rank"],
                peak_in_U=receipt["aggregate_bytes_per_rank"]/U,
                parent_capacity=b,column_capacity=c,
                caller_live_bytes_per_rank=caller_bytes,capacity_receipt=receipt,
                route="local_parent" if local else "routed_child",
                endpoint_budgets=footprint["endpoint_budgets"],
                routed_bytes_per_panel_per_rank=footprint["routed_bytes_per_panel_per_rank"],
                inherited_sigma_peak_status="NOT_MEASURED",
                projection_matrix_bytes_per_rank=int(projection_bytes),
                compiled_peak_status="NOT_MEASURED")


def _shared_pole_inherited_peak(args, meta, *, mesh_xy, kgrid, brackets,
                                pack_brackets, face_kwargs):
    """Compare inherited Sigma lower bounds at the exact current operands.

    The control is the incumbent already-phased W carrier used by the frozen
    evaluator; the candidate supplies that same full W through the new seam.
    Both invoke the unchanged G/spatial/projection owner with the SAME ψ,
    energy/occupation selector, time and output shape. New W synthesis lives
    in its separate three-U reservation. Native scratch is excluded equally
    from this compiler-only regression, as required by coordinator ruling12.
    """
    from runtime.aot_memory import aot_kernel_peak_bytes

    capacity = meta.shared_pole_capacity
    m = int(meta.mu_basis.n_packed) * int(meta.nspinor)
    q = int(meta.nk_tot)
    small = NamedSharding(mesh_xy, P())
    W = jax.ShapeDtypeStruct((1,q,m,m), np.complex128,
                            sharding=NamedSharding(mesh_xy, P(None,None,"x","y")))
    def scalar(value):
        return device_put_process_local(np.asarray(value), small)
    same_args = (*args[:6], W, scalar([0.0+0.0j]), scalar(np.array([0],np.int32)),
                 scalar([[-np.inf,np.inf,-np.inf,-np.inf,np.inf,np.inf]]),
                 scalar([False]), args[11], scalar(0.0), args[13])
    def phased_w(B, _omega, _indices, _bounds, _real, _ref, _time):
        return B[0]
    peaks = []
    for builder in (None, phased_w):
        kernel = get_shared_sigma_tau_kernel(
            mesh_xy=mesh_xy,kgrid=kgrid,brackets=brackets,pack_brackets=pack_brackets,
            w_synthesis=builder,**face_kwargs)
        compiled = jax.jit(kernel).lower(*same_args).compile()
        peaks.append(aot_kernel_peak_bytes(compiled).compiled_peak)
    return capacity.record_sigma_peak(
        peaks[1],peaks[0],reason=(
            "Matched current Sigma operands and same spatial owner; compiler-only "
            "argument+output+temporary-alias lower bounds, native scratch excluded "
            "equally; control=incumbent phased-W carrier, candidate=shared-pole "
            "injection with the same full W; first-window shape covers all tau nodes"))


def _resolve_debug_max_tau_dispatches(*, print_fn=print):
    """Return the debug-only bounded-sweep length, or ``None``.

    A bounded sweep is a performance instrument, not a quadrature rule: the
    executor exits cleanly after the requested number of real-shape tau
    dispatches and never returns a partial Sigma cube to an output consumer.
    """
    raw = os.environ.get(_DEBUG_MAX_TAU_DISPATCHES_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_DEBUG_MAX_TAU_DISPATCHES_ENV} must be a positive integer; "
            f"got {raw!r}") from exc
    if count <= 0:
        raise ValueError(
            f"{_DEBUG_MAX_TAU_DISPATCHES_ENV} must be a positive integer; "
            f"got {raw!r}")
    print_fn(
        "WARNING -- DEBUG: "
        f"{_DEBUG_MAX_TAU_DISPATCHES_ENV}={count}; the MPA Sigma executor "
        "will stop after that many tau dispatches and WILL NOT produce "
        "scientific Sigma/QP output.")
    return count


# The sweep's OWN bands, spelled once and opened under these names below.
# The kernel-internal bands are owned by ``gw.ppm_tau_kernel``; importing its
# tuple rather than restating it is what keeps the profile complete.
_TAU_SWEEP_KERNEL_PHASE = "tau.kernel"
_TAU_SWEEP_ACCUMULATOR_PHASE = "tau.accumulator"
_TAU_SWEEP_PROGRESS_PHASE = "tau.progress"

_TAU_PROFILE_PHASES = TAU_KERNEL_PROFILE_PHASES + (
    _TAU_SWEEP_KERNEL_PHASE,
    _TAU_SWEEP_ACCUMULATOR_PHASE,
    _TAU_SWEEP_PROGRESS_PHASE,
)


def _tau_profile_snapshot():
    """Aggregate the tau profiler's timing nodes by local section name."""
    totals = {}
    for record in timing.records():
        name = record["name"]
        if name not in _TAU_PROFILE_PHASES:
            continue
        count, seconds = totals.get(name, (0, 0.0))
        totals[name] = (
            count + int(record["count"]),
            seconds + float(record["inclusive"]),
        )
    return totals


def _debug_probe_print(line):
    """Emit one live rank-zero line outside the production report sink."""
    if jax.process_index() == 0:
        # gw_jax deliberately sends incidental stdout to /dev/null.  This
        # opt-in diagnostic must survive that production stream boundary.
        print(line, file=sys.stderr, flush=True)


def _print_tau_profile(before, *, n_tau, print_fn=_debug_probe_print):
    """Print post-prewarm timing deltas for the staged tau diagnostic."""
    after = _tau_profile_snapshot()
    print_fn("--- Sigma tau phase profile (post-prewarm, blocking) ---")
    print_fn(f"{'Phase':<31} {'Count':>7} {'Total[s]':>11} {'s/dispatch':>13}")
    for name in _TAU_PROFILE_PHASES:
        count0, seconds0 = before.get(name, (0, 0.0))
        count1, seconds1 = after.get(name, (0, 0.0))
        count = count1 - count0
        seconds = seconds1 - seconds0
        if count <= 0:
            continue
        print_fn(
            f"{name:<31} {count:>7d} {seconds:>11.6f} "
            f"{seconds / max(1, n_tau):>13.6f}")


def _resolve_mpa_odd_residue_debug(ordered_residues, *, print_fn=print):
    """Resolve the shared GN/MPA odd-residue A/B switch for an MPA fit."""
    enabled = env_bool(
        _DEBUG_GN_ODD_RESIDUE_OFF_ENV, False, print_fn=print_fn)
    if enabled and not bool(ordered_residues):
        raise ValueError(
            "GATE debug_gn_odd_residue_off_scope:\n"
            f"  got:  {_DEBUG_GN_ODD_RESIDUE_OFF_ENV}=1 with an "
            "MPA single-residue/TRS fit\n"
            "  want: this debug switch only on a measured-broken-TR "
            "ordered-residue MPA fit\n"
            "  why:  a TRS MPA fit has no time-reversal-odd residue to "
            "discard\n"
            "  fix:   unset LORRAX_DEBUG_GN_ODD_RESIDUE_OFF")
    if enabled:
        print_fn(
            "WARNING -- DEBUG: LORRAX_DEBUG_GN_ODD_RESIDUE_OFF=1; "
            "measured-broken-TR MPA fit is discarding the "
            "anti-Hermitian frequency-odd residue: D=0 and R+=R-=B. "
            "This arm is for A/B diagnosis only, never production.")
    return enabled


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
    shapes, dtypes, and shardings; only the selector values change.
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
        None,
    )


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
    print_fn,
):
    """One spatial executor for streamed fit slabs."""
    # Band fences are profiling boundaries (see ``_unfenced``).  This executor
    # is shared with the incumbent elementwise-MPA route (``w_synthesis is
    # None``), which is never fenced whatever a harness has installed.
    fence = _band_fence if w_synthesis is not None else _unfenced
    fence('tau.setup', sync_ranks=True)
    with timing.section('tau.setup'):
        omega = np.asarray(omega_grid_ry, np.float64)
        if omega.ndim != 1 or not omega.size:
            raise ValueError("omega_grid_ry must be a nonempty vector")
        debug_max_tau = _resolve_debug_max_tau_dispatches(print_fn=print_fn)
        tau_profile = env_bool(
            "LORRAX_SIGMA_TAU_TIMING", False, print_fn=print_fn)

        def tau_band(name):
            """Enter one per-tau-node band.

            The fence is the band-attribution seam and is free in production
            (``_unfenced`` above, and always so on the incumbent route); the
            SECTION is what synchronizes, so it is opened only when the tau
            profile was requested.  ``_print_tau_profile`` consumes exactly
            these names.
            """
            fence(name)
            return timing.section(name) if tau_profile else _UNTIMED_BAND

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
        k_unfold_plan = face_kwargs.get("k_unfold_plan")
        if wfns.layout == "legacy":
            # ``sigma_sum``, not ``full`` — the Σ band sum, not the loaded
            # extent.  Identical on an unsplit deck.  UNVERIFIED on a split
            # one: the public MPA Σ still refuses to run (gw_config.
            # ComputeMode), so this line is wired for consistency and has
            # never executed under a split.
            state_slice = s.full if bracketed else s.sigma_sum
            psi_coh_xn, psi_coh_yr = wfns.xn(state_slice), wfns.yr(state_slice)
            psi_proj_xr, psi_proj_yn = wfns.xr(s.sigma), wfns.yn(s.sigma)
            psi_proj_xr = pad_to_axis(
                psi_proj_xr, sigma_axis, axis=1)
            psi_proj_yn = pad_to_axis(
                psi_proj_yn, sigma_axis, axis=3)
            spatial_shape = (int(psi_proj_xr.shape[0]),
                             int(psi_proj_xr.shape[1]),
                             int(psi_proj_yn.shape[3]))
        else:
            # Face carrier (2026-08-22, mechanical port sharing
            # ppm_tau_kernel's face dispatch — see gw.ppm_sigma._run_sigma_
            # branch's identically-shaped docstring): psi_mun/psi_nmu are used
            # UNSLICED for both roles — the accumulator BUILDS at the mesh-
            # divisible nb_full extent regardless of nb_sigma, and always
            # will: contract_bands.contract_bands_block_reshard's face arm
            # The face projector's GEMM plan now takes the requested projection
            # carrier separately from the resident full-band face.  The producer
            # selects the logical Sigma window, appends exact-zero rows to the
            # runtime-owned carrier, and the accumulator is born at that carrier
            # width.  It stays there until a logical output consumer strips by
            # ``sigma_axis``; no nondivisible sharded array is ever published.
            # The projection operands default to the full-k faces; the parent
            # route below replaces both roles with the parent faces.
            psi_proj_xr, psi_proj_yn = wfns.psi_nmu, wfns.psi_mun
            if k_unfold_plan is not None:
                # Raw parents only: the parent faces feed the G contraction (the
                # plan transports G to full k) and the projection (the spatial
                # tail selects, projects, broadcasts).  Bracket packing is not
                # combined with the route.
                (psi_coh_xn, psi_coh_yr,
                 psi_proj_xr, psi_proj_yn, _, _) = parent_sigma_operands(wfns)
                pack_brackets = False
            elif bracketed and len(brackets) > 1:
                from gw.wavefunction_bundle import pack_band_window
                packed = [pack_band_window(wfns, lo, hi, mesh_xy=mesh_xy)
                          for lo, hi in brackets]
                psi_coh_xn = tuple(pair[0] for pair in packed)
                psi_coh_yr = tuple(pair[1] for pair in packed)
                pack_brackets = True
            else:
                psi_coh_xn, psi_coh_yr = wfns.psi_mun, wfns.psi_nmu
                pack_brackets = False
            psi_proj_xr = pad_to_axis(
                psi_proj_xr, sigma_axis, axis=1)
            psi_proj_yn = pad_to_axis(
                psi_proj_yn, sigma_axis, axis=3)
            spatial_shape = (
                int(meta.nk_tot), sigma_axis.carrier, sigma_axis.carrier)
            face_kwargs["face_band_extent"] = sigma_axis.carrier
        if wfns.layout == "legacy":
            pack_brackets = False
        if bracketed:
            shape = (len(brackets), omega.size, *spatial_shape)
            output_sharding = NamedSharding(
                mesh_xy, P(None, None, None, "x", "y"))
            sigma_shape = (len(brackets), *spatial_shape)
            sigma_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
        else:
            shape = (omega.size, *spatial_shape)
            output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
            sigma_shape = spatial_shape
            sigma_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
        accumulator = DeviceOmegaAccumulator(
            omega, shape=shape, sharding=output_sharding,
            omega_axis=1 if bracketed else 0)
        kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
        tau_kernel = get_shared_sigma_tau_kernel(
            mesh_xy=mesh_xy,
            kgrid=kgrid,
            brackets=brackets,
            pack_brackets=pack_brackets,
            w_synthesis=w_synthesis,
            **face_kwargs)
        small = NamedSharding(mesh_xy, P())

        n_sweeps = n_tau = 0
        logical_tau_pairs = 0
        batch_size = int(pole_batch_size)
        sweep_started = False
        max_b = max_d = 0.0
        # The same bar as the zeta fit and the W roles (common.progress); the
        # step count is exact because window/batch membership is a pure function
        # of the pole ranges the sweep will visit.  Owner request 2026-09-03.
        total_tau = 0
        if w_synthesis is not None:
            # All parent/column panels finish W inside each tau call. They are
            # storage work, never additional state-pole product windows.
            total_tau = sum(len(np.asarray(row.window.nodes.t)) for row in plan)
        else:
            for _lo in range(0, int(n_poles), batch_size):
                _batch = tuple(range(_lo, min(_lo + batch_size, int(n_poles))))
                for _row in plan:
                    if _batch_rows(_row, _batch) is not None:
                        total_tau += len(np.asarray(_row.window.nodes.t))
        progress_total = (
            total_tau if debug_max_tau is None
            else min(total_tau, debug_max_tau))
        progress = LoopProgress(
            max(1, progress_total), print_fn, title="Sigma tau sweep",
            item_name="tau node", max_updates=20)
        progress.start()
        profile_before = None
        sweep_wall_start = None
        stop_probe = False
    for lo, Omega, B, B_odd in batches:
        if w_synthesis is None and getattr(meta, 'mu_basis', None) is not None:
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
        width = 0 if w_synthesis is not None else int(Omega.shape[0])
        batch = tuple(range(int(lo), int(lo) + width))
        for row in plan:
            selected = (
                (row.pole_indices, row.bounds, row.phase_real, None)
                if w_synthesis is not None else _batch_rows(row, batch))
            if selected is None:
                continue
            fence('tau.window_arguments', sync_ranks=True)
            with timing.section('tau.window_arguments'):
                pole_indices, bounds, phase_real, _states = selected
                pole_indices, bounds, phase_real = (
                    device_put_process_local(x, small)
                    for x in (pole_indices, bounds, phase_real))
                win = row.window
                B_branch = (None if w_synthesis is not None else
                            _residue_for_space(row.space, B, B_odd))
                weight = getattr(row, "band_weight", None)
                if weight is None:
                    selector = jnp.asarray(win.mask_A)
                else:
                    selector = (jnp.asarray(win.mask_A, jnp.float64)
                                * jnp.reshape(
                                    jnp.asarray(weight, jnp.float64),
                                    np.asarray(win.mask_A).shape))
                E_A_call = row.E_A
                if k_unfold_plan is not None:
                    # The G contraction runs on the raw parents: its energy and
                    # selector tables are the parents' rows of the star-invariant
                    # full-k tables (one child per raw row, plan.parent_rows).
                    E_A_call = k_unfold_plan.parent_rows(row.E_A)
                    selector = k_unfold_plan.parent_rows(
                        jnp.reshape(selector, np.shape(row.E_A)))
            if not sweep_started:
                fence('tau.initial_compile_and_probe', sync_ranks=True)
                with timing.section('tau.initial_compile_and_probe'):
                    first_t = np.asarray(
                        jax.device_get(win.nodes.t), np.complex128)[0]
                    prewarm_args = (
                        psi_coh_xn, psi_coh_yr,
                        psi_proj_xr, psi_proj_yn,
                        E_A_call, selector, B_branch, Omega,
                        pole_indices, bounds, phase_real,
                        jnp.asarray(win.E_ref_A),
                        jnp.asarray(win.E_ref_B),
                        jnp.asarray(first_t, dtype=jnp.complex128))
                    if w_synthesis is not None:
                        inherited = _shared_pole_inherited_peak(
                            prewarm_args,meta,mesh_xy=mesh_xy,kgrid=kgrid,
                            brackets=brackets,pack_brackets=pack_brackets,
                            face_kwargs=face_kwargs)
                        print_fn(f"  shared-pole inherited Sigma peak: {inherited}")
                    if hasattr(tau_kernel, "lower"):
                        tau_kernel.lower(*prewarm_args).compile()
                    else:
                        # The stage-split diagnostic is a Python dispatcher over
                        # separately-jitted stages.  Execute one real-shape call
                        # to prewarm the same kernels the timed sweep will use.
                        jax.block_until_ready(tau_kernel(*prewarm_args))
                    accumulator.precompile_tau_add(
                        sigma_shape=sigma_shape,
                        sigma_sharding=sigma_sharding)
                    print_fn(
                        "  MPA Sigma sweep begin: shared pane tau kernel "
                        "prewarmed")
                    profile_before = _tau_profile_snapshot()
                    sweep_wall_start = time.perf_counter()
                    sweep_started = True
            fence('tau.window_setup', sync_ranks=True)
            with timing.section('tau.window_setup'):
                t_nodes = np.asarray(
                    jax.device_get(win.nodes.t), np.complex128)
                alpha_nodes = np.asarray(
                    jax.device_get(win.nodes.alpha), np.complex128)
                if debug_max_tau is not None:
                    remaining = debug_max_tau - n_tau
                    if remaining <= 0:
                        stop_probe = True
                        break
                    t_nodes = t_nodes[:remaining]
                    alpha_nodes = alpha_nodes[:remaining]
                accumulator.begin_window(
                    t_nodes, alpha_nodes,
                    omega_sign=win.omega_sign, prefactor=win.prefactor,
                    e_ref_sum=win.E_ref_A + win.E_ref_B,
                    antihermitian=(win.project_code == 1),
                    omega_indices=row.omega_idx,
                    omega_values=row.omega_abs)
            for t in t_nodes:
                tau_args = (
                    psi_coh_xn, psi_coh_yr,
                    psi_proj_xr, psi_proj_yn,
                    E_A_call, selector, B_branch, Omega,
                    pole_indices, bounds, phase_real,
                    jnp.asarray(win.E_ref_A),
                    jnp.asarray(win.E_ref_B),
                    jnp.asarray(t, dtype=jnp.complex128))
                with tau_band(_TAU_SWEEP_KERNEL_PHASE) as sec:
                    sigma_tau = tau_kernel(*tau_args)
                    sec.watch(sigma_tau)
                with tau_band(_TAU_SWEEP_ACCUMULATOR_PHASE) as sec:
                    # Unprofiled, production ignores this return and keeps the
                    # incumbent asynchronous path (ppm_accumulators.py:169-170).
                    sec.watch(accumulator.add_tau(sigma_tau))
                with tau_band(_TAU_SWEEP_PROGRESS_PHASE):
                    # Unprofiled, the bar blocks only at its own milestones;
                    # profiled, the kernel band has already synchronized.
                    progress.step(wait=None if tau_profile else sigma_tau)
                n_tau += 1
            fence('tau.window_finish', sync_ranks=True)
            with timing.section('tau.window_finish'):
                accumulator.end_window()
                n_sweeps += 1
                logical_tau_pairs += len(t_nodes)
            if debug_max_tau is not None and n_tau >= debug_max_tau:
                stop_probe = True
                break
        del B, B_odd, Omega
        gc.collect()
        if stop_probe:
            break
    progress.finish()

    if debug_max_tau is not None:
        # Close every outstanding asynchronous accumulator/end-window update
        # before stopping the measurement clock.  The partial cube dies here;
        # it is never wrapped in SigmaOmegaResult or handed to an output path.
        jax.block_until_ready(accumulator.finalize())
        elapsed = time.perf_counter() - sweep_wall_start
        _debug_probe_print(
            f"  DEBUG bounded Sigma tau sweep: {n_tau} dispatches in "
            f"{elapsed:.6f} s ({elapsed / max(1, n_tau):.6f} s/dispatch)")
        if tau_profile:
            _print_tau_profile(
                profile_before, n_tau=n_tau)
        _debug_probe_print(
            "  DEBUG bounded Sigma tau sweep complete; exiting before "
            "Sigma/QP output (intentional rc=0).")
        raise SystemExit(0)

    fence('tau.finalize', sync_ranks=True)
    with timing.section('tau.finalize'):
        sigma = accumulator.finalize()
        if bracketed:
            sigma = jax.jit(
                lambda values: jnp.cumsum(values, axis=0),
                out_shardings=sigma.sharding)(sigma)
            if band_counts is None:
                band_counts = tuple(
                    int(s.nb_sigma_sum) if hi is None else int(hi)
                    for _lo, hi in brackets)
            else:
                band_counts = tuple(int(count) for count in band_counts)
            if len(band_counts) != len(brackets):
                raise ValueError(
                    "MPA Sigma band_counts must align with band brackets")
        transform_saving = int(logical_tau_pairs - n_tau)
        print_fn(
            f"  MPA Sigma: {n_tau} tau dispatches in {n_sweeps} sweeps "
            f"({n_poles} poles, batches of {batch_size}); "
            f"{transform_saving} undispatched logical tau; "
            f"panes and product windows used one shared tau kernel")
        ratio = None
        if max_b or max_d:
            ratio = max_d / max_b if max_b else np.inf
            state = "DEBUG ODD OFF (D discarded)" if odd_residue_off else "enabled"
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
    :class:`~file_io.mpa_store.PoleReader` whose collective handle the
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

    def batches(reader):
        for lo in range(0, int(n_poles), batch_size):
            hi = min(lo + batch_size, int(n_poles))
            Omega, B, B_odd = reader.read(
                slice(lo, hi), unfold=True, return_sharded=True,
                to_unit="Ry", include_odd=True)
            yield lo, Omega, B, B_odd
            del Omega, B, B_odd
            gc.collect()

    def run(reader):
        return _integrate_sigma_batches(
            wfns, batches(reader), int(n_poles), plan, omega_grid_ry, meta,
            mesh_xy, pole_batch_size=batch_size, brackets=brackets,
            band_counts=band_counts, odd_residue_off=odd_residue_off,
            print_fn=print_fn)

    if isinstance(fit_src, PoleReader):
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
    band whose weight 1−f clears it at weight 1−f.  Nothing is clipped and MP
    overshoot (f<0 or f>1) rides through unchanged
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
    quadrature_reduction_seconds,
    quadrature_cache_dir,
    omega_grid_step_ry,
    quadrature_reduction_steps=DynamicSigmaConfig.quadrature_reduction_steps,
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
    sigma_w_model="mpa",
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
    """
    if sigma_w_model not in ("mpa", "shared_pole"):
        raise ValueError(f"sigma_w_model must be mpa or shared_pole; got {sigma_w_model!r}")
    shared_pole = sigma_w_model == "shared_pole"
    if shared_pole:
        from file_io.shared_pole_store import validate_shared_pole_model
        from file_io.slab_io import SlabIO
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
        n_poles = int(ledger["n_q_irr"])
        ordered_residues = False
        with timing.section("sigma.capacity"):
            schedule = _shared_pole_memory_schedule(meta, ledger, mesh_xy=mesh_xy)
        print_fn(f"  shared-pole Sigma capacity: {schedule}")
    else:
        ledger = validate_fit_store(
            fit_src, expected_identity=fit_identity,
            expected_screening_diagrams=expected_screening_diagrams)
        n_poles = int(ledger["n_p"])
        ordered_residues = bool(ledger["ordered_residues"])
    odd_residue_off = _resolve_mpa_odd_residue_debug(
        ordered_residues, print_fn=print_fn)
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    with timing.section("sigma.branches"):
        branches = (_branches(
            wfns, omega_grid_ry, efermi_ry,
            occupation_state=occupation_state,
            occupation_window_threshold=occupation_window_threshold)
            if sigma_branches is None else tuple(sigma_branches))
    plan_mode = resolve_sigma_plan()
    if shared_pole and plan_mode != "box":
        raise ValueError("shared-pole Sigma requires the production box planner")
    # ONE collective handle for the census walk, the planner, and the
    # executor walk — the whole Σ stage of this iteration.  The reader
    # does its h5py reads (ledger, unfold tables) before that handle
    # exists and none after, so no serial-h5py open on this store
    # overlaps or interleaves with the FFI one anywhere inside a Σ stage
    # (audit A1; hdf5_owner enforces it).  The context manager is the
    # release path: a refusal from the planner or the executor must still
    # close the handle on every rank.
    with (SlabIO(fit_src, mode="r", mesh=mesh_xy) if shared_pole else
          open_pole_reader(fit_src, mesh_xy=mesh_xy)) as reader:
        # One bounded extrema census serves both routes.  In particular, the
        # production route does not read residues into a host histogram and
        # never constructs a sampled state-pole lattice.
        summaries = []
        if shared_pole:
            from file_io.shared_pole_store import read_shared_pole_census
            with timing.section("sigma.census"):
                poles_device, counts_device = read_shared_pole_census(
                    reader, header=ledger, capacity=meta.shared_pole_capacity)
                poles2, counts = map(np.asarray, jax.device_get((poles_device, counts_device)))
                del poles_device, counts_device
                quadrature_cache_dir = sigma_rule_request_cache(
                    quadrature_cache_dir, ledger["identity"], poles2, counts,
                    eta=regularization_width_ry, eps=quadrature_eps)
                frequencies = shared_pole_frequencies(poles2, counts)
                summaries = summarize_shared_poles(
                    poles2, counts, branches,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor,
                    occupation_window_threshold=occupation_window_threshold)
        for lo in (() if shared_pole else range(0, n_poles, int(pole_batch_size))):
            hi = min(lo + int(pole_batch_size), n_poles)
            Omega, B, B_odd = reader.read(
                slice(lo, hi), unfold=True, return_sharded=True,
                to_unit="Ry", include_odd=True)
            _refuse_nonfinite_pole_slab(lo, Omega, B, B_odd)
            if B_odd is not None and odd_residue_off:
                B_odd = jnp.zeros_like(B_odd)
            summaries.extend(summarize_sigma_poles(
                Omega, _geometry_residue(B, B_odd), branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor, pole_offset=lo,
                occupation_window_threshold=occupation_window_threshold))
            del Omega, B, B_odd
            gc.collect()
        if plan_mode == "panes":
            plan, geometry = build_shared_sigma_windows(
                summaries, branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor,
                target_error=_PANE_CONTROL_TARGET_ERROR,
                max_rank=_PANE_CONTROL_MAX_RANK,
                crossing_max_nodes=max(
                    CROSSING_NODE_FLOOR, _PANE_CONTROL_MAX_RANK),
                omega_grid_step_ry=omega_grid_step_ry,
                occupation_window_threshold=occupation_window_threshold)
        else:
            # Rule fitting is its own timing row: on the Si b80/c504 deck the
            # cold fits took ~180 s of a 194 s "Sigma" stage while the tau
            # sweep took 6 s (2026-09-03, runs/DEV/122), and the table
            # could not tell them apart.
            with timing.section("sigma.rule_plan"):
                plan, geometry = plan_sigma_windows(
                    summaries, branches, omega_grid_ry,
                    regularization_width_ry,
                    eps=quadrature_eps,
                    reduction_seconds=quadrature_reduction_seconds,
                    reduction_steps=quadrature_reduction_steps,
                    cache_dir=quadrature_cache_dir,
                    print_fn=print_fn, edge_factor=edge_factor,
                    fixed_rule_session=fixed_quadrature_session)
        if plan_mode == "panes":
            print_fn(
                f"  MPA windows: eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
                f"{geometry['n_windows']} logical windows")
        else:
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
                print_fn(
                    "  SC fixed quadrature: "
                    f"iteration={geometry['sc_fixed_iteration']}, "
                    f"initialized={geometry['sc_fixed_initialized']}, "
                    f"rebuilds_this_iteration="
                    f"{geometry['sc_fixed_rebuilds_this_iteration']}, "
                    f"rebuilds_total="
                    f"{geometry['sc_fixed_total_rebuild_count']}, "
                    f"pair_cost={geometry['window_tau_pairs']}, "
                    f"initial_pair_cost="
                    f"{geometry['sc_fixed_initial_window_tau_pairs']}, "
                    f"state_pad={geometry['sc_state_edge_padding_ev']:.1f} eV, "
                    f"pole_pad="
                    f"{100.0 * geometry['sc_pole_extent_padding_fraction']:.1f}%")
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
                    synthesis = _shared_pole_w_synthesis(
                        reader, meta, ledger, frequencies, schedule, mesh_xy=mesh_xy)
                total = _integrate_sigma_batches(
                    wfns, ((0, None, None, None),), n_poles, plan,
                    omega_grid_ry, meta, mesh_xy, pole_batch_size=n_poles,
                    brackets=band_brackets, band_counts=band_counts,
                    w_synthesis=synthesis, print_fn=print_fn)
                del synthesis
            else:
                total = integrate_sigma_store(
                    wfns, reader, n_poles, plan, omega_grid_ry, meta, mesh_xy,
                    pole_batch_size=pole_batch_size, brackets=band_brackets,
                    band_counts=band_counts, odd_residue_off=odd_residue_off,
                    print_fn=print_fn)
        if not ordered_residues:
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
