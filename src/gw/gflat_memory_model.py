"""The route-G ζ-fit planner and the centroid-load tile rule.

Every ζ channel (charge and the three current channels) runs route G:
``Z_q(μ, G)`` formed a batch of centroids at a time, the whole-tile factor
applied on each G tile.  :func:`plan_zeta_route_g` sizes the batch from the
device budget beside the resident raw-parent ψ carrier and places the Z store;
docs/architecture/zeta_fit_mubatch.md owns the algorithm and its byte table,
docs/architecture/memory-model.md the per-rank inventory.
:func:`centroid_fft_tile_geometry` bounds the k tile of a centroid ψ load,
which the artifact-reuse loaders use without planning a fit.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

from common.gpu_utils import bfc_fragmentation_target_utilization
from gw import comm_model
from runtime.padding import padded_axis


_C128 = 16  # bytes per complex128


def _c128(*dims, shard: int = 1) -> float:
    """Per-rank complex128 bytes for ``dims`` sharded over ``shard`` ranks."""
    n = 1
    for d in dims:
        n *= int(d)
    return _C128 * n / max(int(shard), 1)


def centroid_fft_tile_geometry(
    *, nk: int, band_chunk: int, p_band: int,
) -> tuple[int, int]:
    """Return ``(k_tile, local_flat_rows)`` for a centroid WFN transfer.

    The loader mesh-rounds its global band tile before distributing that
    axis.  Bound ``k_tile * (band_tile / p_band)`` by that same physical band
    tile, so the local FFT-row batch does not grow just because the band axis
    is divided over more ranks.  This pure rule also serves artifact-reuse
    paths, which need a safe centroid resample but deliberately do not run the
    full zeta-fit memory planner.
    """
    nk = int(nk)
    band_chunk = int(band_chunk)
    p_band = int(p_band)
    if nk <= 0 or band_chunk <= 0 or p_band <= 0:
        raise ValueError(
            "centroid FFT geometry requires positive nk, band_chunk, and "
            f"p_band; got nk={nk}, band_chunk={band_chunk}, p_band={p_band}")
    from runtime.padding import padded_axis
    band_tile = padded_axis(
        band_chunk, p_band, name="centroid FFT band tile").carrier
    local_bands = band_tile // p_band
    k_tile = min(nk, max(1, band_tile // local_bands))
    return k_tile, k_tile * local_bands


# ---------------------------------------------------------------------------
# The μ-batch ζ fit (docs/architecture/zeta_fit_mubatch.md)
# ---------------------------------------------------------------------------

#: Host RAM a Z store may claim, as a fraction of the node's live MemAvailable.
_ZSTORE_HOST_FRAC = 0.8


def _host_bytes_per_rank() -> float:
    """Host budget for the Z store per rank: 0.8 of the node's live
    ``MemAvailable`` over the processes sharing the node, the minimum over
    processes (rank-invariant).  Measured at plan time, after nothing large
    is held on the host (the ψ read's staging is released before the fit)."""
    import socket
    import zlib
    import numpy as _np
    from common.collectives import all_gather_processes
    from common.gpu_utils import get_host_memory_available_gb, minimum_process_budget_gb
    avail_gb = get_host_memory_available_gb()
    host = zlib.crc32(socket.gethostname().encode())
    hosts = _np.asarray(all_gather_processes(_np.asarray(host, dtype=_np.int64)))
    per_node = max(1, int(_np.sum(hosts == host)))
    local_gb = 0.0 if avail_gb is None else _ZSTORE_HOST_FRAC * avail_gb / per_node
    return float(minimum_process_budget_gb(min(local_gb, 1e12))) * 1e9


@dataclasses.dataclass(frozen=True)
class MuBatchPlan:
    """Resolved μ-batch fit plan: every size comes from the one budget."""
    route: str                 # 'cache' (flat r blocks) | 'planes'
    source: str                # 'cache' | 'resident' | 'host'
    band_chunk: int            # ψ band chunk (cache build / plane regeneration)
    k_chunk: int               # parents regenerated per step (plane route)
    b: int                     # μ batch (multiple of P)
    n_batch: int
    r_sub: int                 # points (cache) or planes (planes) per sub-block
    row_chunk: int             # FFT rows per scan step
    g_tile: int                # G slots per store tile (multiple of P)
    placement: str             # 'host' | 'disk' (never the device)
    finalize_layout: str       # 'q' (q-local solve) | 'g' (G-split)
    hwm_bytes: float
    budget_bytes: float
    target_bytes: float
    breakdown: dict
    transfer: dict
    green_tile_bytes: float = 0.0   # nk·ns²·μ²·16/P: the GW run's unit
    min_config_bytes: float = 0.0   # the smallest configuration's HWM
    collectives_per_batch: int = 0
    t_model_s: float = 0.0          # modelled loop time (ψ traffic + calls)
    min_call_bytes: float = 0.0     # smallest per-rank collective payload
    min_efficient_bytes: float = 0.0  # gw.comm_model.min_efficient_payload
    runner_up: str | None = None
    store_bytes: float = 0.0        # Z store per rank
    host_budget_bytes: float = 0.0  # host share per rank at plan time
    zeta_tier: str = "local"        # whole-tile back-solve tier (route G's)
    n_vertex: int = 1               # channels sharing the loop (1 charge, 3 currents)
    # (b_src, n_pg, c_out) -> the batch's working-set bytes: whole-orbit
    # source rows b_src, owner plane stage c_out (route_g_plane_chunk)
    working_set: object = dataclasses.field(default=None, repr=False, compare=False)

    def format(self) -> str:
        gt = max(self.green_tile_bytes, 1.0)
        lines = [
            "  ISDF μ-batch plan (one budget; docs/architecture/zeta_fit_mubatch.md)",
            f"    fit route     = {self.route} (source {self.source}, "
            f"band chunk {self.band_chunk}, k chunk {self.k_chunk})",
            f"    μ batch       = {self.b}  ({self.n_batch} batches, "
            f"{self.collectives_per_batch} collectives each, smallest "
            f"{self.min_call_bytes / 1e6:.1f} MB/rank (efficient >= "
            f"{self.min_efficient_bytes / 1e6:.1f}); modelled loop "
            f"{self.t_model_s:.0f} s; runner-up {self.runner_up})",
            f"    r sub-block   = {self.r_sub} "
            f"{'points' if self.route == 'cache' else 'planes per group' if self.route == 'G' else 'plane(s)'}",
            f"    ζ tier        = {self.zeta_tier} (chosen by route G, which applies "
            f"the whole-tile factor on each G tile; `linalg` sets the other stages)",
            f"    channels      = {self.n_vertex} (one k-convolution, accumulator and "
            f"Z store each; every other stage shared)",
            f"    Z store       = {self.placement} ({self.store_bytes / 1e9:.1f} GB/rank; "
            f"host share {self.host_budget_bytes / 1e9:.1f} GB/rank from MemAvailable), "
            f"G-vector tile {self.g_tile}, finalize {self.finalize_layout}-layout",
            f"    G_tile unit   = {gt / 1e9:.3f} GB/dev (nk·ns²·μ²·16/P); "
            f"feasibility ceiling 4·G_tile = {4 * gt / 1e9:.2f}",
            f"    minimum cfg   = {self.min_config_bytes / 1e9:.2f} GB/dev "
            f"({self.min_config_bytes / gt:.2f} G_tile, "
            f"{'within' if self.min_config_bytes <= 4 * gt else 'OVER'} the ceiling)",
            f"    target        = {self.target_bytes / 1e9:.2f} GB/dev of "
            f"{self.budget_bytes / 1e9:.2f}",
            f"    HWM estimate  = {self.hwm_bytes / 1e9:.2f} GB/dev "
            f"({self.hwm_bytes / gt:.2f} G_tile)",
            "    terms (GB/dev, G_tile):",
        ]
        for k, v in sorted(self.breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {k:.<22s} {v / 1e9:>8.3f}  {v / gt:>6.2f}")
        lines.append("    whole-fit volumes (GB/rank): " + ", ".join(
            f"{k} {v / 1e9:.1f}" for k, v in self.transfer.items()))
        return "\n".join(lines)


def plan_zeta_route_g(*, meta, mesh_xy, n_q_selected: int, ngkmax: int,
                      psi_ngkmax: int, fit_nb: int, n_col: int, n_s: int,
                      zeta_tier: str, budget_gb: float,
                      target_utilization: float | None = None,
                      psi_face_bytes: float = 0.0, n_vertex: int = 1,
                      n_parent: int | None = None) -> MuBatchPlan:
    """Size the route-G μ-batch fit (docs/architecture/zeta_fit_mubatch.md).

    Per rank: conj ψ(G) on its G slice (full zone), the batch's pair
    projectors on the slice and after the one all-to-all, and on the owner
    one plane group of ``D(k, μ, r)`` for its ``c = b/P`` centroids.
    ``n_vertex`` channels (the three bispinor currents) share every stage but
    the k-convolution and the accumulate, so their Z rows, ζ-cylinder
    accumulators, factors and stores are priced ``n_vertex`` times.  The
    candidates are the plane-group width ``n_pg`` (each at its largest
    batch); the modelled time counts the per-batch fixed cost and the
    cylinder gathers (``∝ 1/n_pg``) -- the pair-projector all-to-all, the
    X_B psum and the per-centroid arithmetic are the same for every
    candidate.  Feasibility: the smallest configuration (``b = P``,
    ``n_pg = 1``) must fit ``4·G_tile``.  ``n_parent`` (the raw parents the
    kernel holds; default the full zone) prices the source rows in the
    post-packing re-check (:func:`route_g_plane_chunk`); the batch choice
    keeps the full-zone upper bound.
    """
    from runtime.padding import mesh_divisor
    P_ = int(mesh_divisor(mesh_xy))
    nk, ns = int(meta.nk_tot), int(meta.nspinor)
    mu = int(getattr(meta, "n_rmu_padded", None) or meta.n_rmu)
    fft_grid = tuple(int(v) for v in meta.fft_grid)
    n_rtot = int(math.prod(fft_grid))
    n_a = max(fft_grid)
    ps = n_rtot // n_a
    Q, N_G = int(n_q_selected), int(ngkmax)
    Q_pad = math.ceil(Q / P_) * P_
    Q_loc = Q_pad // P_
    Gp = math.ceil(int(psi_ngkmax) / P_)
    nb = padded_axis(int(fit_nb), P_, name="route-G fit bands").carrier
    if target_utilization is None:
        target_utilization = bfc_fragmentation_target_utilization(ns)
    budget = float(budget_gb) * 1e9
    target = budget * float(target_utilization)
    finalize_layout = 'q' if str(zeta_tier) == 'local' else 'g'
    n_v = int(n_vertex)
    base = {
        "C factor": n_v * (_c128(Q_loc, mu, mu) if finalize_layout == 'q'
                           else _c128(Q, mu, mu)),
        "centroid faces": float(psi_face_bytes),
        "conj ψ(G) slice": _c128(nk, nb, ns, Gp),
        "sphere tables": 12.0 * nk * Gp + 4.0 * nk * n_col * n_s + 8.0 * Q * N_G,
    }
    base_total = sum(base.values())

    r_zeta = (3.0 * N_G / (4.0 * math.pi)) ** (1.0 / 3.0)
    n_zc = min(ps, math.ceil(1.3 * math.pi * r_zeta * r_zeta))
    n_za = min(n_a, math.ceil(2 * r_zeta) + 1)

    n_p = nk if n_parent is None else int(n_parent)

    def stages(b, n_pg, c_out=None, n_src=nk):
        """The batch's live sets per stage; the working set is their max
        (XLA frees each stage's inputs before the next; measured VI3 P16:
        28.2 GB peak against 53.4 GB for the old sum of all terms).  The
        source rows (X_B, pair projectors, Z rows) are the batch's ``b``; the
        owner's plane stage streams ``c_out`` rows at a time (default b/P)."""
        c, r_pl = b // P_, n_pg * ps
        co = c if c_out is None else int(c_out)
        n_ap = math.ceil(n_a / n_pg) * n_pg
        rows = {"Z rows (+1 lookahead)": 2 * n_v * _c128(Q, c, N_G)}
        d_g = 2 * _c128(n_src, ns, b, ns, Gp)               # D~ L+R, one copy
        return [
            dict(rows, **{"X_B": 2 * _c128(n_src, nb, ns, b) + _c128(n_src, Gp, b),
                          "pair projectors (GEMM out, all-to-all out)": 2 * d_g}),
            dict(rows, **{"pair projectors (owner)": d_g,
                          "D cylinder (all planes)": _c128(nk, n_ap, ns, 2 * co, ns, n_col)
                          + _c128(ns, 2 * co, ns, n_col, n_s)}),
            dict(rows, **{"D cylinder (all planes)": _c128(nk, n_ap, ns, 2 * co, ns, n_col),
                          # streamed chunks keep the owner's source rows live
                          "pair projectors (owner)": d_g if co < c else 0.0,
                          "plane group": 2 * _c128(nk, n_pg, ns, 2 * co, ns, ps),
                          "k-conv + Z": 9 * _c128(nk, co, r_pl) + 3 * n_v * _c128(Q, co, r_pl),
                          "ζ cylinder accumulator": n_v * _c128(Q, co, n_zc, n_za)}),
        ]

    def ws(b, n_pg, c_out=None, n_src=nk):
        return max(stages(b, n_pg, c_out, n_src), key=lambda d: sum(d.values()))

    # The memory split (owner rule): ψ(G) resident iff what is left after
    # the fixed terms holds it and the smallest batch; then every remaining
    # byte buys centroids at c_μ bytes each (the per-centroid slope of the
    # batch working set).  Partially cached ψ is never optimal (the stream
    # cost ∝ (1-x)/(M_f - xΨ) is monotone in x), so there is no middle tier.
    psi_bytes = base.pop("conj ψ(G) slice")
    base_total = sum(base.values())
    M_f = target - base_total

    def batch_bytes(b, n_pg):
        return sum(ws(b, n_pg).values())

    green = _c128(nk, ns * ns, mu, mu, shard=P_)
    need_min = base_total + psi_bytes + batch_bytes(P_, 1)
    if M_f - psi_bytes < batch_bytes(P_, 1):
        raise ValueError(
            f"GATE zeta-mubatch-capacity: got {need_min / 1e9:.2f} GB/dev for the "
            f"smallest route-G configuration (ψ(G) resident {psi_bytes / 1e9:.2f}, "
            f"b = P = {P_}, one plane), want <= {target / 1e9:.2f} GB/dev; why: "
            "ψ(G) streaming in two band-chunk buffers is not implemented.  Fix: "
            "more ranks or more memory per device.")
    b_top = math.ceil(mu / P_) * P_
    cands = []
    n_pg = 1
    while True:
        c_mu = (batch_bytes(2 * P_, n_pg) - batch_bytes(P_, n_pg)) / P_
        fixed = batch_bytes(P_, n_pg) - c_mu * P_
        room = M_f - psi_bytes - fixed
        if room >= c_mu * P_:
            b = min(b_top, int(room // c_mu) // P_ * P_)
            n_b = math.ceil(mu / b)
            b = math.ceil(math.ceil(mu / n_b) / P_) * P_          # balance
            n_grp = math.ceil(n_a / n_pg)
            # Per batch: the X_B psum and the pair-projector all-to-all
            # (gw.comm_model), n_grp·nk scan steps, and the owner's cylinder
            # gathers.  ponytail: gathers at 1 TB/s and 5 µs per scan step,
            # measured on A100; the per-centroid arithmetic is the same for
            # every candidate and is left out.
            # ponytail: per plane group 3 ms of launches plus the owner's
            # k-conv and plane FFTs at 0.65 s per centroid-grid per batch of
            # c = 1, amortized as c/(c+1) (VI3 P16 A100 measurement); the
            # comm-model service prices the two collectives.
            c_ = b // P_
            t_b = (comm_model.comm_time(_c128(nk, nb, ns, b), P_ - 1)
                   + comm_model.comm_time(2 * _c128(nk, ns, b, ns, Gp), P_ - 1)
                   + 3e-3 * n_grp + 0.65 * (c_ + 1) / 2)
            cands.append((n_b * t_b, n_pg, b))
        if n_pg >= n_a:
            break
        n_pg = min(2 * n_pg, n_a)
    cands.sort()
    t_model, n_pg, b = cands[0]
    ru = (f"n_pg={cands[1][1]} b={cands[1][2]}: {cands[1][0]:.0f} s"
          if len(cands) > 1 else None)
    base["conj ψ(G) slice (resident)"] = psi_bytes
    # The all-to-all floor: every pair projector crosses the network once.
    t_a2a_floor = mu * 2 * _c128(nk, ns, 1, ns, Gp) / comm_model.BETA_BPS
    n_batch = math.ceil(mu / b)
    per_g = (6.0 * _c128(Q_pad, mu, 1, shard=P_)
             + (_c128(Q, mu, mu) if finalize_layout == 'g' else 0.0) / max(N_G, 1))
    g_tile = int(max(P_, (0.25 * target // max(per_g, 1.0)) // P_ * P_))
    g_tile = min(g_tile, math.ceil(N_G / P_) * P_)
    n_Gt = math.ceil(N_G / g_tile)
    store = n_v * _c128(Q, n_batch * b, n_Gt * g_tile, shard=P_)
    host_budget = _host_bytes_per_rank()
    placement = 'host' if store <= host_budget else 'disk'
    br = dict(base)
    br.update(ws(b, n_pg))
    store_total = n_v * Q * mu * N_G * 16.0
    transfer = {
        f"pair-projector all-to-all (floor {t_a2a_floor:.0f} s)":
            mu * 2 * _c128(nk, ns, 1, ns, Gp),
        f"X_B psum ({_c128(nk, nb, ns, b) / 1e6:.0f} MB per batch)":
            mu * _c128(nk, nb, ns, 1),
        "Z store write": store_total / P_, "Z store read": store_total / P_,
    }
    return MuBatchPlan(
        green_tile_bytes=float(green), min_config_bytes=float(need_min),
        collectives_per_batch=2, t_model_s=float(t_model), runner_up=ru,
        store_bytes=float(store), host_budget_bytes=float(host_budget),
        zeta_tier=str(zeta_tier),
        min_call_bytes=float(_c128(nk, nb, ns, b)),
        min_efficient_bytes=comm_model.min_efficient_payload(P_ - 1),
        n_vertex=n_v, route='G', source='resident', band_chunk=int(nb), k_chunk=int(nk),
        working_set=lambda b_src, n_pg_, c_out: (
            base_total + _c128(n_p, nb, ns, Gp) + sum(ws(b_src, n_pg_, c_out, n_p).values())),
        b=int(b), n_batch=int(n_batch), r_sub=int(n_pg), row_chunk=0,
        g_tile=int(g_tile), placement=placement, finalize_layout=finalize_layout,
        hwm_bytes=float(sum(br.values())), budget_bytes=float(budget),
        target_bytes=float(target), breakdown=br, transfer=transfer)


def route_g_plane_chunk(plan: MuBatchPlan, c_src: int, n_ranks: int) -> int:
    """The owner's plane-stage width ``c_out`` for whole-orbit bins of ``c_src`` rows.

    The fit packs whole centroid orbits per owner
    (:func:`isdf.zeta_mubatch.best_owner_orbit_batches`), so the executed
    ``c_src`` is at least the widest orbit and can exceed the planned
    ``c = b/P``.  The owner streams its rows through the planes in balanced
    chunks of ``c_out ≤ c``, so the D cylinder, plane group and k-convolution
    stay at the planned width; the source rows (X_B, pair projectors, Z rows)
    are re-priced at ``P·c_src`` over the raw parents the kernel holds.
    ``c_out`` is the widest balanced chunk that fits the plan's target;
    refuses by name when even one row does not.
    """
    P_ = int(n_ranks)
    c_plan = max(1, int(plan.b) // P_)
    c_src = int(c_src)
    need = float("inf")
    for n_ch in range(-(-c_src // c_plan), c_src + 1):
        c_out = -(-c_src // n_ch)
        need = plan.working_set(P_ * c_src, int(plan.r_sub), c_out)
        if need <= plan.target_bytes:
            return c_out
    raise ValueError(
        f"GATE zeta-mubatch-orbit-capacity: whole-orbit bins of {c_src} centroids "
        f"per owner (planned {c_plan}; the widest orbit sets the floor) need "
        f"{need / 1e9:.2f} GB/dev with the planes streamed one row at a time, want "
        f"<= {plan.target_bytes / 1e9:.2f} GB/dev.  Fix: more memory per device "
        "(the source rows scale with the orbit width, not with P).")
