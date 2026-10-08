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

from gw import comm_model
from runtime.padding import padded_axis
from runtime.tiles import tile_units


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


def loader_band_chunk(*, nb: int, nk: int, ns: int, ngkmax: int, n_rmu: int,
                      mesh_xy, p_band: int, floor: int) -> int:
    """The ψ loader's band tile off the ζ-fit plan (ζ reuse, current faces), from the run budget.

    Per band of the tile, per rank: the streamed G-flat rows and the tile's
    centroid samples, ``(k_tile/p_band)·ns·16·(ngkmax + n_rmu)``, and its X/Y
    faces ``k_tile·ns·16·(μ_x + μ_y)``, with ``k_tile`` from
    :func:`centroid_fft_tile_geometry`.  The tile count is the fewest whose
    tile fits the fixed tile (``runtime.tiles``); the tile is ``nb`` over that
    count, rounded up to ``p_band`` (least band padding), and never below
    ``floor`` (the automatic chunk, ``gw_config.AUTOMATIC_BAND_CHUNK_SIZE``).
    """
    from runtime.tiles import tile_units
    nb, p_band = int(nb), int(p_band)
    k_tile, _ = centroid_fft_tile_geometry(nk=int(nk), band_chunk=max(int(floor), 1),
                                           p_band=p_band)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    mu_x, mu_y = -(-int(n_rmu) // px), -(-int(n_rmu) // py)
    per_band = (k_tile / p_band * int(ns) * 16.0 * (int(ngkmax) + int(n_rmu))
                + k_tile * int(ns) * 16.0 * (mu_x + mu_y))
    fit = tile_units(per_band, nb)
    n_tiles = -(-nb // max(fit, 1))
    tile = padded_axis(-(-nb // n_tiles), p_band, name="psi loader band tile").carrier
    return int(min(max(tile, int(floor)), padded_axis(nb, p_band, name="psi loader band extent").carrier))


# ---------------------------------------------------------------------------
# The μ-batch ζ fit (docs/architecture/zeta_fit_mubatch.md)
# ---------------------------------------------------------------------------

#: Host RAM a Z store may claim, as a fraction of the node's live MemAvailable
#: (``common.gpu_utils.host_bytes_per_process``), measured at plan time, after
#: the ψ read's staging is released.
_ZSTORE_HOST_FRAC = 0.8


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
    hwm_bytes: float
    budget_bytes: float
    target_bytes: float
    breakdown: dict
    transfer: dict
    green_tile_bytes: float = 0.0   # nk·ns²·μ²·16/P: the GW run's unit
    min_config_bytes: float = 0.0   # the smallest configuration's HWM
    collectives_per_batch: int = 0
    min_call_bytes: float = 0.0     # smallest per-rank collective payload
    min_efficient_bytes: float = 0.0  # gw.comm_model.min_efficient_payload (diagnostic)
    store_bytes: float = 0.0        # Z store per rank
    host_budget_bytes: float = 0.0  # host share per rank at plan time
    n_vertex: int = 1               # channels sharing the loop (1 charge, 3 currents)
    # b -> raw parents per stage-0 chunk of a batch of b slots
    p_chunk: object = dataclasses.field(default=None, repr=False, compare=False)
    # (b_src, n_pg, c_out, n_blk) -> the batch's working-set bytes: whole-
    # orbit source rows b_src, owner plane stage c_out rows x n_blk blocks
    # (route_g_plane_chunk)
    working_set: object = dataclasses.field(default=None, repr=False, compare=False)
    n_planes: int = 0               # planes along the fit axis (n_a)
    min_c: int = 1                  # widest centroid orbit: the smallest owner bin
    c_out: int = 0                  # owner plane stage of the planned batch:
    n_blk: int = 1                  # c_out rows at a time, n_blk plane blocks

    def format(self) -> str:
        gt = max(self.green_tile_bytes, 1.0)
        lines = [
            "  ISDF μ-batch plan (one budget; docs/architecture/zeta_fit_mubatch.md)",
            f"    fit route     = {self.route} (source {self.source}, "
            f"band chunk {self.band_chunk}, k chunk {self.k_chunk})",
            f"    μ batch       = {self.b}  (whole orbits per owner, at least "
            f"{self.min_c} each; {self.n_batch} batches, "
            f"{self.collectives_per_batch} collectives each, smallest "
            f"{self.min_call_bytes / 1e6:.1f} MB/rank (efficient >= "
            f"{self.min_efficient_bytes / 1e6:.1f}))",
            f"    r sub-block   = {self.r_sub} planes per group; owner plane stage "
            f"{self.c_out} rows at a time, {self.n_blk} plane block(s)",
            f"    ζ back-solve  = whole-tile factor on its q owners, applied on each "
            f"G tile (`linalg` sets the other stages)",
            f"    channels      = {self.n_vertex} (one k-convolution, accumulator and "
            f"Z store each; every other stage shared)",
            f"    Z store       = {self.placement} ({self.store_bytes / 1e9:.1f} GB/rank; "
            f"host share {self.host_budget_bytes / 1e9:.1f} GB/rank from MemAvailable), "
            f"G-vector tile {self.g_tile}, q-local finalize",
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
                      budget_gb: float,
                      psi_face_bytes: float = 0.0, n_vertex: int = 1,
                      n_parent: int | None = None, orbit_width: int = 1) -> MuBatchPlan:
    """Size the route-G μ-batch fit (docs/architecture/zeta_fit_mubatch.md).

    Per rank: conj ψ(G) on its G slice (full zone), the owner's pair
    projectors D̃ (formed and sent one parent chunk at a time, ``p_chunk``),
    and on the owner one plane group of ``D(k, μ, r)`` for its ``c = b/P``
    centroids.
    ``n_vertex`` channels (the three bispinor currents) share every stage but
    the k-convolution and the accumulate, so their Z rows, ζ-cylinder
    accumulators, factors and stores are priced ``n_vertex`` times.  The
    batch ``b`` fills the budget at one plane per group; the parent chunk
    and the plane-group width ``n_pg`` are the most units whose bytes fit
    the fixed tile (``runtime.tiles``): a parent's X_B rows, and a plane's
    share of the owner plane stage for one row.  No timing
    model chooses a shape.  ψ(G), X_B and the pair projectors are priced over the
    ``n_parent`` raw parents the kernel holds (default the full zone).
    Every owner holds whole centroid orbits
    (:func:`isdf.zeta_mubatch.owner_orbit_batches`), so ``orbit_width``, the
    widest orbit the packing keeps together
    (:func:`gw.centroid_k_unfold.widest_unfold_orbit`), is the smallest owner
    bin: ``b ≥ P·c_orb``.  The
    smallest configuration is ``b = P·c_orb`` with one plane per group and
    per block; the planned batch keeps the whole plane axis in one block when
    it fits, else its owner plane stage streams rows and plane blocks
    (:func:`_plane_stage`, the rule :func:`route_g_plane_chunk` applies in
    the fit).  The HWM is that configuration's working set; over the target,
    one warning line is printed here, before anything is compiled, and the
    plan runs.
    """
    from runtime.padding import mesh_divisor
    P_ = int(mesh_divisor(mesh_xy))
    nk, ns = int(meta.nk_tot), int(meta.nspinor)
    n_p = nk if n_parent is None else int(n_parent)
    mu = int(getattr(meta, "n_rmu_padded", None) or meta.n_rmu)
    c_orb = max(1, int(orbit_width))
    b_min = P_ * c_orb
    fft_grid = tuple(int(v) for v in meta.fft_grid)
    n_rtot = int(math.prod(fft_grid))
    n_a = max(fft_grid)
    ps = n_rtot // n_a
    Q, N_G = int(n_q_selected), int(ngkmax)
    Q_pad = math.ceil(Q / P_) * P_
    Q_loc = Q_pad // P_
    Gp = math.ceil(int(psi_ngkmax) / P_)
    nb = padded_axis(int(fit_nb), P_, name="route-G fit bands").carrier
    budget = target = float(budget_gb) * 1e9
    n_v = int(n_vertex)
    base = {
        "C factor": n_v * _c128(Q_loc, mu, mu),
        "centroid faces": float(psi_face_bytes),
        "conj ψ(G) slice": _c128(n_p, nb, ns, Gp),
        "sphere tables": 12.0 * nk * Gp + 4.0 * nk * n_col * n_s + 8.0 * Q * N_G,
    }
    base_total = sum(base.values())

    r_zeta = (3.0 * N_G / (4.0 * math.pi)) ** (1.0 / 3.0)
    n_zc = min(ps, math.ceil(1.3 * math.pi * r_zeta * r_zeta))
    n_za = min(n_a, math.ceil(2 * r_zeta) + 1)

    def stages(b, n_pg, c_out=None, n_blk=1):
        """The batch's live sets per stage; the working set is their max
        (XLA frees each stage's inputs before the next; measured VI3 P16:
        28.2 GB peak against 53.4 GB for the old sum of all terms).  The
        source rows (X_B, pair projectors, Z rows) are the batch's ``b``; the
        owner's plane stage streams ``c_out`` rows at a time (default b/P)
        and ``n_blk`` blocks of the plane axis."""
        c, r_pl = b // P_, n_pg * ps
        co = c if c_out is None else int(c_out)
        n_ap = math.ceil(math.ceil(n_a / n_pg) / int(n_blk)) * n_pg   # planes per block
        rows = {"Z rows (+1 lookahead)": 2 * n_v * _c128(Q, c, N_G)}
        d_g = 2 * _c128(n_p, ns, b, ns, Gp)                 # D~ L+R, one copy
        n_pc = p_chunk(b)
        # One chunk is the pad of the all-to-all output, so the owner's D~ is
        # a separate array only from two.
        return [
            dict(rows, **{"pair projectors (owner)": d_g if n_pc < n_p else 0.0,
                          "parent chunk in flight": in_flight(n_pc, b)}),
            dict(rows, **{"pair projectors (owner)": d_g,
                          "D cylinder (plane block)": _c128(nk, n_ap, ns, 2 * co, ns, n_col)
                          + _c128(ns, 2 * co, ns, n_col, n_s)}),
            dict(rows, **{"D cylinder (plane block)": _c128(nk, n_ap, ns, 2 * co, ns, n_col),
                          # streamed chunks keep the owner's source rows live
                          "pair projectors (owner)": d_g if co < c or n_blk > 1 else 0.0,
                          "plane group": 2 * _c128(nk, n_pg, ns, 2 * co, ns, ps),
                          "k-conv + Z": 9 * _c128(nk, co, r_pl) + 3 * n_v * _c128(Q, co, r_pl),
                          "ζ cylinder accumulator": n_v * _c128(Q, co, n_zc, n_za)}),
        ]

    def in_flight(n_pc, b):
        """A stage-0 chunk of ``n_pc`` raw parents in flight: at its GEMM, X_B,
        its psum and the two weighted copies, the phases, the ψ slice and its
        conj beside the GEMM output; at its all-to-all, its D~ rows in and out."""
        x_c = 3 * _c128(n_pc, nb, ns, b) + _c128(n_pc, Gp, b) + 2 * _c128(n_pc, nb, ns, Gp)
        d_c = 2 * _c128(n_pc, ns, b, ns, Gp)
        return max(x_c + d_c, 2 * d_c)

    def p_chunk(b):
        """Raw parents per stage-0 chunk: the most whose X_B rows fit one tile
        (runtime.tiles), balanced over the parents."""
        return _balanced(n_p, tile_units(_c128(1, nb, ns, b), n_p))

    def ws(b, n_pg, c_out=None, n_blk=1):
        return max(stages(b, n_pg, c_out, n_blk), key=lambda d: sum(d.values()))

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

    def working_set(b_src, n_pg_, c_out, n_blk):
        return base_total + psi_bytes + sum(ws(b_src, n_pg_, c_out, n_blk).values())

    green = _c128(nk, ns * ns, mu, mu, shard=P_)
    # The smallest route-G configuration: ψ(G) resident (its streaming in
    # band-chunk buffers is not implemented), one widest orbit per owner, one
    # plane per block.
    need_min = working_set(b_min, 1, 1, n_a)
    b_top = math.ceil(mu / P_) * P_
    # The batch, at one plane per group (the smallest plane stage): every byte
    # the fixed terms and ψ(G) leave buys centroids at c_μ bytes each, with
    # the whole plane axis in one block; with no such b, b = P·c_orb and the
    # owner plane stage streams rows and plane blocks.
    c_mu = (batch_bytes(2 * P_, 1) - batch_bytes(P_, 1)) / P_
    room = M_f - psi_bytes - (batch_bytes(P_, 1) - c_mu * P_)
    b = b_min
    if room >= c_mu * b_min:
        b = min(b_top, int(room // c_mu) // P_ * P_)
        b = max(b_min, math.ceil(math.ceil(mu / math.ceil(mu / b)) / P_) * P_)   # balance
    # The plane-group width: the most planes whose share of the owner plane
    # stage for one row (plane group, k-convolution and Z rows) fits one tile,
    # balanced over the plane axis; the rows per chunk then fill the budget.
    plane = stages(b, 1, 1, 1)[2]
    n_pg = _balanced(n_a, tile_units(plane["plane group"] + plane["k-conv + Z"], n_a))
    c_out, n_blk, fits = _plane_stage(working_set, target, c_plan=b // P_, c_src=b // P_,
                                      n_pg=n_pg, n_planes=n_a, n_ranks=P_)
    base["conj ψ(G) slice (resident)"] = psi_bytes
    # The all-to-all floor: every pair projector crosses the network once.
    t_a2a_floor = mu * 2 * _c128(n_p, ns, 1, ns, Gp) / comm_model.BETA_BPS
    n_batch = math.ceil(mu / b)
    n_pc = p_chunk(b)
    n_ch0 = -(-n_p // n_pc)
    per_g = 6.0 * _c128(Q_pad, mu, 1, shard=P_)
    g_tile = int(max(P_, (0.25 * target // max(per_g, 1.0)) // P_ * P_))
    g_tile = min(g_tile, math.ceil(N_G / P_) * P_)
    n_Gt = math.ceil(N_G / g_tile)
    store = n_v * _c128(Q, n_batch * b, n_Gt * g_tile, shard=P_)
    from common.gpu_utils import host_bytes_per_process
    host_budget = host_bytes_per_process(_ZSTORE_HOST_FRAC)
    placement = 'host' if store <= host_budget else 'disk'
    br = dict(base)
    br.update(ws(b, n_pg, c_out, n_blk))
    hwm = sum(br.values())
    if not fits:
        from common.gpu_utils import warn_over_budget
        warn_over_budget(f"zeta mu-batch ({b // P_} centroids per owner, whole orbits of up "
                         f"to {c_orb}, {c_out} row(s) x {n_blk} plane block(s))", hwm, target)
    store_total = n_v * Q * mu * N_G * 16.0
    transfer = {
        f"pair-projector all-to-all (floor {t_a2a_floor:.0f} s)":
            mu * 2 * _c128(n_p, ns, 1, ns, Gp),
        f"X_B psum ({_c128(n_p, nb, ns, b) / 1e6:.0f} MB per batch, "
        f"{n_ch0} parent chunk(s))":
            mu * _c128(n_p, nb, ns, 1),
        "Z store write": store_total / P_, "Z store read": store_total / P_,
    }
    return MuBatchPlan(
        green_tile_bytes=float(green), min_config_bytes=float(need_min),
        collectives_per_batch=2 * n_ch0,
        store_bytes=float(store), host_budget_bytes=float(host_budget),
        min_call_bytes=float(min(_c128(n_pc, nb, ns, b), 2 * _c128(n_pc, ns, b, ns, Gp))),
        min_efficient_bytes=comm_model.min_efficient_payload(P_ - 1),
        n_vertex=n_v, route='G', source='resident', band_chunk=int(nb), k_chunk=int(nk),
        working_set=working_set, p_chunk=p_chunk, min_c=int(c_orb), c_out=int(c_out), n_blk=int(n_blk),
        b=int(b), n_batch=int(n_batch), r_sub=int(n_pg), row_chunk=0, n_planes=int(n_a),
        g_tile=int(g_tile), placement=placement,
        hwm_bytes=float(hwm), budget_bytes=float(budget),
        target_bytes=float(target), breakdown=br, transfer=transfer)


def _balanced(n: int, fit: int) -> int:
    """Units per chunk when ``n`` units go in chunks of at most ``fit``: the
    fewest chunks, balanced (``ceil(n / ceil(n / fit))``)."""
    return -(-int(n) // -(-int(n) // max(int(fit), 1)))


def _plane_stage(working_set, target, *, c_plan: int, c_src: int, n_pg: int,
                 n_planes: int, n_ranks: int) -> tuple[int, int, bool]:
    """The owner's plane stage ``(c_out, n_blk, fits)`` for bins of ``c_src`` rows.

    ``c_out`` is the widest balanced chunk of the ``c_src`` rows, at most
    ``c_plan``, whose working set fits the cap with the whole plane axis in
    one block; when even one row does not fit, the plane axis is cut into the
    fewest blocks that do (each block redoes the unfold).  The cap is
    ``target``, or the smallest split's working set when that is larger: the
    source stage (the owner's D̃ and one parent chunk) does not depend on
    the split, so a split under the bytes it already holds costs nothing.
    ``fits`` says whether the returned split is within ``target``.
    """
    P_, c_src = int(n_ranks), int(c_src)
    c_plan = max(1, int(c_plan))
    n_grp = -(-int(n_planes) // int(n_pg))
    at = lambda c_out, n_blk: working_set(P_ * c_src, n_pg, c_out, n_blk)
    cap = max(float(target), at(1, n_grp))
    for n_ch in range(-(-c_src // c_plan), c_src + 1):
        if at(-(-c_src // n_ch), 1) <= cap:
            return -(-c_src // n_ch), 1, at(-(-c_src // n_ch), 1) <= target
    n_blk = next((n for n in range(2, n_grp) if at(1, n) <= cap), n_grp)
    return 1, n_blk, at(1, n_blk) <= target


def route_g_plane_chunk(plan: MuBatchPlan, c_src: int, n_ranks: int) -> tuple[int, int]:
    """The owner's plane stage ``(c_out, n_blk)`` for the packed whole-orbit bins of ``c_src`` rows.

    The fit packs whole centroid orbits per owner
    (:func:`isdf.zeta_mubatch.best_owner_orbit_batches`) into bins of at most
    the planned ``c = b/P``, which the planner floors at the widest orbit.
    The owner streams its rows through the planes in balanced chunks of
    ``c_out ≤ c`` (:func:`_plane_stage`), so the D cylinder, plane group and
    k-convolution stay within its cap (the target, or the bytes the source
    stage already holds); the source rows (X_B, pair projectors, Z rows) are
    priced at ``P·c_src``.  A split that does not fit
    was already announced by the planner at its own (wider) batch; a packed
    bin wider than the plan's, which the plan did not price, prints one
    warning line here.
    """
    P_ = int(n_ranks)
    c_plan = max(1, int(plan.b) // P_)
    c_out, n_blk, fits = _plane_stage(
        plan.working_set, plan.target_bytes, c_plan=c_plan, c_src=c_src,
        n_pg=int(plan.r_sub), n_planes=int(plan.n_planes), n_ranks=P_)
    if not fits and int(c_src) > c_plan:
        from common.gpu_utils import warn_over_budget
        warn_over_budget(f"zeta mu-batch orbit bins ({int(c_src)} centroids per owner)",
                         plan.working_set(P_ * int(c_src), int(plan.r_sub), c_out, n_blk),
                         plan.target_bytes)
    return c_out, n_blk
