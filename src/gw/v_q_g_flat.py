"""V_q orchestrator for the G-flat ζ on-disk format.

This is the post-G-flat rewrite of the V_q hot loop.  It replaced the
old r-space tile driver (``gw/v_q_tile.py``, deleted 2026-07-02) when
the on-disk ζ is in WFN.h5-style per-q sphere layout — most of the old
complexity (Case A/B chooser, ``μ × ν`` tiling, in-kernel FFT, shared
sphere conversion) goes away because:

* ``ζ̃`` already lives on the per-q sphere on disk (no FFT here).
* The contract chunks over **G** (a fixed-cost reduction axis), not μ
  / ν — one G-chunk is a small GEMM, and the V[μ,ν] output is the
  whole problem at once.
* One q-TILE per kernel launch: a ``lax.scan`` over the tile's q, and
  inside it a scan over G panels that gathers one ``(μ, g_chunk)`` panel
  per operand per step (SUMMA-style), so no rank ever holds a whole
  ``(μ, n_G)`` face on fewer than all P ranks.

I/O is synchronous q-tiles: every q whose ζ̃ fits the V_q budget is read
in one collective call (all of them whenever they fit — the whole-slab
read), contracted, and only then is the next tile read.  Overlapping a
PHDF5 read with the kernel's NCCL collectives deadlocked historically;
see the tile loop in :func:`_compute_V_q_g_flat_one_tile`.

Math:

    V_q[μ, ν] = Σ_G  conj(ζ̃_{q,μ}(G)) · v(q+G) · ζ̃_{q,ν}(G)
    g0_μ(q)   = ζ̃_{q,μ}(G=0)               # = ζ̃[μ, 0] by sphere convention

The tile builder retains q parents; the scalar consumer unfolds ``V_q``.
The one-leg literal-``G=0`` coefficient unfolds after the tile loop.  The latter must inspect
the parent G table: a star operation can map a nonzero parent G onto the
full-zone literal G=0.  The V_q output sharding ``P(None, 'x', 'y')`` matches.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


if TYPE_CHECKING:                       # pragma: no cover — typing only
    # The DOOR, top-level name only.  No ``ensure_on_path()`` bootstrap is
    # needed here and none is added: this import never executes (the
    # module has ``from __future__ import annotations``, so the annotation
    # it feeds is a string), and a runtime path edit smuggled into a
    # typing-only block is how a "type-checking import" stops being one.
    from file_io.restart_bundle import (open_zeta as ZetaLoader)


# ---------------------------------------------------------------------------
# Inner kernel: one q-tile of ζ (q, μ, G) + v (q, G) → V_q (μ, μ) at P('x','y')
# ---------------------------------------------------------------------------

_Q_TILE_KERNEL_CACHE: dict = {}


def _make_q_tile_kernel(mesh_xy: Mesh, n_rmu_L: int, n_rmu_R: int,
                        ngkmax: int, g_chunk: int, q_tile: int,
                        *, write_g0: bool, same_zeta: bool):
    """Compile-once kernel: contract one q-tile into the (V_acc, g0_acc) buffers.

    Single signature handles both:
      * Charge / diagonal bispinor tiles: ``same_zeta=True``; caller
        passes ``zeta_R_tile is zeta_L_tile`` and the kernel reshards one
        buffer for the two operands of the GEMM.
      * Bispinor off-diagonal tiles: ``same_zeta=False``; caller passes
        two separate slabs (potentially different ``n_rmu_*``).

    Returns ``fn(V_acc, g0_acc, zeta_L_tile, zeta_R_tile, v_tile, q0)
              -> (V_new, g0_new)`` where the tiles are the ``q_tile`` rows
    starting at global q index ``q0``.  Donates the two accumulators so the
    per-q update is in place.

    SUMMA-STYLE PANELS.  ζ_q arrives μ-sharded over every rank,
    ``P(('x','y'), None)``, and stays that way for the whole contraction.
    Each G-chunk step gathers ONE panel per operand — L onto 'x', R onto 'y'
    — multiplies it into the local ``(μ_L/p_x, μ_R/p_y)`` block and drops
    it.  Per rank that is ``ζ_q/P + (n_μL/p_x + n_μR/p_y)·g_chunk`` live
    instead of the ``ζ_q/p_x + ζ_q/p_y`` face gathers this replaced, which
    were O(1/√P) and one 16·μ·n_G/p_x payload per collective.  Total bytes
    moved are unchanged: every G column still reaches each rank once.  The
    G scan runs in ``shard_map`` with the two panel all-gathers written in
    its body; see ``_panel_contract`` for why that is structural.

    THE q LOOP IS A ``lax.scan`` INSIDE THE KERNEL.  The ζ slices are taken
    off the traced tile index, so the caller hands in whole tile slabs and
    no per-q eager slice, per-q dispatch or per-q host sync exists.  The
    donated (V_acc, g0_acc) chain is the scan carry; its dynamic-update-
    slice is in place.  The tile slabs, which every iteration reads, are
    deliberately NOT donated.

    A G tail that ``g_chunk`` does not divide is handled by clamping the last
    chunk's start to ``ngkmax - g_chunk`` and zeroing the weight of the
    columns an earlier chunk already counted, so any ``g_chunk <= ngkmax``
    is exact.  When it divides, the mask is all-true and every GEMM is the
    one the whole-face kernel ran, bit for bit.
    """
    key = (id(mesh_xy), int(n_rmu_L), int(n_rmu_R), int(ngkmax),
           int(g_chunk), int(q_tile), bool(write_g0), bool(same_zeta))
    hit = _Q_TILE_KERNEL_CACHE.get(key)
    if hit is not None:
        return hit

    from common.shard_map import shard_map
    from common.vma import mark_varying

    face_sh = NamedSharding(mesh_xy, P(('x', 'y'), None))
    # The R face in ('y','x') block order: then each 'y' block of μ_R is the
    # tiled all-gather over 'x' of one contiguous run of local blocks, just as
    # each 'x' block of μ_L is the all-gather over 'y' in ('x','y') order.
    # Reaching it is one collective-permute of the ζ_q/P face (the whole-face
    # kernel did the same permute).
    face_yx_sh = NamedSharding(mesh_xy, P(('y', 'x'), None))
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    g0_block_sh = NamedSharding(mesh_xy, P('x'))
    v_sh = NamedSharding(mesh_xy, P(None))
    # The two slabs arrive exactly as ``ZetaLoader.read_zeta_G_slab``
    # returns them — q replicated, μ over ('x','y') — and the v(q+G) rows
    # replicated by ``device_put_process_local``.  The entry sharding is
    # stated rather than inherited: it keeps the per-q slice on the one
    # (μ, G) face instead of on the whole (q_tile, μ, ngkmax) tensor.
    zeta_all_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    v_all_sh = NamedSharding(mesh_xy, P(None, None))

    n_chunks = -(-int(ngkmax) // int(g_chunk))
    last_start = int(ngkmax) - int(g_chunk)
    p_x, p_y = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    def _panel_contract(face_L, face_R, v_q):
        """Per rank: V_q[μ_L/p_x, μ_R/p_y] from one G panel per scan step.

        The panel all-gathers are written inside the scan body, in manual
        (``shard_map``) mode, so no partitioner can hoist them to a whole-face
        gather ahead of the loop — which is exactly what the same loop
        written with ``with_sharding_constraint`` compiled to (measured in the
        CPU 2x2 HLO and as 1.4 s/q on VI3 P16, 2026-09-23).
        """
        col = jnp.arange(g_chunk, dtype=jnp.int32)
        V0 = mark_varying(jnp.zeros((n_rmu_L // p_x, n_rmu_R // p_y),
                                    dtype=face_L.dtype), ('x', 'y'))

        # V[μ,ν] += conj(L_panel) · v · R_panelᵀ, one G panel per step.
        def _g_chunk_body(V_carry, i):
            done = i * g_chunk
            start = jnp.minimum(done, last_start)
            L_panel = jax.lax.all_gather(
                jax.lax.dynamic_slice_in_dim(face_L, start, g_chunk, axis=1),
                'y', axis=0, tiled=True)                # (n_rmu_L/p_x, g_chunk)
            R_panel = jax.lax.all_gather(
                jax.lax.dynamic_slice_in_dim(face_R, start, g_chunk, axis=1),
                'x', axis=0, tiled=True)                # (n_rmu_R/p_y, g_chunk)
            v_chunk = jax.lax.dynamic_slice_in_dim(
                v_q, start, g_chunk, axis=0)            # (g_chunk,)
            v_chunk = jnp.where(start + col >= done, v_chunk,
                                jnp.zeros_like(v_chunk))
            L_w = jnp.conj(L_panel) * v_chunk[None, :]
            return V_carry + L_w @ R_panel.T, None

        V_blk, _ = jax.lax.scan(
            _g_chunk_body, V0, jnp.arange(n_chunks, dtype=jnp.int32),
            unroll=1)
        return V_blk

    panel_contract = shard_map(
        _panel_contract, mesh=mesh_xy,
        in_specs=(P(('x', 'y'), None), P(('y', 'x'), None), P(None)),
        out_specs=P('x', 'y'))

    @partial(jax.jit, donate_argnums=(0, 1))
    def fn(V_acc, g0_acc, zeta_L_tile, zeta_R_tile, v_tile, q0):
        q0_32 = q0.astype(jnp.int32)
        zero32 = jnp.int32(0)

        zeta_L_tile = jax.lax.with_sharding_constraint(zeta_L_tile, zeta_all_sh)
        zeta_R_src = zeta_L_tile if same_zeta else (
            jax.lax.with_sharding_constraint(zeta_R_tile, zeta_all_sh))
        v_tile = jax.lax.with_sharding_constraint(v_tile, v_all_sh)

        def _one_q(carry, j):
            V_acc, g0_acc = carry
            # Drop the size-1 q axis FIRST, then work on the real (μ, G)
            # tensor.  Staging through P(('x','y'), None) on the (1, μ, G)
            # slice — sharding the size-1 q axis — is what XLA cannot
            # reshard to a μ-sharded layout; it fell back to a full
            # replicate-then-repartition ("[SPMD] Involuntary full
            # rematerialization" on the V_q g-flat tensor).  THIS ORDER IS
            # THE POINT — do not fold the [0] into a sharding constraint.
            #
            # ONE ``[0]`` PER SLAB, BOUND TO A NAME (shared by both operands
            # when same_zeta), then an ``optimization_barrier`` on the
            # (μ, ngkmax) face.  The (1, μ, G) slice has a degenerate leading
            # axis, so XLA may prefer a ``{2,0,1}`` minor-to-major for it — a
            # different physical layout for the slab it is sliced from — and
            # answers by copying the WHOLE slab parameter into that layout
            # once per call (measured on the whole-slab kernel: two
            # ``c128[n_q,μ/p,ngkmax] copy`` ops distinct-ζ, one shared-ζ).
            # The barrier is a scheduling fence, not an op.
            face_L = jax.lax.optimization_barrier(
                jax.lax.dynamic_slice_in_dim(zeta_L_tile, j, 1, axis=0)[0])
            face_R = (face_L if same_zeta
                      else jax.lax.optimization_barrier(
                          jax.lax.dynamic_slice_in_dim(
                              zeta_R_src, j, 1, axis=0)[0]))
            face_L = jax.lax.with_sharding_constraint(face_L, face_sh)
            face_R = jax.lax.with_sharding_constraint(face_R, face_yx_sh)
            v_q = jax.lax.with_sharding_constraint(
                jax.lax.dynamic_slice_in_dim(v_tile, j, 1, axis=0)[0], v_sh)

            V_q = panel_contract(face_L, face_R, v_q)   # (μ_L, μ_R) P('x','y')

            q_32 = q0_32 + j
            V_acc = jax.lax.dynamic_update_slice(
                V_acc, V_q[None, :, :], (q_32, zero32, zero32))
            V_acc = jax.lax.with_sharding_constraint(V_acc, V_sh)
            if write_g0:
                g0_q = jax.lax.with_sharding_constraint(
                    face_L[:, 0], g0_block_sh)          # (n_rmu_L/p_x,)
                g0_acc = jax.lax.dynamic_update_slice(
                    g0_acc, g0_q[None, :], (q_32, zero32))
                g0_acc = jax.lax.with_sharding_constraint(g0_acc, g0_sh)
            return (V_acc, g0_acc), None

        (V_new, g0_new), _ = jax.lax.scan(
            _one_q, (V_acc, g0_acc), jnp.arange(q_tile, dtype=jnp.int32),
            unroll=1)
        return V_new, g0_new

    _Q_TILE_KERNEL_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Small shared helpers (used by both the charge wrapper and the bispinor
# tile loop in gw.v_q_bispinor)
# ---------------------------------------------------------------------------

def _resolve_ibz_q_list(*, sym, centroid_indices, kgrid, fft_grid,
                        context="V_q / W q-grid reduction",
                        return_resolution=False, mu_basis=None):
    """Pick IBZ q's via centroid orbit closure, fall back to full BZ.

    Returns ``(q_irr_kgrid_int, q_irr_frac, q_full_to_irr_idx,
    q_full_to_irr_sym, sym_perm, L_table, use_ibz)``.  With
    ``return_resolution=True``, the resolution carrying the closure verdict
    is appended.  When
    ``use_ibz`` is False the *_idx / *_sym / sym_perm / L_table fields
    are None; caller skips the post-loop unfold.

    ``L_table`` is the per-(sym, μ) integer real-space lattice wrap
    captured by ``centroid_source_map_and_wrap``; ``unfold_isdf_operator`` uses
    it
    to build the umklapp phase ``exp(2π i q · (L_μ − L_ν))``.

    THE CLOSURE DECISION IS NOT TAKEN HERE.  It is taken once, in
    ``symmetry_maps.resolve_qgrid_symmetry``, and announced once by
    ``gw.qgrid_symmetry.resolve_qgrid_symmetry_tables``; this function
    consumes the resolution and shapes it into the seven-tuple its three
    callers already read.

    THE ``verbose`` ARGUMENT IS GONE.  It gated exactly one thing — the
    line that said the fallback had happened — and ``gw/screening.py``
    passed ``verbose=False``, which is why the W Dyson solve could drop
    from ``n_q_ibz`` blocks to ``n_q_full`` without a word.  A knob whose
    only effect is to hide a degradation is not a verbosity knob.
    """
    nkx, nky, nkz = kgrid
    use_ibz = False
    q_irr_kgrid_int = None
    q_full_to_irr_idx = None
    q_full_to_irr_sym = None
    sym_perm = None
    L_table = None
    res = None
    if sym is not None and centroid_indices is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        res = resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=centroid_indices, fft_grid=fft_grid,
            context=context)
        if res.use_ibz:
            sym_perm, L_table = res.tables()
        if sym_perm is not None and mu_basis is not None:
            # In-memory consumers (W, χ) hold their operators in the run's
            # packed centroid order: conjugate the canonical tables into it.
            # The pad bake below is then a no-op (the packed extent is
            # already the complete-mesh carrier).  V is built from the
            # canonical ζ file and passes no basis.
            sym_perm, L_table = mu_basis.pack_tables(sym_perm, L_table)
        if sym_perm is not None:
            # Bake the μ pad into the tables ONCE at construction:
            # identity tail on the permutation (pad centroids map to
            # themselves), zero tail on the umklapp wrap (pad centroids
            # never wrap).  Consumers (``unfold_isdf_operator``,
            # ``unfold_isdf_one_leg``, gw_jax's W unfold) then
            # REQUIRE an exact extent match instead of each re-padding
            # per site — the too-small/too-large guards there replace a
            # silent ``promise_in_bounds`` OOB gather (the TRS-bug
            # failure shape) with a loud error.
            from runtime.padding import padded_mu_extent
            n_rmu_log = int(sym_perm.shape[-1])
            n_rmu_pad = (mu_basis.n_packed if mu_basis is not None else
                         padded_mu_extent(n_rmu_log, int(jax.device_count())))
            if n_rmu_pad > n_rmu_log:
                tail = np.broadcast_to(
                    np.arange(n_rmu_log, n_rmu_pad, dtype=sym_perm.dtype),
                    (sym_perm.shape[0], n_rmu_pad - n_rmu_log))
                tail = np.where(np.all(sym_perm == -1, axis=1)[:, None], -1, tail)
                sym_perm = np.concatenate([sym_perm, tail], axis=-1)
                L_table = np.concatenate(
                    [L_table,
                     np.zeros((L_table.shape[0], n_rmu_pad - n_rmu_log, 3),
                              dtype=L_table.dtype)], axis=1)
            q_irr_kgrid_int = sym.q_irr_kgrid_int
            q_full_to_irr_idx = sym.irr_idx_q
            q_full_to_irr_sym = sym.sym_idx_q
            use_ibz = True

    if not use_ibz:
        q_irr_kgrid_int = np.array(
            [(qx, qy, qz) for qx in range(nkx)
             for qy in range(nky) for qz in range(nkz)],
            dtype=np.int32)

    # The symmetry service owns BGW's strict half-grid tie convention.
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import bgw_integer_q_to_fractional
    q_irr_frac = bgw_integer_q_to_fractional(q_irr_kgrid_int, kgrid)
    result = (q_irr_kgrid_int, q_irr_frac,
              q_full_to_irr_idx, q_full_to_irr_sym,
              sym_perm, L_table, use_ibz)
    return (*result, res) if return_resolution else result


#: Upper bound on the automatic G panel width.  With the panel gathers inside
#: the G scan this is also the per-step collective width, so it is capped by
#: the collective payload bound as well (``_plan_vq_tiles``).
_VQ_G_CHUNK_TARGET = 4096


def vq_tile_bytes(*, n_q: int, n_rmu_L: int, n_rmu_R: int, ngkmax: int,
                  same_zeta: bool, n_sub: int, p_x: int, p_y: int,
                  g_chunk: int) -> dict:
    """Per-rank bytes of one V tile's contraction; the V_q memory model.

    complex128, ``P = p_x·p_y``, μ extents already mesh-padded::

        resident = V_acc 16·n_q·μ_L·μ_R/P + g0_acc 16·n_q·μ_L/p_x
                 + one-leg columns 16·n_q·μ_L·n_sub/P
        per_q    = ζ rows 16·(μ_L [+ μ_R])·n_G/P + v row 16·n_G (replicated)
        host_per_q = ζ rows 16·(μ_L [+ μ_R])·n_G/P  (phdf5 host read staging)
        work     = faces 16·(μ_L + μ_R [+ μ_R])·n_G/P  (the sliced face(s)
                   and the ('y','x') permute of the R face)
                 + V_q carry 2·16·μ_L·μ_R/P
                 + panels 16·g·(2·μ_L/p_x + μ_R/p_y)   (L, L·v, R)

    The single owner of these terms: ``_plan_vq_tiles`` sizes the run from
    them and ``gflat_memory_model`` prices its V_q stage E with them.
    """
    c = 16.0
    p_all = int(p_x) * int(p_y)
    rows = n_rmu_L + (0 if same_zeta else n_rmu_R)
    faces = n_rmu_L + n_rmu_R + (0 if same_zeta else n_rmu_R)
    panel_col = c * (2.0 * n_rmu_L / p_x + n_rmu_R / p_y)
    fixed = c * (faces * ngkmax / p_all + 2.0 * n_rmu_L * n_rmu_R / p_all)
    return dict(
        resident=c * (n_q * n_rmu_L * n_rmu_R / p_all + n_q * n_rmu_L / p_x
                      + n_q * n_rmu_L * n_sub / p_all),
        per_q=c * (rows * ngkmax / p_all + ngkmax),
        host_per_q=c * rows * ngkmax / p_all,
        fixed_work=fixed, panel_col=panel_col,
        work=fixed + panel_col * int(g_chunk))


def _plan_vq_tiles(*, n_q: int, n_rmu_L: int, n_rmu_R: int, ngkmax: int,
                   same_zeta: bool, n_sub: int, mesh_xy: Mesh,
                   g_chunk: int | None, budget_bytes: float,
                   host_budget_bytes: float = float('inf')):
    """Size the G panel and the ζ q-tile of one V tile from the V_q memory budget.

    Bytes are :func:`vq_tile_bytes`.  ``g`` (0/None = auto) is the largest
    width ≤ ``_VQ_G_CHUNK_TARGET`` whose widest per-step panel all-gather
    fits ``LORRAX_COLLECTIVE_CHUNK_MB`` and whose panels take at most half of
    what one q leaves free; an explicit ``g_chunk`` (deck
    ``vq_g_chunk_size``) is used as given.  The q-tile is then every q that
    fits the device budget and whose read staging, ``host_per_q`` per q,
    fits ``host_budget_bytes`` (``host_bytes_per_process``) — all of them
    whenever they do, which is the whole-slab read — balanced so the last
    tile is not a sliver.  Every input is
    rank-invariant (the caller agrees the budget across processes), so every
    rank issues the same collective reads.

    Returns ``(q_tile, g_chunk, priced)``; refuses when V_acc plus one q
    does not fit, the only case no tile or panel choice can rescue.
    """
    from common.collectives import _owner_gather_chunk_bytes

    p_x, p_y = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    shape = dict(n_q=n_q, n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R, ngkmax=ngkmax,
                 same_zeta=same_zeta, n_sub=n_sub, p_x=p_x, p_y=p_y)
    b = vq_tile_bytes(**shape, g_chunk=0)
    if g_chunk:
        g = min(int(g_chunk), int(ngkmax))
        if g < 1:
            raise ValueError(f"vq_g_chunk_size must be positive; got {g_chunk}.")
    else:
        widest_row = 16.0 * max(n_rmu_L / p_x, n_rmu_R / p_y)
        g = min(int(ngkmax), _VQ_G_CHUNK_TARGET,
                max(1, int(_owner_gather_chunk_bytes() // widest_row)))
        free = budget_bytes - b['resident'] - b['per_q'] - b['fixed_work']
        g = max(1, min(g, int(0.5 * free // b['panel_col'])))
    b = vq_tile_bytes(**shape, g_chunk=g)
    resident, per_q, work = b['resident'], b['per_q'], b['work']
    host_per_q = b['host_per_q']
    q_fit = int(min((budget_bytes - resident - work) // per_q,
                    host_budget_bytes // host_per_q, n_q))
    if q_fit < 1:
        raise ValueError(
            "GATE vq_tile_budget: "
            f"got V_acc/g0/one-leg {resident / 1e9:.2f} GB + one q "
            f"{per_q / 1e9:.2f} GB + faces/panels {work / 1e9:.2f} GB per rank; "
            f"want <= the V_q budget {budget_bytes / 1e9:.2f} GB, and one q's "
            f"host read staging {host_per_q / 1e9:.2f} GB <= "
            f"{host_budget_bytes / 1e9:.2f} GB; "
            "why: the output accumulator and one q's ζ face cannot both be "
            "resident, so no q-tile or G-panel choice can run this tile; "
            "fix: add ranks (every term above is ÷P) or free device memory "
            "before V_q.")
    q_max = min(int(n_q), q_fit)
    n_tiles = -(-int(n_q) // q_max)
    q_tile = -(-int(n_q) // n_tiles)
    priced = resident + work + q_tile * per_q
    return q_tile, g, dict(resident=resident, per_q=per_q, work=work,
                           priced=priced, n_tiles=n_tiles,
                           host_staged=q_tile * host_per_q)


def _vq_budget_bytes(budget_bytes: float | None) -> float:
    """The per-rank V_q device budget, agreed across processes (the minimum).

    ``None`` measures it live: ``common.gpu_utils.get_device_memory_info``
    ``budget_gb`` (0.9 of this device's currently available memory), the
    same rule the gw_init ``V_q budget`` line prints.  The q-tile count sets
    how many collective reads every rank issues, so the value must be the
    same on every rank; the minimum is the only safe agreement.
    """
    from common.gpu_utils import get_device_memory_info, minimum_process_budget_gb
    if budget_bytes is None:
        local_gb = float(get_device_memory_info()['budget_gb'])
    else:
        local_gb = float(budget_bytes) / 1e9
    return minimum_process_budget_gb(local_gb) * 1e9


def _make_read_q_tile(zeta_loader, n_rmu_padded: int, mesh_xy: Mesh):
    """Return ``read_q_tile(q_offset, q_count) -> (q_count, n_rmu_padded, ngkmax)``.

    One read shape per tile: ``ZetaLoader.read_zeta_G_slab`` at
    ``n_rmu_padded`` rows.  SlabIO zero-fills past the dataset's own μ
    extent (decisions.md 2026-08-04), so the caller states the extent it
    wants to consume and nothing else.  One call per q-TILE, not per q:
    each ``read_slab`` call is a fresh ``_per_rank`` closure and a trace
    cache miss in the FFI dispatch (2026-05-12: 63 per-q reads on the MoS2
    3×3 bispinor deck became 7 whole-slab reads), and the tile count is 1
    whenever every q fits the budget.
    """
    def read_q_tile(q_offset: int, q_count: int) -> jax.Array:
        return zeta_loader.read_zeta_G_slab(
            q_offset=int(q_offset), q_count=int(q_count),
            mu_offset=0, mu_count=int(n_rmu_padded),
            mesh=mesh_xy,
        )

    return read_q_tile


def _one_leg_columns(gvec_components, *, sym, sym_idx, q_irr_frac, kgrid):
    """``(n_q_ibz, n_sub)`` parent-sphere slots the IBZ one-leg unfold reads.

    ``symmetry_maps.isdf_one_leg_source_slots`` names the slot of every full
    q's literal-G=0 coefficient; a parent keeps the distinct slots of its
    star, padded to a common ``n_sub`` with the lowest slots the star does
    not use.  Those are genuine sphere columns whose G the star does not
    need, so on the sub-sphere the service finds every needed G exactly once
    and gathers the same coefficients it would from the whole sphere.
    """
    from symmetry_maps import isdf_one_leg_source_slots
    slots = isdf_one_leg_source_slots(
        gvec_components, sym=sym, sym_idx=sym_idx,
        q_irr_frac=q_irr_frac, kgrid=kgrid)
    parent = np.asarray(sym.irr_idx_q, dtype=np.int32)
    n_q, _, n_G = (int(s) for s in np.shape(gvec_components))
    need = [np.unique(slots[parent == p]) for p in range(n_q)]
    n_sub = max(1, max(len(cols) for cols in need))
    out = np.empty((n_q, n_sub), dtype=np.int32)
    for p, cols in enumerate(need):
        filler = np.setdiff1d(np.arange(n_G, dtype=np.int32), cols)
        out[p] = np.concatenate([cols, filler[:n_sub - len(cols)]])
    return out


_ONE_LEG_TAKE_CACHE: dict = {}


def _one_leg_take(mesh_xy: Mesh):
    """Cached ``(take, stack)`` jits for the per-tile one-leg column carrier."""
    hit = _ONE_LEG_TAKE_CACHE.get(id(mesh_xy))
    if hit is not None:
        return hit
    sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    cols_sh = NamedSharding(mesh_xy, P(None, None))

    @partial(jax.jit, in_shardings=(sh, cols_sh), out_shardings=sh)
    def take(zeta_tile, cols):
        # G is unsharded, so this gather is local to every rank.
        return jnp.take_along_axis(zeta_tile, cols[:, None, :], axis=2)

    @partial(jax.jit, out_shardings=sh)
    def stack(parts):
        return jnp.concatenate(parts, axis=0)

    _ONE_LEG_TAKE_CACHE[id(mesh_xy)] = (take, stack)
    return take, stack


# ---------------------------------------------------------------------------
# Per-tile core (one (μ_L, ν_L) tile)
# ---------------------------------------------------------------------------

def _compute_V_q_g_flat_one_tile(
    zeta_L_loader,
    zeta_R_loader,                     # None ⇒ same_zeta=True
    *,
    v_per_G_builder,                   # callable(q_irr_frac, gvec_components) -> (n_q, ngkmax) c128
    kgrid, fft_grid, mesh_xy,
    g_chunk: int | None,
    sym, centroid_indices,             # IBZ closure check is on the L centroids
    is_charge_cc: bool,
    write_g0: bool,
    one_leg_action: str,
    qgrid_policy=None,
    source_component: int | None = None,
    timing_label: str,
    verbose: bool,
    budget_bytes: float | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Contract one q-parent V tile and its separately transported full-q G=0 leg.

    ``budget_bytes`` is the per-rank V_q memory allowance that sizes the ζ
    q-tile and the G panel (``_plan_vq_tiles``); ``None`` measures it live
    (``_vq_budget_bytes``).  Either way it is agreed across processes.
    """
    same_zeta = (zeta_R_loader is None) or (zeta_R_loader is zeta_L_loader)
    # ``n_rmu_*`` is the logical centroid count for each side — read off the
    # loader so callers don't repeat themselves.
    n_rmu_L = int(zeta_L_loader.n_rmu)
    n_rmu_R = n_rmu_L if same_zeta else int(zeta_R_loader.n_rmu)
    if str(getattr(zeta_L_loader, 'zeta_layout', '')) != 'G_flat':
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: zeta_L "
            f"layout must be 'G_flat'; got "
            f"{getattr(zeta_L_loader, 'zeta_layout', None)!r}")
    if (not same_zeta and
            str(getattr(zeta_R_loader, 'zeta_layout', '')) != 'G_flat'):
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: zeta_R "
            "layout must be 'G_flat'.")

    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_one_leg

    # ---- IBZ list + per-tile v(q+G) -----------------------------------
    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, L_table, use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid,
        context=f"V_q g-flat tile [{timing_label}]")
    n_q_ibz = int(q_irr_frac.shape[0])

    policy = qgrid_policy
    unfold_sym = full_to_irr_sym
    if use_ibz:
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        if policy is None:
            from .qgrid_symmetry import qgrid_trs_policy_for
            policy = qgrid_trs_policy_for(
                sym=sym, irr_idx_q=full_to_irr_idx,
                sym_idx_q=full_to_irr_sym, kgrid=tuple(kgrid),
                n_sym_spatial=n_sym_spatial,
                context=f"V_q / one-leg [{timing_label}]")
        unfold_sym = np.asarray(policy.unfold_sym_idx, dtype=np.int32)
        if unfold_sym.shape != np.asarray(full_to_irr_sym).shape:
            raise ValueError(
                f"_compute_V_q_g_flat_one_tile[{timing_label}]: shared "
                "QgridTrsPolicy has the wrong q extent.")

    gvec_components = np.asarray(
        zeta_L_loader.gvec_components, dtype=np.int32)
    if gvec_components.shape[0] != n_q_ibz:
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: ζ_L on "
            f"disk has {gvec_components.shape[0]} q's; resolved IBZ "
            f"has {n_q_ibz}.  Mismatch — was the file written with the "
            f"same write_ibz_only setting?")
    if not same_zeta:
        gvec_R = np.asarray(zeta_R_loader.gvec_components, dtype=np.int32)
        if gvec_R.shape != gvec_components.shape:
            raise ValueError(
                f"_compute_V_q_g_flat_one_tile[{timing_label}]: ζ_L vs "
                f"ζ_R gvec_components shape mismatch "
                f"({gvec_components.shape} vs {gvec_R.shape}).  Both "
                f"files must be written with matching zeta_cutoff_ry "
                f"and q-layout.")
    ngkmax = int(gvec_components.shape[-1])

    v_q_table = np.asarray(
        v_per_G_builder(q_irr_frac, gvec_components),
        dtype=np.complex128)                                # (n_q_ibz, ngkmax)
    if v_q_table.shape != (n_q_ibz, ngkmax):
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: "
            f"v_per_G_builder returned shape {v_q_table.shape}; "
            f"expected ({n_q_ibz}, {ngkmax}).")

    # ---- μ padding to mesh-product per side ---------------------------
    # ``padded_mu_extent`` = the same round-up (+ test-only
    # LORRAX_EXTRA_MU_PAD rows) as ``Meta.n_rmu_padded`` — the V tiles
    # built here must match the ψ-side μ extent exactly.
    from runtime.padding import padded_mu_extent
    def _pad(n: int) -> int:
        return padded_mu_extent(int(n), mesh_xy)
    n_rmu_L_padded = _pad(int(n_rmu_L))
    n_rmu_R_padded = _pad(int(n_rmu_R))

    # ---- IBZ one-leg columns (literal full-zone G=0) -------------------
    # On an IBZ the literal full-zone G=0 may be a nonzero parent G, so the
    # kernel's slot-zero g0 is unsafe there.  The symmetry service names the
    # parent slots it will read; each q-tile contributes just those columns,
    # and the whole slab never has to outlive its tile for the unfold.
    one_leg = bool(write_g0 and use_ibz)
    one_leg_cols = (_one_leg_columns(
        gvec_components, sym=sym, sym_idx=unfold_sym,
        q_irr_frac=q_irr_frac, kgrid=kgrid) if one_leg else None)
    n_sub = int(one_leg_cols.shape[1]) if one_leg else 0

    # ---- G panel and q-tile from the V_q budget ------------------------
    budget = _vq_budget_bytes(budget_bytes)
    # A ζ q-tile read stages each rank's slab in its phdf5 file context's host
    # buffer (``ctx->read_buf``, ``src/ffi/cpp/phdf5/context.cc``), kept at the
    # largest read until ``ZetaLoader.release_read_staging``.
    from common.gpu_utils import host_bytes_per_process
    host_budget = host_bytes_per_process()
    q_tile, g_chunk, priced = _plan_vq_tiles(
        n_q=n_q_ibz, n_rmu_L=n_rmu_L_padded, n_rmu_R=n_rmu_R_padded,
        ngkmax=ngkmax, same_zeta=same_zeta, n_sub=n_sub, mesh_xy=mesh_xy,
        g_chunk=g_chunk, budget_bytes=budget, host_budget_bytes=host_budget)
    n_chunks = -(-ngkmax // g_chunk)
    n_tiles = int(priced['n_tiles'])
    if verbose and jax.process_index() == 0:
        print(f"  V_q g-flat [{timing_label}]: n_q_ibz={n_q_ibz}, "
              f"ngkmax={ngkmax}, g_chunk={g_chunk} ({n_chunks}/q), "
              f"n_rmu_L={n_rmu_L}→{n_rmu_L_padded}, "
              f"n_rmu_R={n_rmu_R}→{n_rmu_R_padded}, "
              f"storage={'q-IBZ' if use_ibz else 'full-BZ'}",
              flush=True)
        print(f"  V_q plan [{timing_label}]: q_tile={q_tile} "
              f"({n_tiles} tile(s)), priced {priced['priced'] / 1e9:.2f} GB/rank "
              f"= resident {priced['resident'] / 1e9:.2f} + "
              f"{q_tile}×{priced['per_q'] / 1e9:.3f} ζ/q + faces/panels "
              f"{priced['work'] / 1e9:.2f}, budget {budget / 1e9:.2f} GB; "
              f"host read staging {priced['host_staged'] / 1e9:.2f} of "
              f"{host_budget / 1e9:.2f} GB/rank",
              flush=True)

    # ---- Accumulators ---------------------------------------------------
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    V_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_L_padded, n_rmu_R_padded), dtype=jnp.complex128),
        out_shardings=V_sh)()
    # ``g0_acc`` is also the donate-target when ``write_g0=False`` — the
    # kernel still needs the buffer; the contents are simply unread.
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    g0_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_L_padded), dtype=jnp.complex128),
        out_shardings=g0_sh)()
    # Process-local placement, NOT plain ``jax.device_put``: the latter
    # fires JAX's hidden ``assert_equal`` all-gather on a multi-process
    # mesh (scorecard AA.1).  The v(q+G) rows are a pure function of the
    # q-grid + cutoff, identical on every rank; ``LORRAX_CHECK_REPLICA=1``
    # re-arms the check.  Placed per tile, so the replicated v table is
    # bounded by the tile too.
    from common.collectives import device_put_process_local
    v_rows_sh = NamedSharding(mesh_xy, P(None, None))

    read_L = _make_read_q_tile(zeta_L_loader, n_rmu_L_padded, mesh_xy)
    read_R = (read_L if same_zeta
              else _make_read_q_tile(zeta_R_loader, n_rmu_R_padded, mesh_xy))
    take_cols, stack_cols = _one_leg_take(mesh_xy) if one_leg else (None, None)
    one_leg_parts = []

    # ---- q-tile loop: read (sync) → contract → next --------------------
    # THE READ IS SYNCHRONOUS BETWEEN KERNEL CALLS, and that is load-bearing.
    # The historical per-q PHDF5 read inside the kernel loop interleaved its
    # MPI collectives with the kernel's NCCL collectives and was the root
    # cause of the async-prefetch deadlock.  So each tile's read completes
    # (``block_until_ready``) before its kernel is dispatched, and the kernel
    # completes before the next tile's read is issued.  With every q in one
    # tile — whenever they fit the budget — this is the old single batched
    # pre-read followed by one kernel launch.
    #
    # Multiplier: ``v_q_bispinor`` calls this function once per UNIQUE_TILE,
    # so the counts here are per V tile (7 tiles × n_tiles reads).
    import time as _t
    _read_total = 0.0
    _kernel_total = 0.0
    for t in range(n_tiles):
        q0 = t * q_tile
        qn = min(q_tile, n_q_ibz - q0)
        _t0 = _t.perf_counter()
        zeta_L_tile = read_L(q0, qn)                    # (qn, n_rmu_L_padded, ngkmax)
        zeta_R_tile = zeta_L_tile if same_zeta else read_R(q0, qn)
        jax.block_until_ready(zeta_L_tile)
        if not same_zeta:
            jax.block_until_ready(zeta_R_tile)
        _t1 = _t.perf_counter()
        _read_total += _t1 - _t0
        kernel = _make_q_tile_kernel(
            mesh_xy, n_rmu_L_padded, n_rmu_R_padded, ngkmax, g_chunk, qn,
            write_g0=bool(write_g0 and not use_ibz), same_zeta=same_zeta)
        v_tile = device_put_process_local(
            np.ascontiguousarray(v_q_table[q0:q0 + qn]), v_rows_sh)
        V_acc, g0_acc = kernel(
            V_acc, g0_acc, zeta_L_tile, zeta_R_tile, v_tile, jnp.int32(q0))
        if one_leg:
            one_leg_parts.append(take_cols(
                zeta_L_tile, device_put_process_local(
                    np.ascontiguousarray(one_leg_cols[q0:q0 + qn]),
                    NamedSharding(mesh_xy, P(None, None)))))
        # Every process rendezvouses here (the wait consumes a sharded
        # accumulator, so it cannot sit under the rank-0 print gate;
        # INVARIANTS row 21); this is also what orders the kernel's NCCL
        # before the next tile's collective read.
        jax.block_until_ready(V_acc)
        _kernel_total += _t.perf_counter() - _t1
        del zeta_L_tile, zeta_R_tile, v_tile
        if verbose and jax.process_index() == 0:
            print(f"    [{timing_label}] q-tile {t + 1}/{n_tiles} "
                  f"(q {q0}..{q0 + qn - 1}): read={_t1 - _t0:.2f}s "
                  f"kernel={_t.perf_counter() - _t1:.2f}s "
                  f"({(_t.perf_counter() - _t1) / qn:.3f}s/q)", flush=True)
    # The read contexts keep their largest tile staged on the host until the
    # file closes, and the bispinor build holds four ζ loaders open across
    # seven V tiles; without this their staging accumulates (VI3 12x12 P16:
    # host OOM-kill at the fourth file).  Collective, like the reads.
    zeta_L_loader.release_read_staging()
    if not same_zeta:
        zeta_R_loader.release_read_staging()
    if verbose and jax.process_index() == 0:
        print(f"    [{timing_label}] {n_q_ibz} IBZ q in {n_tiles} tile(s): "
              f"read={_read_total:.2f}s kernel={_kernel_total:.2f}s "
              f"({_kernel_total / max(1, n_q_ibz):.3f}s/q)", flush=True)

    if one_leg:
        zeta_cols = (one_leg_parts[0] if len(one_leg_parts) == 1
                     else stack_cols(one_leg_parts))
        del one_leg_parts
        g0_acc = unfold_isdf_one_leg(
            zeta_cols,
            gvec_components=np.take_along_axis(
                gvec_components, one_leg_cols[:, None, :], axis=2),
            sym=sym,
            sym_idx=unfold_sym,
            sym_perm=sym_perm,
            L_table=L_table,
            q_irr_frac=q_irr_frac,
            kgrid=kgrid,
            mesh_xy=mesh_xy,
            component_action=one_leg_action,
            source_component=source_component,
        )
        del zeta_cols

    # ---- IBZ → full-BZ unfold (centroid double-permute) -------------
    if use_ibz:
        # ``sym_perm`` came from ``centroid_source_map_and_wrap(...,
        # extend_trs=True)`` so its shape[0] is ``2·ntran``; the second half
        # encodes the TRS-augmented rows (centroid permutation unchanged under
        # TRS, but the unfold helper conjugates V_q at TRS-tagged q's).
        # ``L_table`` is the per-(sym, μ) integer lattice wrap; the
        # umklapp phase ``exp(2π i q_irr · (L_μ − L_ν))`` is essential
        # for non-cubic / non-symmorphic systems.
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        if is_charge_cc:
            # TIME REVERSAL IS EXPLICIT, NEVER ASSUMED HERE. This block used to
            # compose q with −q through Θ and project the self-negative
            # rows unconditionally; on a ferromagnet that fabricates a
            # symmetry the reference verdict says is absent. The policy reads
            # ``sym.trs_allowed`` and
            # this site no longer contains a TRS branch of its own.
            # The point-group covariance the unfold below ASSUMES of the
            # finite ζ basis, measured on the stored parents while they are
            # still the pre-unfold wedge.  The q↔−q gate downstream is
            # structurally blind to it at a self-negative q (there it
            # degenerates to "V_q is real"); this is not.
            cov = policy.measure_covariance(
                V_acc, q_irr_frac=q_irr_frac,
                q_irr_full_idx=sym.q_irr_full_idx,
                sym_mats_k=sym.sym_mats_k, sym_perm=sym_perm,
                L_table=L_table)
            V_acc, removed = policy.project_fixed_q(
                V_acc, sym.q_irr_full_idx)
            if jax.process_index() == 0:
                from common import sanity
                sanity.report_parent_covariance(
                    "V_q[CC] IBZ parents", cov, removed=removed)
        # THE PRE-UNFOLD BLOCK, OFFERED TO WHOEVER IS WRITING THE RESTART.
        # This is the array the q_irr format persists — the design's
        # load-bearing decision, because ``unfold(stored)`` is then the
        # SAME CALL on the SAME ARGUMENTS the line below makes, an identity
        # rather than a property that depends on the op-selection policy.
        # It exists for exactly one statement, which is why the offer is
        # here and not at the writer.
        #
        # A NO-OP unless a driver has opened a capture scope, so the
        # compute path takes no restart decision and this line costs a list
        # check on every other run.  Only the CC tile is offered: the
        # bispinor CT/TT tiles are not restart tensors.
        if is_charge_cc:
            from .restart_q_storage import deposit_pre_unfold
            deposit_pre_unfold(
                "V_qmunu", V_acc,
                n_rmu_logical=int(zeta_L_loader.n_rmu),
                q_irr_frac=q_irr_frac, irr_idx_q=full_to_irr_idx,
                sym_idx_q=unfold_sym, sym_perm=sym_perm,
                L_table=L_table, n_sym_spatial=n_sym_spatial)

    V_qmunu = jax.lax.with_sharding_constraint(V_acc, V_sh)
    if write_g0:
        g0_spec = (P(None, 'x') if int(g0_acc.ndim) == 2
                   else P(None, None, 'x'))
        return V_qmunu, jax.lax.with_sharding_constraint(
            g0_acc, NamedSharding(mesh_xy, g0_spec))
    return V_qmunu, None


# ---------------------------------------------------------------------------
# Public charge entry point (CC tile only)
# ---------------------------------------------------------------------------

def compute_all_V_q_g_flat(
    zeta_loader,                       # ZetaLoader (G-flat)
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    sys_dim: int,
    bdot: np.ndarray | None = None,
    bare_coulomb_cutoff_ry: float | None = None,
    bgw_v_grid_fn=None,
    mc_average_vcoul_body: bool = True,
    g_chunk: int | None = None,
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    budget_bytes: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """V_q^{0,0} (charge-channel CC tile) on a G-flat-on-disk ζ file.

    Thin wrapper that builds the bare-Coulomb ``v(q+G)`` per-q-sphere
    builder and dispatches to :func:`_compute_V_q_g_flat_one_tile`
    with ``zeta_R=None`` (same_zeta) and ``write_g0=True``.  The sync
    per-q loop is already ~6× faster than the legacy μ × ν tile driver
    on MoS2 3×3.

    See :func:`_compute_V_q_g_flat_one_tile` for the math + I/O flow.
    """
    if sys_dim not in (2, 3):
        raise NotImplementedError(
            f"compute_all_V_q_g_flat: sys_dim must be 2 or 3 "
            f"(0-D box per-q v(G) not wired); got {sys_dim}.")
    # compute_v_q_per_G is gw's wfn-facing translation over the vcoul door
    # (old bvec/cell_volume/sys_dim signature) and correctly stays a gw
    # import; build_v_head_miniBZ_fn_3d is a pure service symbol, so its
    # true dependency is the door (replumbed 2026-08-07).  ORDER IS
    # LOAD-BEARING: .compute_vcoul runs the service path bootstrap at its
    # module scope, so it must be imported BEFORE the door — the blind
    # audit arm measured the swapped order dying with ModuleNotFoundError
    # in a stripped process where nothing else had bootstrapped yet.
    from .compute_vcoul import compute_v_q_per_G
    from vcoul import build_v_head_miniBZ_fn_3d

    # 3D bulk: build the mini-BZ-averaged head ⟨v(K+δq)⟩ ONCE, as a
    # function of the Cartesian K = q+G.  ``v_qG_table`` evaluates it at
    # every slot attaining argmin |q+G| — see its HEAD SLOT note; the old
    # per-q table keyed on the Miller-(0,0,0) label was not equivariant
    # under q → −q and cost V_q 6.0e−3 of reciprocity against a 1.16e−7
    # floor.
    # The IBZ → full-BZ V_q unfold is bilinear in ζ and inherits this
    # head value through ``unfold_isdf_operator``'s centroid-permute + L-phase,
    # so injecting at every IBZ q is sufficient — no separate full-BZ pass.
    # 2D ``f2d → 0`` regularizes v at G=0 already; the MC flag is a 3D-
    # only refinement and is silently no-op'd for sys_dim=2.
    _v_head_fn = None
    if mc_average_vcoul_body and sys_dim == 3:
        _v_head_fn = build_v_head_miniBZ_fn_3d(
            kgrid, bvec, cell_volume)

    def _bare_v_per_G(q_irr_frac, gvec_components):
        v = compute_v_q_per_G(
            q_irr_frac, gvec_components,
            bvec=bvec, cell_volume=cell_volume,
            sys_dim=sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
            bdot=bdot,
            v_head_fn=_v_head_fn,
        )                                                   # (n_q_ibz, ngkmax) f64
        # Optional BGW vcoul overlay — host-side scatter from BGW's
        # full-FFT-grid v into the per-q WFN.h5 sphere positions.
        if bgw_v_grid_fn is not None:
            nx, ny, nz = (int(s) for s in fft_grid)
            for qi in range(q_irr_frac.shape[0]):
                v_full = np.asarray(
                    bgw_v_grid_fn(tuple(q_irr_frac[qi]))).reshape(-1)
                miller = gvec_components[qi]                # (3, ngkmax)
                ix = miller[0] % nx
                iy = miller[1] % ny
                iz = miller[2] % nz
                v_at_sphere = v_full[ix * ny * nz + iy * nz + iz]
                v[qi] = np.where(v_at_sphere != 0.0, v_at_sphere, v[qi])
        return v.astype(np.complex128)

    V_q, g0 = _compute_V_q_g_flat_one_tile(
        zeta_loader, None,
        v_per_G_builder=_bare_v_per_G,
        kgrid=kgrid, fft_grid=fft_grid,
        mesh_xy=mesh_xy,
        g_chunk=g_chunk,
        sym=sym, centroid_indices=centroid_indices,
        is_charge_cc=True,
        write_g0=True,
        one_leg_action="scalar",
        timing_label='CC',
        verbose=verbose,
        budget_bytes=budget_bytes,
    )

    from symmetry_maps import unfold_isdf_operator
    from .qgrid_symmetry import qgrid_trs_policy_for
    _, q_frac, irr, rows, perm, wraps, reduced = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid)
    if reduced:
        policy = qgrid_trs_policy_for(
            sym=sym, irr_idx_q=irr, sym_idx_q=rows, kgrid=kgrid,
            n_sym_spatial=len(perm) // 2, context="scalar V consumer")
        V_q = unfold_isdf_operator(
            V_q, irr_idx=irr, sym_idx=policy.unfold_sym_idx,
            sym_perm=perm, L_table=wraps, q_irr_frac=q_frac,
            mesh_xy=mesh_xy, n_sym_spatial=policy.n_sym_spatial)
    return V_q, g0


def compute_head_channel_zeta(
    zeta_loader,
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    sys_dim: int,
    bdot: np.ndarray | None = None,
    bare_coulomb_cutoff_ry: float | None = None,
    mc_average_vcoul_body: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    verbose: bool = True,
):
    """The q != 0 Coulomb head channel, in the centroid basis, on the full BZ.

    Returns ``(g_head, table)`` where ``g_head`` is
    ``(n_q_full, k, n_rmu_padded)`` complex128 at ``P(None, None, 'x')`` and
    ``table`` is the :class:`vcoul.HeadSlotTable` on the IBZ q-list.
    ``g_head[q, j, :]`` is ``zeta(q, mu, G_j)`` for the j-th slot attaining
    ``argmin |q+G|``, already multiplied by the tie mask — so padding
    columns, Γ, and any q whose head slot the bare-Coulomb cutoff zeroes are
    EXACT zeros and the projector

        P_q = sum_j conj(g_head[q, j]) (x) g_head[q, j]

    is the head-slot part of ``V_q`` divided by the value ``v_qG_table`` put
    there.  Consumed by ``gw.head_channel``; built only when a deck turns
    ``mc_average_placement`` on, which is why this is a second short read of
    the same ζ slabs rather than a third output of the V_q hot loop — the
    default V_q path stays byte-for-byte and compile-for-compile unchanged.

    Each selected source column is unfolded by
    ``symmetry_maps.unfold_isdf_one_leg`` with its actual parent Miller
    vector, including the centroid source gather, L phase, nonsymmorphic tau
    phase and measured antiunitary convention.  A tied set maps onto a tied
    set under the little group (``|q+G|`` is invariant), so the sum over
    columns is independent of their image ordering.
    """
    from .compute_vcoul import compute_v_q_per_G  # bootstrap, see the CC path
    from vcoul import (CoulombGeometry, build_v_head_miniBZ_fn_3d, get_kernel,
                       head_slot_table)

    del compute_v_q_per_G  # imported for the service-path bootstrap only

    if str(getattr(zeta_loader, 'zeta_layout', '')) != 'G_flat':
        raise ValueError(
            "compute_head_channel_zeta: zeta layout must be 'G_flat'; got "
            f"{getattr(zeta_loader, 'zeta_layout', None)!r}")

    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, L_table, use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid,
        context="head-channel zeta")
    n_q_ibz = int(q_irr_frac.shape[0])
    gvec_components = np.asarray(zeta_loader.gvec_components, dtype=np.int32)

    # THE SAME estimator object the V_q path builds — same seed, same draw
    # count, same centrosymmetrisation.  Built here rather than shared
    # because the two calls are in different stages and a deterministic pure
    # function is cheaper to rebuild than to thread.
    #
    # NOT GATED ON ``mc_average_vcoul_body``.  That flag decides whether the
    # average is substituted into V — i.e. what Sigma_X receives — and the
    # placement mode decides where the average lands in W.  Gating the head
    # function on the flag made ``mc_average_placement = bgw`` with the flag
    # off a silent no-op (<v> == v_c => r == 1), which is precisely the
    # "knob that quietly does nothing" failure this feature is supposed to
    # be immune to.  The flag is honoured where it belongs: in ``v_in_V``,
    # the value the production tile actually carries.
    v_head_fn = None
    if sys_dim == 3:
        v_head_fn = build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume)
    del mc_average_vcoul_body

    table = head_slot_table(
        get_kernel(sys_dim), q_irr_frac, gvec_components,
        geometry=CoulombGeometry(bvec=bvec, cell_volume=cell_volume,
                                 bdot=bdot, fft_grid=fft_grid),
        vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
        v_head_fn=v_head_fn,
    )

    from runtime.padding import padded_mu_extent
    n_rmu_padded = padded_mu_extent(
        int(zeta_loader.n_rmu),
        int(mesh_xy.shape['x']) * int(mesh_xy.shape['y']))

    # ponytail: this second consumer still reads every q at once (ζ_all/P per
    # rank); q-tile it through ``_make_read_q_tile`` like the V_q loop if a
    # deck with ``mc_average_placement`` or the metal q0 shift needs it.
    read_all = _make_read_q_tile(zeta_loader, n_rmu_padded, mesh_xy)
    zeta_all = read_all(0, n_q_ibz)           # (n_q_ibz, mu_pad, ngkmax)

    policy = None
    if use_ibz:
        from .qgrid_symmetry import qgrid_trs_policy_for
        policy = qgrid_trs_policy_for(
            sym=sym, irr_idx_q=full_to_irr_idx,
            sym_idx_q=full_to_irr_sym, kgrid=tuple(kgrid),
            n_sym_spatial=int(np.asarray(sym_perm).shape[0]) // 2,
            context="head-channel one-leg")
        from symmetry_maps import unfold_isdf_one_leg

    sel_dev = jnp.asarray(np.asarray(table.sel, dtype=np.int32))
    mask_dev = jnp.asarray(np.asarray(table.mask, dtype=np.float64),
                           dtype=jnp.complex128)
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    # (n_q, mu, k) gather on the UNSHARDED G axis, then transpose.  Done
    # per j so each intermediate is (n_q, mu) — the same class as g0_acc —
    # and so the sharding constraint lands on the shape the unfold wants.
    cols = []
    for j in range(int(table.sel.shape[1])):
        col = jnp.take_along_axis(
            zeta_all, sel_dev[:, None, j:j + 1], axis=2)[:, :, 0]
        col = col * mask_dev[:, j:j + 1]
        col = jax.lax.with_sharding_constraint(col, g0_sh)
        if use_ibz:
            source_slot = np.broadcast_to(
                np.asarray(table.sel[:, j], dtype=np.int32)[:, None, None],
                (n_q_ibz, 3, 1))
            source_g = np.take_along_axis(
                gvec_components, source_slot, axis=2)[:, :, 0]
            col = unfold_isdf_one_leg(
                col,
                source_gvec_components=source_g,
                sym=sym,
                sym_idx=policy.unfold_sym_idx,
                sym_perm=sym_perm,
                L_table=L_table,
                q_irr_frac=q_irr_frac,
                kgrid=kgrid,
                mesh_xy=mesh_xy,
                component_action="scalar",
            )
        cols.append(jax.lax.with_sharding_constraint(col, g0_sh))
    del zeta_all

    g_head = jnp.stack(cols, axis=1)          # (n_q_full, k, mu)
    g_head = jax.lax.with_sharding_constraint(
        g_head, NamedSharding(mesh_xy, P(None, None, 'x')))
    if verbose and jax.process_index() == 0:
        live = np.asarray(table.v_bare) > 0.0
        eta = np.zeros_like(np.asarray(table.v_bare))
        eta[live] = (np.asarray(table.v_avg)[live]
                     / np.asarray(table.v_bare)[live] - 1.0)
        print(f"  head channel: {int((table.mult > 0).sum())}/{n_q_ibz} IBZ q "
              f"carry a head slot, k={int(table.sel.shape[1])}, "
              f"tie histogram="
              f"{dict(zip(*[c.tolist() for c in np.unique(table.mult, return_counts=True)]))}",
              flush=True)
        # Per-shell eta, printed as data rather than summarised: it is the
        # ONE input the whole rescale is a function of, it is directly
        # comparable to BerkeleyGW's own vcoul dumps, and a run whose log
        # carries it can be audited without re-running anything.
        for qi in np.nonzero(live)[0]:
            print(f"    head slot q[{qi}] |q+G|^2={float(table.len2[qi]):.6f} "
                  f"mult={int(table.mult[qi])} v_c={float(table.v_bare[qi]):.6f} "
                  f"<v>={float(table.v_avg[qi]):.6f} eta={eta[qi]:.6f}",
                  flush=True)
    return g_head, table, full_to_irr_idx


__all__ = ["compute_all_V_q_g_flat", "_compute_V_q_g_flat_one_tile",
            "vq_tile_bytes",
            "_resolve_ibz_q_list", "_plan_vq_tiles", "_make_read_q_tile",
            "compute_head_channel_zeta"]
