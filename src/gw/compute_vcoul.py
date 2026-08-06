"""Host-side bare-Coulomb helpers + the V_q dispatcher.

After the 2026-07-02 r-space/FFT-tile deletion this module holds only
what the live G-flat V_q path needs:

* :func:`compute_v_q_per_G` — evaluate ``v(q+G)`` at the writer's per-q
  WFN.h5-style G-sphere (the ``isdf_header/gvec_components`` table).
  This is the only ``v`` builder the G-flat contract consumes.
* :func:`build_v_head_miniBZ_avg_3d` — the 3D mini-BZ-averaged
  ``<v(q, G=0)>`` head table injected into ``compute_v_q_per_G`` for
  bulk systems.
* :func:`compute_all_V_q` — the thin dispatcher: for a G-flat on-disk ζ
  it hands off to :func:`gw.v_q_g_flat.compute_all_V_q_g_flat`; any
  other layout raises ``NotImplementedError``.

The actual V_q contract (``V_q[μ,ν] = Σ_G conj(ζ̃_μ) v(q+G) ζ̃_ν``,
G-chunked) lives in ``gw.v_q_g_flat`` / ``gw.v_q_bispinor``; nothing in
this file does FFTs or μ × ν tiling any more.
"""

from typing import Callable, Optional

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from file_io.paths import resolve_input_path

from .coulomb.kernel import TOL_QG_ZERO, v_qG


def build_v_head_miniBZ_avg_3d(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> np.ndarray:
    """Mini-BZ-averaged bare Coulomb head ``<v(q+δq, G=0)>_miniBZ`` per q.

    3D bulk only.  Returns ``(nkx, nky, nkz)`` real array of head values
    in Rydberg / cell-volume units.  q=0 returns 0 (the actual head is
    injected separately via a rank-1 correction in the Σ_X path).

    The MC integration draws ``nmc`` (default 2¹⁸) points uniformly on
    the Voronoi cell of the mini-BZ and averages ``8π/|q+δq|²``.  The
    G-flat path (:func:`compute_v_q_per_G`, called from
    :func:`gw.v_q_g_flat.compute_all_V_q_g_flat`) consumes this table
    as the ``v_head_miniBZ`` argument.
    """
    from .vcoul import wrap_points_to_voronoi
    nkx, nky, nkz = (int(s) for s in kgrid)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = (randvals @ bvec.T)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), bvec_j, nmax=1))
    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    dq_cart = (randlims @ wrapped.T).T  # (nmc, 3) mini-BZ offsets in Cartesian

    v_head_avg = np.zeros((nkx, nky, nkz), dtype=np.float64)
    for qx in range(nkx):
        for qy in range(nky):
            for qz in range(nkz):
                qw = np.array([qx, qy, qz], dtype=np.float64)
                qw = np.where(qw > kgrid_arr / 2, qw - kgrid_arr, qw)
                q_frac = qw / kgrid_arr
                q_cart = q_frac @ bvec
                if np.dot(q_cart, q_cart) < 1e-12:
                    v_head_avg[qx, qy, qz] = 0.0  # q=0 head handled separately
                else:
                    shifted = q_cart[None, :] + dq_cart  # (nmc, 3)
                    denom = np.sum(shifted**2, axis=1)
                    v_head_avg[qx, qy, qz] = np.mean(8.0 * np.pi / denom)
    return v_head_avg * (1.0 / float(cell_volume))


def compute_v_q_per_G(
    q_irr_frac: np.ndarray,
    gvec_components: np.ndarray,
    *,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    vcoul_cutoff_ry: float | None = None,
    bdot: np.ndarray | None = None,
    v_head_miniBZ: np.ndarray | None = None,
    pad_policy: str = "leave",
    ngk_per_q: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``v(q+G)`` at the per-q WFN.h5-style G-list.

    Evaluates the bare (optionally 2D-truncated) Coulomb ``v(q+G)``
    directly on the per-q ``gvec_components`` table (rather than a full
    FFT grid) — the writer's ``isdf_header/gvec_components`` is exactly
    the input the G-flat V_q contract needs.  Returns one ``(ngkmax,)``
    row of ``v(q+G)`` per q in ``q_irr_frac``.

    The formula is :func:`gw.coulomb.kernel.v_qG` — the same source line
    the BSE builder and the mini-BZ sampler evaluate; the values are
    BYTE-IDENTICAL to the pre-2026-08-05 local copy in both ``sys_dim``
    (measured, Frontera job 7890613).

    ``pad_policy`` — what the pad slots of ``gvec_components`` (sentinel
    Miller index ``(-nx/2, -ny/2, -nz/2)``) hold.  ``"leave"`` (the
    default, and the only value this caller wants) evaluates ``v`` there
    like any other slot: the G-flat V_q contract has ζ̃ = 0 at pad slots,
    so the value is multiplied out.  ``"zero"`` is the OTHER live
    convention (:func:`bse.vq_interp.v_sphere_padded`), which needs the
    explicit zero because its ``ZG`` carries junk in the pad columns; it
    requires ``ngk_per_q`` (the per-q valid sphere length) because "which
    slot is pad" is caller knowledge, not something this function can
    infer from a Miller index.  Both conventions are correct for their own
    consumer; the parameter exists so that neither gets silently handed
    the other's.

    Parameters
    ----------
    q_irr_frac : ``(n_q, 3)`` float64
        Fractional q-vectors in BGW-wrap convention (already divided
        by kgrid).
    gvec_components : ``(n_q, 3, ngkmax)`` int32
        Per-q Miller indices from ``isdf_header.gvec_components``.
    bvec, cell_volume, sys_dim, vcoul_cutoff_ry, bdot
        Coulomb geometry / truncation.  ``vcoul_cutoff_ry`` zeroes
        ``v`` at G's with |q+G|² past the cutoff (== V_q's bare-Coulomb
        cutoff; may be < the ζ-sphere cutoff that built
        ``gvec_components``).

    Returns
    -------
    v_q_per_G : ``(n_q, ngkmax)`` float64
        ``v(q+G)`` evaluated at every (q, G) in the components table.

    Notes
    -----
    This is a *host-side* helper — the per-q ``v(q+G)`` is built once
    at consumer setup and pushed to device.  Not jitted; not sharded.
    For very large ngkmax this could be vectorised across q on device,
    but it's a one-shot cost per V_q run.
    """
    if sys_dim not in (0, 2, 3):
        raise NotImplementedError(
            f"compute_v_q_per_G: sys_dim must be 0 / 2 / 3; got {sys_dim}")
    if pad_policy not in ("leave", "zero"):
        raise ValueError(f"compute_v_q_per_G: pad_policy must be 'leave' or "
                         f"'zero'; got {pad_policy!r}")
    if pad_policy == "zero" and ngk_per_q is None:
        raise ValueError("compute_v_q_per_G: pad_policy='zero' needs "
                         "ngk_per_q (per-q valid sphere length)")
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)         # (n_q, 3, ngkmax)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"gvec_components must be (n_q, 3, ngkmax); got {gvec.shape}")
    n_q, _, ngkmax = gvec.shape
    bvec_f = np.asarray(bvec, dtype=np.float64)

    if sys_dim == 2:
        zc = float(np.pi / float(bvec_f[2, 2]))
    if v_head_miniBZ is not None:
        # Per-q grid index: round (q_frac * kgrid) and wrap modulo kgrid.
        # ``v_head_miniBZ`` is indexed by integer (qx, qy, qz) on the
        # k-grid (the table the legacy ``get_sqrt_v_and_phase`` consumes).
        head_arr = np.asarray(v_head_miniBZ, dtype=np.float64)
        if head_arr.ndim != 3:
            raise ValueError(
                f"v_head_miniBZ must be (nkx, nky, nkz); got shape "
                f"{head_arr.shape}")
        head_kgrid = np.array(head_arr.shape, dtype=np.float64)
    out = np.zeros((n_q, ngkmax), dtype=np.float64)

    for qi in range(n_q):
        qf = q_irr_frac[qi]
        # gvec[qi]: (3, ngkmax) -> per-G Miller; (q + G) in fractional.
        qG_frac = qf[:, None] + gvec[qi]                          # (3, ngkmax)
        qG_cart = bvec_f.T @ qG_frac                              # (3, ngkmax)
        denom = np.sum(qG_cart * qG_cart, axis=0)                 # (ngkmax,)
        if sys_dim == 0:
            # sys_dim == 0: caller passes ``bdot`` and we'd build the
            # FFT-grid sqrt_v0d here; not yet wired to per-q lookup.
            raise NotImplementedError(
                "compute_v_q_per_G: sys_dim=0 path not wired — the 0-D "
                "box truncation builds v on the full FFT grid via "
                "compute_sqrt_vcoul_0d; the per-q gather would map "
                "components → flat-FFT index → v(G).  Plumb when "
                "needed.")
        v = v_qG(qG_cart, axis=0, sys_dim=sys_dim, channel="full",
                 units="per_volume", celvol=cell_volume,
                 zc=(zc if sys_dim == 2 else None), zero_tol=TOL_QG_ZERO)
        if sys_dim == 3 and v_head_miniBZ is not None:
            # Replace the G=0 entry (the (0,0,0) Miller slot) with the
            # mini-BZ averaged head value for this q.  The head TABLE is
            # built around the SAME slot (the Miller-(0,0,0) shift q_cart),
            # so slot and shift agree.  ``argmin_G |q+G|²`` and
            # Miller-(0,0,0) do NOT coincide for every BGW-wrapped q on a
            # non-orthogonal lattice (12/64 q on the Si 4×4×4 fcc deck,
            # Frontera job 7890613) — moving the slot without moving the
            # shift would inject an average of the wrong neighbourhood.
            # See docs/dev/STATE.md, "Coulomb head slot".
            qx_i = int(np.round(qf[0] * head_kgrid[0])) % int(head_kgrid[0])
            qy_i = int(np.round(qf[1] * head_kgrid[1])) % int(head_kgrid[1])
            qz_i = int(np.round(qf[2] * head_kgrid[2])) % int(head_kgrid[2])
            g0_mask = np.all(gvec[qi] == 0.0, axis=0)             # (ngkmax,)
            v = np.where(g0_mask, head_arr[qx_i, qy_i, qz_i], v)
        if pad_policy == "zero":
            v = v.copy()
            v[int(ngk_per_q[qi]):] = 0.0
        if vcoul_cutoff_ry is not None:
            v = np.where(denom > float(vcoul_cutoff_ry), 0.0, v)
        out[qi] = v
    return out


def compute_all_V_q(
    zeta_io,
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    sys_dim: int = 2,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
    bgw_v_grid_fn=None,
    mu_chunk_size: int | None = None,   # legacy arg (allgather path); ignored
    q_batch_size: int | None = None,    # legacy arg (allgather path); ignored
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    g_chunk_size: int = 0,              # 0 = auto _pick_g_chunk(ngkmax)
) -> tuple[jax.Array, jax.Array]:
    """Compute V_qmunu(q,μ,ν) and g0_μ(q) at q=0 from a sharded ζ HDF5.

    Thin dispatcher over the single live V_q path.  When the on-disk
    ``ζ`` is in **G-flat** layout — the only thing
    :func:`fit_zeta_to_h5` writes — this routes to
    :func:`gw.v_q_g_flat.compute_all_V_q_g_flat`: per-q, G-chunked
    contract on the writer's per-q WFN.h5-style sphere (no FFT, no
    shared-sphere conversion, no μ × ν tiling).  Any other on-disk
    layout raises :class:`NotImplementedError` — the legacy r-space
    tile path (``gw/v_q_tile.py``) was deleted 2026-07-02.

    Working-set memory is bounded by ``g_chunk_size`` (per-q G-chunk)
    and the mesh-sharded ζ slabs; there is no separate byte budget knob.
    ``mu_chunk_size`` / ``q_batch_size`` are inert legacy args (no
    caller passes them; the G-flat chooser is trivial).
    """
    # G-flat dispatch — when the loader carries the new per-q sphere
    # components, hand off to the G-flat orchestrator (sync per-q loop;
    # it pre-reads all IBZ ζ̃ slabs in one batched call).
    if getattr(zeta_io, 'zeta_layout', None) == 'G_flat':
        from .v_q_g_flat import compute_all_V_q_g_flat
        return compute_all_V_q_g_flat(
            zeta_io,
            kgrid=kgrid, fft_grid=fft_grid,
            bvec=bvec, cell_volume=cell_volume,
            mesh_xy=mesh_xy,
            sys_dim=sys_dim, bdot=bdot,
            bare_coulomb_cutoff_ry=bare_coulomb_cutoff,
            bgw_v_grid_fn=bgw_v_grid_fn,
            mc_average_vcoul_body=mc_average_vcoul_body,
            g_chunk=(int(g_chunk_size) if g_chunk_size > 0 else None),
            verbose=verbose, sym=sym,
            centroid_indices=centroid_indices,
        )

    raise NotImplementedError(
        "compute_all_V_q: only the G-flat zeta layout is supported. "
        "fit_zeta_to_h5 writes G_flat exclusively; the r-space tile path "
        "was removed 2026-07-02.")


# ---------------------------------------------------------------------------
# Optional BGW vcoul override (moved from gw/gw_driver_helpers.py 2026-07-09
# — flag-gated diagnostic that belongs with the V_q machinery it feeds).
# ---------------------------------------------------------------------------

def build_bgw_v_grid_fn(
    config, *,
    wfn,
    sym,
    input_dir: str,
    print_fn=print,
) -> Optional[Callable[[tuple], np.ndarray]]:
    """Build the optional BGW vcoul override closure (or None).

    When ``config.head.use_bgw_vcoul`` is set the GW driver substitutes
    BGW's MC-averaged ``v(q+G)`` for LORRAX's internal head-only
    mini-BZ average — this enables bit-reproducible BGW comparisons.
    The returned callable maps a fractional q-tuple to a dense G-grid
    of Coulomb values; pass it through to ``compute_V_q``.

    Returns ``None`` when the override is disabled, so callers can do
    ``bgw_v_grid_fn = build_bgw_v_grid_fn(...)`` unconditionally.
    """
    head = config.head
    if not head.use_bgw_vcoul:
        return None
    if head.bgw_vcoul_file is None:
        raise ValueError(
            "head.use_bgw_vcoul=true requires head.bgw_vcoul_file to be set"
        )

    from file_io import read_bgw_vcoul, fill_v_grid_for_q

    bgw_path = resolve_input_path(input_dir, head.bgw_vcoul_file)
    print_fn(f"  BGW vcoul override: loading {bgw_path}")
    bgw_table = read_bgw_vcoul(bgw_path)
    print_fn(
        f"    {bgw_table.q_fracs.shape[0]} unique q-points, "
        f"G counts per q: {[len(g) for g in bgw_table.G_miller_per_q]}"
    )

    cell_volume = float(wfn.cell_volume)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # BGW's vcoul file only stores unique IBZ q's; mapping LORRAX's
    # full-BZ q to those needs the full crystal sym group.  A nosym WFN
    # stores only identity in mf_header/symmetry/mtrx, so allow pulling
    # the 48 ops from an aux sym-reduced WFN when provided.
    if head.bgw_vcoul_sym_wfn:
        aux_path = resolve_input_path(input_dir, head.bgw_vcoul_sym_wfn)
        with h5py.File(aux_path, "r") as fsym:
            sym_real = np.asarray(
                fsym["mf_header/symmetry/mtrx"][:], dtype=np.int32)
        print_fn(f"    crystal sym ops loaded from {aux_path}: {sym_real.shape[0]}")
        sym_mats_k = sym_real.transpose(0, 2, 1).copy()
    else:
        sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)

    def bgw_v_grid_fn(q_frac_tuple):
        return fill_v_grid_for_q(
            bgw_table, q_frac_tuple, fft_grid, cell_volume,
            sym_mats_k=sym_mats_k,
        )

    return bgw_v_grid_fn
