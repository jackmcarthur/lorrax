"""Host-side bare-Coulomb helpers + the V_q dispatcher.

After the 2026-07-02 r-space/FFT-tile deletion this module holds only
what the live G-flat V_q path needs:

* :func:`compute_v_q_per_G` — evaluate ``v(q+G)`` at the writer's per-q
  WFN.h5-style G-sphere (the ``isdf_header/gvec_components`` table).
  This is the only ``v`` builder the G-flat contract consumes.  Since
  2026-08-06 it is a **thin dispatcher** over
  :func:`gw.coulomb.base.v_qG_table`: the arithmetic lives once per
  dimensionality in :mod:`gw.coulomb`, shared with that package's
  ``CoulombKernel.v_qG``.  There is no second implementation to keep in
  step any more.
* :func:`build_v_head_miniBZ_fn_3d` — the 3D mini-BZ-averaged Coulomb
  head ``<v(K + δq)>_miniBZ`` as a function of the Cartesian ``K = q+G``,
  injected into ``compute_v_q_per_G`` for bulk systems.
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


def build_miniBZ_dq_cart(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> np.ndarray:
    """CENTROSYMMETRIC Monte-Carlo sample of the mini-BZ, Cartesian.

    Returns ``(2 * nmc, 3)`` offsets δq filling the Voronoi cell of the
    mini-BZ, as the union of an ``nmc``-point draw with its own negation.

    The centrosymmetry is LOAD-BEARING, not cosmetic.  The head injected
    at a slot is ``⟨v(K + δq)⟩``, and ``V_q = conj(V_{−q})`` requires the
    injected value to satisfy ``f(K) = f(−K)``.  Since
    ``⟨v(−K + δq)⟩ = ⟨v(K − δq)⟩``, that identity holds EXACTLY iff the
    finite sample set obeys ``{δq} = {−δq}``.  The historical one-sided
    draw (``rng.uniform(0, 1, ...)`` then Voronoi-wrapped) is symmetric
    only in the ``nmc → ∞`` limit, so it left an MC-noise-sized residual
    in reciprocity even once the slot selection was right.
    """
    from .vcoul import wrap_points_to_voronoi
    kgrid_arr = np.array([int(s) for s in kgrid], dtype=np.float64)
    bvec = np.asarray(bvec, dtype=np.float64)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (int(nmc), 3))
    # ``u @ bvec``, i.e. bvec ROWS are the lattice vectors — the convention
    # of ``randlims`` below and of ``q_frac @ bvec`` elsewhere in this
    # module, and identical to the sanctioned sampler
    # ``gw.coulomb.base.minibz_voronoi_batches`` (``(bvec.T @ u.T).T``).  The
    # pre-2026-08-08 line here was ``randvals @ bvec.T`` — the transpose,
    # i.e. bvec COLUMNS.  That is a fundamental domain of the row lattice
    # only for special cells.  MEASURED, folded-fractional coords of 200k
    # points (uniform on the Voronoi cell => mean 0.5, var 1/12 = 0.08333):
    #   fcc Si   old mean [.5002 .5005 .4998] var [.08338 .08312 .08323]
    #            new mean [.4998 .5002 .5005] var [.08323 .08338 .08312]
    #            -> both uniform, same distribution up to an axis permutation
    #   hexagonal old mean [.4994 .5427 .5005] var [.08342 .06334 .08312]
    #            -> NOT uniform: off-centre on b2, variance off by ~24%
    # Corrected here because this restructuring stands on the line; the
    # defect was latent (no 3D non-cubic fixture exists — every hexagonal
    # fixture is sys_dim=2, where this table is not built) and this is the
    # only behaviour change in this function beyond centrosymmetrisation.
    # On Si the head it produces shifts by ~1.5e-4 relative, which MEASURED
    # sits inside the seed-to-seed spread of the sampler itself.
    randcart = randvals @ bvec
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), bvec_j, nmax=1))
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    dq = (randlims @ wrapped.T).T          # (nmc, 3) mini-BZ offsets, Cartesian
    return np.concatenate([dq, -dq], axis=0)


def build_v_head_miniBZ_fn_3d(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Mini-BZ-averaged bare Coulomb head as a function of Cartesian ``q+G``.

    3D bulk only.  Returns ``head_fn(K_cart) -> (m,) float64``, where
    ``K_cart`` is ``(m, 3)`` Cartesian ``q+G`` and the value is
    ``⟨8π/|K + δq|²⟩_miniBZ / Ω_cell`` in Rydberg.  This is the
    ``v_head_fn`` argument of :func:`compute_v_q_per_G` /
    :func:`gw.coulomb.base.v_qG_table`, which evaluates it at every slot
    attaining ``argmin |q+G|²`` — see that function's HEAD SLOT note for
    why the selection is by argmin and why ALL of the argmin.

    This REPLACES the pre-2026-08-08 ``build_v_head_miniBZ_avg_3d``, which
    returned a ``(nkx, nky, nkz)`` table indexed by the BGW-wrapped q
    index.  That shape was unfixable: a per-q value is attached to
    ``q_frac @ bvec``, which is NOT the argmin ``q+G`` on 12 of Si's 64 q,
    and it carries no per-slot resolution — the caller must evaluate the
    head at each tied slot's own K before it can average over the tied set.

    ``f(K) = f(−K)`` holds to machine precision here because
    :func:`build_miniBZ_dq_cart` is centrosymmetric; Γ is not special-cased
    (the caller skips it — the q→0 head is the separate rank-1 Σ_X term).
    """
    dq_cart = build_miniBZ_dq_cart(kgrid, bvec, nmc=nmc, seed=seed)
    fact = 1.0 / float(cell_volume)

    def head_fn(K_cart: np.ndarray) -> np.ndarray:
        K = np.asarray(K_cart, dtype=np.float64).reshape(-1, 3)
        out = np.empty(K.shape[0], dtype=np.float64)
        for i in range(K.shape[0]):
            shifted = K[i][None, :] + dq_cart          # (2*nmc, 3)
            out[i] = np.mean(8.0 * np.pi / np.sum(shifted * shifted, axis=1))
        return out * fact

    return head_fn


def compute_v_q_per_G(
    q_irr_frac: np.ndarray,
    gvec_components: np.ndarray,
    *,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    vcoul_cutoff_ry: float | None = None,
    bdot: np.ndarray | None = None,
    fft_grid: np.ndarray | None = None,
    v_head_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute ``v(q+G)`` at the per-q WFN.h5-style G-list.

    Evaluates the bare (optionally 2D-truncated) Coulomb ``v(q+G)``
    directly on the per-q ``gvec_components`` table (rather than a full
    FFT grid) — the writer's ``isdf_header/gvec_components`` is exactly
    the input the G-flat V_q contract needs.  Returns one ``(ngkmax,)``
    row of ``v(q+G)`` per q in ``q_irr_frac``.

    Pad slots in ``gvec_components`` (sentinel Miller index
    ``(-nx/2, -ny/2, -nz/2)``) get whatever ``v`` is at that
    position — caller need not zero them because the contract uses
    ζ̃ = 0 at those slots.

    Parameters
    ----------
    q_irr_frac : ``(n_q, 3)`` float64
        Fractional q-vectors in BGW-wrap convention (already divided
        by kgrid).
    gvec_components : ``(n_q, 3, ngkmax)`` int32
        Per-q Miller indices from ``isdf_header.gvec_components``.
    bvec, cell_volume, sys_dim, vcoul_cutoff_ry, bdot, fft_grid
        Coulomb geometry / truncation.  ``vcoul_cutoff_ry`` zeroes
        ``v`` at G's with |q+G|² past the cutoff (== V_q's bare-Coulomb
        cutoff; may be < the ζ-sphere cutoff that built
        ``gvec_components``).  ``bdot`` / ``fft_grid`` are read only by
        the 0-D box kernel.

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

    Since 2026-08-06 this is a **thin dispatcher** over
    :func:`gw.coulomb.base.v_qG_table` — the arithmetic lives once per
    dimensionality in ``gw/coulomb/{bulk_3d,slab_2d,box_0d}.py`` and is
    shared with :meth:`gw.coulomb.base.CoulombKernel.v_qG`.  The
    dimension-independent capabilities (``vcoul_cutoff_ry`` masking, G=0
    head-slot injection, the ``(n_q, ngkmax)`` float64 contract) are
    implemented once in that driver.  ``sys_dim=0`` is now SERVED rather
    than refused — ``Box0D`` supplies the WS-box formula at q=0 — but the
    G-flat V_q pipeline still cannot reach it, because
    ``gw.isdf_fitting`` builds no per-q sphere for ``sys_dim == 0``
    (``isdf_fitting.py:670``) and ``compute_all_V_q_g_flat`` refuses
    ``sys_dim not in (2, 3)`` (``v_q_g_flat.py:527``).  Those two are the
    real, accurate gate; the refusal that used to live here was not.
    """
    if sys_dim not in (0, 2, 3):
        raise NotImplementedError(
            f"compute_v_q_per_G: sys_dim must be 0 / 2 / 3; got {sys_dim}")
    from .coulomb import get_kernel
    from .coulomb.base import v_qG_table
    return v_qG_table(
        get_kernel(sys_dim), q_irr_frac, gvec_components,
        bvec=bvec, cell_volume=cell_volume,
        vcoul_cutoff_ry=vcoul_cutoff_ry,
        v_head_fn=v_head_fn,
        bdot=bdot, fft_grid=fft_grid,
    )


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
