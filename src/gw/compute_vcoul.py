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
  injected into ``compute_v_q_per_G`` for bulk systems.  It LIVES in
  ``vcoul.minibz`` and is re-exported here (production
  ``gw.v_q_g_flat`` imports it from the door since the 2026-08-07
  replumb; this path remains for the head-draw guard and sibling
  branches).
* :func:`compute_all_V_q` — the thin dispatcher: for a G-flat on-disk ζ
  it hands off to :func:`gw.v_q_g_flat.compute_all_V_q_g_flat`; any
  other layout raises ``NotImplementedError``.

The actual V_q contract (``V_q[μ,ν] = Σ_G conj(ζ̃_μ) v(q+G) ζ̃_ν``,
G-chunked) lives in ``gw.v_q_g_flat`` / ``gw.v_q_bispinor``; nothing in
this file does FFTs or μ × ν tiling any more.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import h5py
import numpy as np
import jax
from jax.sharding import Mesh

from file_io.paths import resolve_input_path
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

#: The 3D mini-BZ body head.  Lives in :func:`vcoul.minibz` since
#: 2026-08-07 and is re-exported here VERBATIM — including the seed=42 /
#: nmc=2**18 defaults, which the ``si_bse_debug`` frozen reference pins.
#: ``tests/test_vcoul_minibz_head_draw.py`` imports it from THIS path
#: (the head-draw guard doubles as shim coverage); ``gw.v_q_g_flat``
#: imports the door since the 2026-08-07 replumb.
from vcoul import (                                         # noqa: E402,F401
    build_miniBZ_dq_cart,
    build_v_head_miniBZ_fn_3d,
)


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

    Since 2026-08-06 this is a **thin dispatcher** over the single
    ``v(q+G)`` driver — since the 2026-08-07 audit-arm consensus it goes
    straight to :func:`vcoul.v_qG_table`; the arithmetic lives once per
    dimensionality in ``vcoul.{bulk_3d,slab_2d,box_0d}`` and is
    shared with :meth:`CoulombKernel.v_qG`.  The
    dimension-independent capabilities (``vcoul_cutoff_ry`` masking, the
    ``argmin |q+G|`` head-slot injection, the ``(n_q, ngkmax)`` float64
    contract) are implemented once in that driver.  ``sys_dim=0`` is now SERVED rather
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
    # Straight to the door (audit-arm consensus, 2026-08-07): the loose
    # arguments here ARE the CoulombGeometry fields, so hopping through
    # the gw.coulomb shim adapter was a redundant double translation —
    # and the only src/ site that reached a shim submodule.
    from vcoul import CoulombGeometry, get_kernel, v_qG_table
    return v_qG_table(
        get_kernel(sys_dim), q_irr_frac, gvec_components,
        geometry=CoulombGeometry(bvec=bvec, cell_volume=cell_volume,
                                 bdot=bdot, fft_grid=fft_grid),
        vcoul_cutoff_ry=vcoul_cutoff_ry,
        v_head_fn=v_head_fn,
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
    bgw_v_sphere_fn=None,
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
            bgw_v_sphere_fn=bgw_v_sphere_fn,
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


@dataclass(frozen=True)
class BGWMetalVcoulSource:
    """Validated production BGW Coulomb source.

    ``v_sphere_fn`` maps an explicit q/G sphere to LORRAX's ``v/Omega``
    convention.  ``q0_vcoul_raw`` remains pre-``1/(Nq*Omega)`` because the
    scalar head channel applies that factor at its existing reduction seam.
    """

    v_sphere_fn: Callable[[tuple, np.ndarray], np.ndarray]
    q0_vcoul_raw: float
    path: str
    sha256: str | None
    n_q: int
    n_G: int


def build_bgw_metal_vcoul_source(
    config, *, wfn, sym, input_dir: str, print_fn=print,
) -> BGWMetalVcoulSource | None:
    """Read, validate, and bind ``bgw_metal_vcoul_file`` to this WFN.

    The file is BGW's formatted, raw-Ry ``write_vcoul`` output.  Validation
    is intentionally eager: q representatives must match the WFN grid, every
    q's integer G sphere must match ``bare_coulomb_cutoff``, and all regular
    values must agree with the WFN reciprocal metric in BGW's 8*pi Ry
    convention.  The only exempt value is q=0,G=0, which BGW cell-averages.
    """
    requested = getattr(config.head, "bgw_metal_vcoul_file", None)
    if not requested:
        return None

    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from vcoul import (
        CoulombGeometry,
        fill_v_sphere_for_q,
        read_bgw_vcoul,
    )

    bgw_path = resolve_input_path(input_dir, requested)
    table = read_bgw_vcoul(
        bgw_path, compute_sha256=(jax.process_index() == 0))
    kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
    if kgrid.shape != (3,) or np.any(kgrid <= 0):
        raise ValueError(
            "bgw_metal_vcoul_file requires a positive 3-D WFN q grid; "
            f"found {tuple(np.asarray(wfn.kgrid).reshape(-1))}.")

    q_int = np.asarray(sym.q_irr_kgrid_int, dtype=np.int64)
    q_wrapped = np.where(
        q_int > kgrid[None, :] / 2,
        q_int - kgrid[None, :], q_int).astype(np.float64)
    q_expected = q_wrapped / kgrid[None, :]
    if int(table.q_fracs.shape[0]) != int(q_expected.shape[0]):
        raise ValueError(
            "bgw_metal_vcoul_file q-grid mismatch: BGW's first sigma q walk "
            f"contains {int(table.q_fracs.shape[0])} q points, but the WFN "
            f"symmetry/q grid requires {int(q_expected.shape[0])} irreducible "
            f"representatives on {tuple(int(x) for x in kgrid)}.")
    for q in q_expected:
        try:
            table.find_q_index(q, sym_mats_k=None)
        except ValueError as exc:
            raise ValueError(
                "bgw_metal_vcoul_file q-grid mismatch: no BGW first-walk "
                f"representative matches WFN q={tuple(float(x) for x in q)} "
                f"on grid {tuple(int(x) for x in kgrid)}.") from exc

    cutoff = (
        float(wfn.ecutwfc) if config.head.bare_coulomb_cutoff is None
        else float(config.head.bare_coulomb_cutoff))
    bvec = CoulombGeometry.from_wfn(wfn).bvec
    sphere = compute_per_q_bare_coulomb_components(
        tuple(int(x) for x in wfn.fft_grid), bvec, q_expected, cutoff)
    components = sphere["gvec_components_padded"]
    ngk = np.asarray(sphere["ngk_per_q"], dtype=np.int64)
    cell_volume = float(wfn.cell_volume)
    sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)

    # Exact set and unit/metric gate.  BGW prints eight digits in E format;
    # 8e-8 relative comfortably covers last-digit formatting without hiding
    # a Hartree/Rydberg factor-of-two or a different lattice/cutoff.
    for iq, q in enumerate(q_expected):
        G = np.asarray(components[iq, :, :int(ngk[iq])].T, dtype=np.int32)
        scaled = fill_v_sphere_for_q(
            table, q, G, cell_volume,
            sym_mats_k=sym_mats_k, require_exact_sphere=True)
        K = (q[None, :] + G.astype(np.float64)) @ bvec
        norm2 = np.einsum("gi,gi->g", K, K)
        regular = norm2 > 1.0e-13
        reference_raw = 8.0 * np.pi / norm2[regular]
        actual_raw = scaled[regular] * cell_volume
        if not np.allclose(
                actual_raw, reference_raw, rtol=8.0e-8, atol=2.0e-8):
            bad = int(np.flatnonzero(~np.isclose(
                actual_raw, reference_raw, rtol=8.0e-8,
                atol=2.0e-8))[0])
            raise ValueError(
                "bgw_metal_vcoul_file WFN/unit mismatch: a regular value at "
                f"q={tuple(float(x) for x in q)} differs from BGW's Rydberg "
                f"8*pi/|q+G|^2 convention (file={actual_raw[bad]:.12e}, "
                f"WFN={reference_raw[bad]:.12e}). Check that BGW and LORRAX "
                "use the same WFN, lattice, q grid, and bare_coulomb_cutoff.")

    q0_raw = table.q0_vcoul_raw()
    counts = [len(g) for g in table.G_miller_per_q]
    print_fn(
        "  [bgw metal vcoul provenance] "
        f"file={bgw_path} sha256={table.sha256 or '<rank0-only>'} "
        f"n_q={int(table.q_fracs.shape[0])} n_G={table.n_G} "
        f"(per-q {min(counts)}..{max(counts)}), qgrid="
        f"{tuple(int(x) for x in kgrid)}, cutoff={cutoff:.8g} Ry; "
        f"raw q0,G0={q0_raw:.12e}. BGW values replace bare v in Sigma_x "
        "and W; bgw_q0shift, if selected, still supplies finite-q0 epsinv.")

    def v_sphere_fn(q_frac_tuple, g_miller):
        # ζ may use a sphere larger than the bare-Coulomb cutoff.  Match the
        # transformed BGW integer-G set into its padded row explicitly;
        # unmatched ζ slots remain zero and no numeric vcoul value doubles
        # as a "missing" marker.
        requested_G = np.asarray(g_miller, dtype=np.int32)
        iq, S_k, kg0 = table.find_q_index(
            q_frac_tuple, sym_mats_k=sym_mats_k)
        stored_G = np.asarray(table.G_miller_per_q[iq], dtype=np.int32)
        expected_G = np.einsum(
            "ij,gj->gi", S_k.astype(np.int32), stored_G) - kg0[None, :]
        requested_positions: dict[tuple[int, int, int], list[int]] = {}
        for pos, G in enumerate(requested_G):
            requested_positions.setdefault(
                tuple(int(x) for x in G), []).append(pos)
        missing = []
        duplicate = []
        positions = []
        for G in expected_G:
            key = tuple(int(x) for x in G)
            matches = requested_positions.get(key, [])
            if not matches:
                missing.append(key)
            elif len(matches) != 1:
                duplicate.append((key, len(matches)))
            else:
                positions.append(matches[0])
        if missing or duplicate:
            raise ValueError(
                "bgw_metal_vcoul_file G-sphere/cutoff mismatch: the ζ "
                f"sphere for q={tuple(q_frac_tuple)} is missing "
                f"{len(missing)} BGW G rows and duplicates {len(duplicate)}; "
                f"first missing={missing[:3]}, first duplicates={duplicate[:3]}.")
        positions_arr = np.asarray(positions, dtype=np.int64)
        logical = fill_v_sphere_for_q(
            table, q_frac_tuple, requested_G[positions_arr], cell_volume,
            sym_mats_k=sym_mats_k, require_exact_sphere=True)
        # LORRAX carries the singular Gamma head as a separate rank-one
        # scalar channel.  Preserve that decomposition: the file value is
        # consumed through ``q0_vcoul_raw`` below, not duplicated in V_q.
        q_bz = np.mod(np.asarray(q_frac_tuple, dtype=np.float64) + 0.5, 1.0) - 0.5
        if np.all(np.abs(q_bz) < 1.0e-10):
            g0 = np.flatnonzero(np.all(
                requested_G[positions_arr] == 0, axis=1))
            if g0.size != 1:
                raise ValueError(
                    "bgw_metal_vcoul_file q=0 sphere must map exactly one "
                    f"G=(0,0,0) slot; found {int(g0.size)}.")
            logical[int(g0[0])] = 0.0
        out = np.zeros(requested_G.shape[0], dtype=np.float64)
        out[positions_arr] = logical
        return out

    return BGWMetalVcoulSource(
        v_sphere_fn=v_sphere_fn,
        q0_vcoul_raw=q0_raw,
        path=bgw_path,
        sha256=table.sha256,
        n_q=int(table.q_fracs.shape[0]),
        n_G=table.n_G,
    )

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
