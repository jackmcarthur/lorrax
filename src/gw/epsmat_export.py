"""LORRAX screened interaction <-> BerkeleyGW ε⁻¹(G,G'); see docs/theory/bgw_epsmat_export.md.

Pair densities and χ are EXPANDED in ζ, V and W are PROJECTED on it
(``Z[G, μ] = ζ̃_qμ(G)``, ``V = Zᴴ diag(v) Z``, ``W = Zᴴ W(G,G') Z``), so

    ε⁻¹_q(G,G') − δ = v_q(G) [Z χ Zᴴ](G,G'),   χ = V⁺ (W − V) V⁺        (E4)

evaluated as ``diag(v) Y K Yᴴ`` with ``B Bᴴ = V⁺`` (``isdf.cplus.factor``),
``Y = Z B`` and the whitened correlation ``K = Bᴴ (W − V) B`` (−1 ≼ K ≼ 0 at
ω = 0).  The reverse is ``W = V + Z_εᴴ [(ε⁻¹ − 1) diag(v)] Z_ε`` on the file's
own G list.  One q at a time: ζ̃_q arrives μ-sharded, is gathered onto the
BerkeleyGW G list and moved G-sharded by one all-to-all; each device builds
its ``P('x','y')`` tile of the ``(G, G')`` block from the x- and y-row blocks
of Y (one all-gather over ``y`` and one transpose-partner permute).  V, W, B
and K are ``N_μ²`` and REPLICATED on every device, the charge-ζ factor's
plan (``isdf.cplus``): 16 N_μ² B each (15 MB at N_μ = 978).  The block is
the only ``N_G²`` object and never exists outside its tiles.
"""
from __future__ import annotations

import functools
import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

__all__ = [
    "match_q_umklapp", "zeta_slots", "q0_head", "coulomb_whitener",
    "whitened_correlation", "epsinv_block", "project_block", "coulomb_gram",
    "run_epsmat_export", "run_epsmat_projection",
]

_XY = ("x", "y")
#: GATE thresholds.  V from ζ and v is an identity of the producer's own
#: arithmetic (measured 1e-14 on CrI3 978); the block projection is an
#: identity of this module's assembly (measured 1e-14 on the same bundle).
V_RECONSTRUCTION_TOL = 1.0e-9
BLOCK_PROJECTION_TOL = 1.0e-8
STATIC_W_HERMITIAN_TOL = 1.0e-8


def _refuse(gate, got, want, why, fix):
    """The standard refusal: rule id, got, want, consequence, fix."""
    raise ValueError(f"GATE {gate}: got {got}; want {want}; why: {why}; fix: {fix}")


# ---------------------------------------------------------------------------
# Host tables
# ---------------------------------------------------------------------------

def match_q_umklapp(q_stored, q_targets, *, tol=1e-6):
    """Stored row and umklapp of each target q: ``q_t = q_stored[i] + G0``.

    Parameters
    ----------
    q_stored : (n_s, 3) float, crystal
    q_targets : (n_t, 3) float, crystal

    Returns
    -------
    idx : (n_t,) int, -1 where no stored q matches by a lattice vector
    g0 : (n_t, 3) int
    """
    qs = np.asarray(q_stored, dtype=np.float64).reshape(-1, 3)
    qt = np.asarray(q_targets, dtype=np.float64).reshape(-1, 3)
    d = qt[:, None, :] - qs[None, :, :]
    hit = np.all(np.abs(d - np.rint(d)) < tol, axis=-1)
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    g0 = np.rint(d[np.arange(qt.shape[0]), np.maximum(idx, 0)]).astype(np.int64)
    return idx, g0


def zeta_slots(sphere_miller, ngk, miller, *, fill):
    """ζ-sphere slot of each Miller row; ``fill`` where the row is outside it.

    ``sphere_miller`` is the stored per-q sphere ``(ngkmax, 3)`` of which the
    first ``ngk`` rows are physical.  Returns ``(n,)`` int32.
    """
    s = np.asarray(sphere_miller, dtype=np.int64)[: int(ngk)]
    m = np.asarray(miller, dtype=np.int64).reshape(-1, 3)
    lo = np.minimum(s.min(0), m.min(0))
    span = np.maximum(s.max(0), m.max(0)) - lo + 1

    def key(a):
        a = a - lo
        return (a[:, 0] * span[1] + a[:, 1]) * span[2] + a[:, 2]

    ks = key(s)
    order = np.argsort(ks)
    km = key(m)
    pos = np.clip(np.searchsorted(ks[order], km), 0, ks.size - 1)
    found = ks[order][pos] == km
    return np.where(found, order[pos], int(fill)).astype(np.int32)


def q0_head(S_cart, q0_cart, v_q0_ry):
    """ε⁻¹_00(q0) = 1 / (1 − v(q0) q0ᵀ S q0), the direct (no-local-field) head.

    ``S_cart`` (3, 3) is the Cartesian q² coefficient of χ₀₀ in
    1/(Ry bohr²) (docs/theory/s-tensor-convention.md), ``q0_cart`` (3,) in
    1/bohr, ``v_q0_ry`` the truncated Coulomb at q0 over Ω, in Ry.
    """
    q = np.asarray(q0_cart, dtype=np.float64)
    qsq = complex(np.einsum("i,ij,j->", q, np.asarray(S_cart, dtype=np.complex128), q))
    return 1.0 / (1.0 - float(v_q0_ry) * qsq)


# ---------------------------------------------------------------------------
# μ-space algebra (replicated N_μ²)
# ---------------------------------------------------------------------------

def coulomb_whitener(V_log, *, rcond):
    """``(B, figures)`` with ``B Bᴴ = V⁺`` on the retained Coulomb span.

    ``V_log`` ``(n, n)`` is the logical, Hermitian-PSD V_q (Ry).  The cut keeps
    ``λ > rcond λ_max`` closed over degenerate multiplets and certified by
    ``isdf.cplus.factor``; dropped modes are zero columns of B, so shapes do
    not depend on the rank.  Columns of ``U = diag(v)^{1/2} Z B`` are
    orthonormal in G space.
    """
    from isdf import cplus
    B, figures = cplus.factor(V_log[None], rcond=float(rcond), rank_log=True,
                              n_log=int(V_log.shape[-1]), where="epsmat V+")
    return B[0], figures


def whitened_correlation(Wc_log, B):
    """K = Bᴴ (W − V) B, Hermitian part: the symmetrized response ``v^{1/2} χ v^{1/2}``
    in the orthonormal basis of the Coulomb span (−1 ≼ K ≼ 0 at ω = 0, W 9)."""
    K = jnp.conj(B).T @ Wc_log @ B
    return 0.5 * (K + jnp.conj(K).T)


# ---------------------------------------------------------------------------
# Sharded per-q programs (built once per static signature)
# ---------------------------------------------------------------------------

def _square_side(mesh):
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if px != py:
        _refuse("epsmat_square_mesh", f"mesh {px}x{py}", "a square mesh",
                "the column block of a P('x','y') tile is the transpose partner's row block",
                "run on 1, 4, 9, 16, ... devices")
    return px


def _transpose_perm(p):
    """ppermute pairs over the flattened (x, y) axis: device (y, x) -> (x, y)."""
    return [(y * p + x, x * p + y) for x in range(p) for y in range(p)]


def _row_blocks(rows, p):
    """``(rows of the x block, rows of the y block)`` of a row-chunked matrix.

    Inside ``shard_map``: chunk ``c = x p + y`` holds rows ``[c n/P, (c+1) n/P)``,
    so the all-gather over ``y`` is the x row block, and the y row block is
    the x row block of the transpose partner.
    """
    rx = jax.lax.all_gather(rows, "y", axis=0, tiled=True)
    ry = rx if p == 1 else jax.lax.ppermute(rx, _XY, _transpose_perm(p))
    return rx, ry


def _to_g_rows(zt_local):
    """μ-sharded ``(μ/P, n)`` -> G-row chunk ``(n/P, μ)`` by one all-to-all."""
    zc = jax.lax.all_to_all(zt_local, _XY, 1, 0, tiled=True)
    return zc.T


def _tile_indices(n_tile):
    a = jax.lax.axis_index("x") * n_tile + jnp.arange(n_tile)
    b = jax.lax.axis_index("y") * n_tile + jnp.arange(n_tile)
    return a, b


@functools.lru_cache(maxsize=None)
def _epsinv_program(mesh, n_pad, mu_pad, ng_disk, is_q0):
    """jit: (ζ̃ (1, μ, ngkmax) μ-sharded, slots, B, K, v, n_valid, head) ->
    (block (n, n, 2) at P('x','y',None), diagonal (n,) at P(('x','y')))."""
    p = _square_side(mesh)
    n_tile = n_pad // p
    zsh = NamedSharding(mesh, P(None, _XY, None))
    rep = NamedSharding(mesh, P())

    def tile(zt, B, K, v_x, v_y, v_c, n_valid, head):
        y = _to_g_rows(zt) @ B                               # (n/P, μ) Y = Z B
        yx, yy = _row_blocks(y, p)
        # T[a, b] = ε⁻¹(G_b, G_a) − δ = v_b (Y K Yᴴ)[b, a]   (BGW block orientation)
        t = (jnp.conj(yx) @ K.T) @ yy.T * v_y[None, :]
        a, b = _tile_indices(n_tile)
        eye = (a[:, None] == b[None, :]) & (a[:, None] < n_valid)
        t = t + eye.astype(t.dtype)
        d = 1.0 + v_c * jnp.sum((y @ K) * jnp.conj(y), axis=1)
        if is_q0:
            # q -> 0: wings zero, head from the long-wavelength model.
            wing = (a[:, None] == 0) | (b[None, :] == 0)
            t = jnp.where(wing, 0.0, t)
            t = jnp.where((a[:, None] == 0) & (b[None, :] == 0), head, t)
            c0 = (jax.lax.axis_index("x") * p + jax.lax.axis_index("y")) == 0
            d = d.at[0].set(jnp.where(c0, head, d[0]))
        return jnp.stack([t.real, t.imag], axis=-1), d

    smap = shard_map(
        tile, mesh=mesh,
        in_specs=(P(_XY, None), P(), P(), P("x"), P("y"), P(_XY), P(), P()),
        out_specs=(P("x", "y", None), P(_XY)), check_vma=False)

    def run(zeta, slots, B, K, v, n_valid, head):
        z = jnp.pad(zeta[0], ((0, 0), (0, 1)))               # slot ng_disk reads zero
        z = jnp.take(z, slots, axis=1)                        # (μ, n) μ-sharded, local
        return smap(z, B, K, v, v, v, n_valid, head)

    return jax.jit(run, in_shardings=(zsh, rep, rep, rep, rep, rep, rep),
                   out_shardings=(NamedSharding(mesh, P("x", "y", None)),
                                  NamedSharding(mesh, P(_XY))))


@functools.lru_cache(maxsize=None)
def _projection_program(mesh, n_pad, mu_pad, ng_disk, drop_head):
    """jit: (block (1,1,1,n,n,2) BGW orientation, ζ̃, slots, v, n_valid) ->
    W_c = Z_εᴴ [(ε⁻¹ − 1) diag(v)] Z_ε, (μ, μ) replicated."""
    p = _square_side(mesh)
    n_tile = n_pad // p
    zsh = NamedSharding(mesh, P(None, _XY, None))
    bsh = NamedSharding(mesh, P(None, None, None, "x", "y", None))
    rep = NamedSharding(mesh, P())

    def tile(blk, zt, v_x, n_valid):
        z = _to_g_rows(zt)                                    # (n/P, μ)
        zx, zy = _row_blocks(z, p)
        a, b = _tile_indices(n_tile)
        live = (a[:, None] < n_valid) & (b[None, :] < n_valid)
        t = jnp.where(live, blk[..., 0] + 1j * blk[..., 1], 0.0)
        t = t - (a[:, None] == b[None, :]).astype(t.dtype) * live
        if drop_head:
            t = jnp.where((a[:, None] == 0) | (b[None, :] == 0), 0.0, t)
        # Σ_ab conj(Z[G_b, μ]) T[a, b] v_a Z[G_a, ν]
        part = jnp.conj(zy).T @ t.T @ (v_x[:, None] * zx)
        return jax.lax.psum(part, _XY)

    smap = shard_map(tile, mesh=mesh,
                     in_specs=(P("x", "y", None), P(_XY, None), P("x"), P()),
                     out_specs=P(), check_vma=False)

    def run(block, zeta, slots, v, n_valid):
        z = jnp.pad(zeta[0], ((0, 0), (0, 1)))
        z = jnp.take(z, slots, axis=1)
        return smap(block[0, 0, 0], z, v, n_valid)

    return jax.jit(run, in_shardings=(bsh, zsh, rep, rep, rep), out_shardings=rep)


@functools.lru_cache(maxsize=None)
def _gram_program(mesh, n_pad, mu_pad, ng_disk):
    """jit: (ζ̃ (1, μ, ngkmax) μ-sharded, slots (n,), v (n,)) ->
    ``Z_sᴴ diag(v) Z_s`` on the gathered G list, (μ, μ) replicated."""
    zsh = NamedSharding(mesh, P(None, _XY, None))
    rep = NamedSharding(mesh, P())

    def tile(zt, v_c):
        z = _to_g_rows(zt)
        return jax.lax.psum((jnp.conj(z) * v_c[:, None]).T @ z, _XY)

    smap = shard_map(tile, mesh=mesh, in_specs=(P(_XY, None), P(_XY)),
                     out_specs=P(), check_vma=False)

    def run(zeta, slots, v):
        z = jnp.pad(zeta[0], ((0, 0), (0, 1)))
        return smap(jnp.take(z, slots, axis=1), v)

    return jax.jit(run, in_shardings=(zsh, rep, rep), out_shardings=rep)


def _pad_to(n, m):
    return -(-int(n) // int(m)) * int(m)


def epsinv_block(zeta_q, slots, B, K, v_eps, *, mesh, n_valid, q0_head_value=None):
    """ε⁻¹_q(G,G') on one BerkeleyGW G list, in the file's block orientation.

    Parameters
    ----------
    zeta_q : (1, μ_pad, ngkmax) complex128, ``P(None, ('x','y'), None)``
        ζ̃ on the stored per-q sphere (``ZetaLoader.read_zeta_G_slab``).
    slots : (n,) int
        ζ-sphere slot of each eps-ordered G (``ngkmax`` = outside the sphere).
    B, K : (μ_pad, μ_pad) complex128, replicated
        Whitener and whitened correlation (:func:`coulomb_whitener`).
    v_eps : (n,) float
        LORRAX v(q+G)/Ω in Ry at the eps G list (the V_q values).
    n_valid : int
        nmtx; rows/columns past it are zero (mesh pad).
    q0_head_value : complex | None
        For the q → 0 file: the head ε⁻¹_00(q0); wings are written zero.

    Returns
    -------
    block : (n_pad, n_pad, 2) float64 at ``P('x','y',None)``, ``[a, b]`` =
        (Re, Im) ε⁻¹(G_b, G_a); diagonal : (n_pad,) complex at ``P(('x','y'))``.
    """
    p = _square_side(mesh)
    mu_pad, ngk = int(zeta_q.shape[1]), int(zeta_q.shape[2])
    n_pad = _pad_to(max(int(n_valid), 1), p * p)
    sl = np.full(n_pad, ngk, dtype=np.int32)
    sl[: len(slots)] = np.asarray(slots, dtype=np.int32)
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[: len(v_eps)] = np.asarray(v_eps, dtype=np.float64)
    prog = _epsinv_program(mesh, n_pad, mu_pad, ngk, q0_head_value is not None)
    head = 1.0 + 0.0j if q0_head_value is None else complex(q0_head_value)
    return prog(zeta_q, jnp.asarray(sl), B, K, jnp.asarray(vv),
                jnp.int32(n_valid), jnp.complex128(head))


def project_block(block, zeta_q, slots, v_eps, *, mesh, n_valid, drop_head):
    """W_c(q) = Z_εᴴ [(ε⁻¹ − 1) diag(v)] Z_ε from one block in file orientation.

    ``block`` ``(1, 1, 1, n_pad, n_pad, 2)`` at
    ``P(None, None, None, 'x', 'y', None)``; ``drop_head`` removes the G = 0
    row and column (the q → 0 head and wings, which LORRAX carries outside
    the body).  Returns ``(μ_pad, μ_pad)`` complex128, replicated, Ry.
    """
    p = _square_side(mesh)
    mu_pad, ngk = int(zeta_q.shape[1]), int(zeta_q.shape[2])
    n_pad = int(block.shape[3])
    if n_pad % (p * p):
        _refuse("epsmat_block_extent", f"block extent {n_pad}",
                f"a multiple of {p * p}", "the G axis is chunked over every device",
                "read the block through bgw_epsmat.matrix_block_window")
    sl = np.full(n_pad, ngk, dtype=np.int32)
    sl[: len(slots)] = np.asarray(slots, dtype=np.int32)
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[: len(v_eps)] = np.asarray(v_eps, dtype=np.float64)
    prog = _projection_program(mesh, n_pad, mu_pad, ngk, bool(drop_head))
    return prog(block, zeta_q, jnp.asarray(sl), jnp.asarray(vv), jnp.int32(n_valid))


def coulomb_gram(zeta_q, slots, v, *, mesh):
    """``Z_sᴴ diag(v) Z_s`` over the G rows ``slots`` of the stored sphere, replicated.

    With every physical slot this is the bundle's own V_q (the
    normalization gate); with the eps G list it is ``V_ε``.
    """
    p = _square_side(mesh)
    mu_pad, ngk = int(zeta_q.shape[1]), int(zeta_q.shape[2])
    n = len(slots)
    n_pad = _pad_to(max(n, 1), p * p)
    sl = np.full(n_pad, ngk, dtype=np.int32)
    sl[:n] = np.asarray(slots, dtype=np.int32)
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[:n] = np.asarray(v, dtype=np.float64)
    return _gram_program(mesh, n_pad, mu_pad, ngk)(zeta_q, jnp.asarray(sl), jnp.asarray(vv))


# ---------------------------------------------------------------------------
# The bundle side: stored q, V/W rows, ζ, v
# ---------------------------------------------------------------------------

class ExportSource(NamedTuple):
    """Everything one export or projection reads, opened once (host facts + handles)."""
    bundle: str
    zeta: object                 # zeta_loader.ZetaLoader on the run mesh
    q_stored: np.ndarray         # (n_q, 3) crystal, BGW wrap
    kgrid: tuple
    n_mu: int
    mu_pad: int
    geometry: object             # vcoul.CoulombGeometry
    config: object               # gw_config.LorraxConfig of the producer deck
    head: dict                   # S_cart, G0, whead, head_correction label
    wfn_file: str


def _bundle_file(source_restart):
    """The ``isdf_tensors_<mu>.h5`` of a run directory, or the file itself."""
    if os.path.isfile(source_restart):
        return os.path.abspath(source_restart)
    import glob
    hits = sorted(glob.glob(os.path.join(source_restart, "tmp", "isdf_tensors_*.h5")))
    if len(hits) != 1:
        _refuse("epsmat_source_restart", f"{len(hits)} bundles under {source_restart}/tmp",
                "exactly one isdf_tensors_<mu>.h5",
                "the export reads one ISDF basis", "name the bundle file itself")
    return hits[0]


def open_export_source(cfg, mesh, *, print_fn=print):
    """Open the producer's bundle, ζ and Coulomb geometry; refuse unsupported decks."""
    import h5py
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry
    from wfn_loader import WfnLoader
    from zeta_loader import ZetaLoader
    from symmetry_maps import bgw_integer_q_to_fractional
    from runtime.padding import padded_mu_axis
    from .gw_config import LorraxConfig

    bundle = _bundle_file(cfg.source_restart)
    zeta_path = cfg.zeta_file or os.path.join(os.path.dirname(bundle), "zeta_q.h5")
    gw = LorraxConfig.from_input_file(cfg.parent_input_file, print_fn=lambda *a, **k: None)
    if gw.head.use_bgw_vcoul or gw.head.mc_average_placement != "off":
        _refuse("epsmat_coulomb_policy",
                f"use_bgw_vcoul={gw.head.use_bgw_vcoul}, mc_average_placement="
                f"{gw.head.mc_average_placement}",
                "LORRAX's own v(q+G) carried by V_q",
                "the export rebuilds v with gw.compute_vcoul and gates V on it",
                "export a run without the BGW vcoul overlay or head placement")
    wfn = WfnLoader(gw.paths.wfn_file)
    geometry = CoulombGeometry.from_wfn(wfn)
    with h5py.File(bundle, "r") as f:
        kgrid = tuple(int(k) for k in np.asarray(f["kgrid"][()]))
        if "W0_qmunu" not in f or not bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            _refuse("epsmat_requires_w0", f"{bundle} without a ready W0_qmunu",
                    "a GW bundle with persisted static W", "ε⁻¹ is built from W(0)",
                    "rerun the producer with write_restart_tensors = true")
        storage = str(f["V_qmunu"].attrs.get("q_storage", "full"))
        q_stored = (np.asarray(f["V_qmunu__qirr/q_irr_frac"][()]) if storage == "ibz"
                    else bgw_integer_q_to_fractional(
                        wfn.symmetry().kvecs_asints, np.asarray(kgrid)))
        n_mu = int(f["V_qmunu"].shape[-1])
        head = {"S_cart": (np.asarray(f["S_cart_head"][()]) if "S_cart_head" in f else None),
                "G0": (np.asarray(f["G0_mu_nu"][()]) if "G0_mu_nu" in f else None),
                "whead": (np.asarray(f["whead"][()]) if "whead" in f else None),
                "head_correction": (str(f["whead"].attrs.get("head_correction", ""))
                                    if "whead" in f else str(gw.head.correction.value))}
        if storage == "ibz" and str(f["W0_qmunu"].attrs.get("q_storage", "full")) != "ibz":
            _refuse("epsmat_q_storage", "V on the wedge and W0 on the full zone",
                    "one stored q set", "V and W rows are read pairwise",
                    "regenerate the bundle")
    zl = ZetaLoader(zeta_path, mesh=mesh)
    if int(zl.n_q_on_disk) != q_stored.shape[0] or int(zl.n_rmu) != n_mu:
        _refuse("epsmat_zeta_bundle_pair", f"ζ (q {zl.n_q_on_disk}, μ {zl.n_rmu})",
                f"bundle (q {q_stored.shape[0]}, μ {n_mu})",
                "V and W are projected on this ζ", "point zeta_file at the bundle's own ζ")
    mu_pad = int(padded_mu_axis(n_mu, mesh).carrier)
    print_fn(f"  bundle      {bundle}\n  zeta        {zeta_path}\n"
             f"  stored q    {q_stored.shape[0]} ({storage}) on k-grid {kgrid}, "
             f"μ {n_mu} (carrier {mu_pad})\n  head        {head['head_correction']}")
    return ExportSource(bundle=bundle, zeta=zl, q_stored=q_stored, kgrid=kgrid,
                        n_mu=n_mu, mu_pad=mu_pad, geometry=geometry, config=gw,
                        head=head, wfn_file=gw.paths.wfn_file)


def lorrax_v(src, q_frac, miller):
    """LORRAX v(q+G)/Ω in Ry at Miller rows ``(n, 3)``: the values V_q carries.

    One owner (``gw.compute_vcoul.compute_v_q_per_G`` with
    ``gw.v_q_g_flat.v_head_fn_in_V``): truncation, the bare-Coulomb cutoff,
    the q+G = 0 zero and the 3D mini-BZ head slot.
    """
    from .compute_vcoul import compute_v_q_per_G
    from .v_q_g_flat import v_head_fn_in_V
    g = src.geometry
    head_fn = v_head_fn_in_V(
        mc_average_vcoul_body=src.config.head.mc_average_vcoul_body,
        sys_dim=src.config.sys_dim, kgrid=src.kgrid, bvec=g.bvec,
        cell_volume=g.cell_volume)
    return compute_v_q_per_G(
        np.asarray(q_frac, dtype=np.float64)[None],
        np.asarray(miller, dtype=np.int32).T[None],
        bvec=g.bvec, cell_volume=g.cell_volume, sys_dim=int(src.config.sys_dim),
        vcoul_cutoff_ry=src.config.head.bare_coulomb_cutoff, bdot=g.bdot,
        v_head_fn=head_fn)[0]


def bare_v_bgw_units(src, q_frac, miller):
    """BerkeleyGW's ``vcoul`` column: the exact truncated 8π/|q+G|² in Ry bohr³."""
    from .compute_vcoul import compute_v_q_per_G
    g = src.geometry
    v = compute_v_q_per_G(
        np.asarray(q_frac, dtype=np.float64)[None],
        np.asarray(miller, dtype=np.int32).T[None],
        bvec=g.bvec, cell_volume=g.cell_volume, sys_dim=int(src.config.sys_dim),
        vcoul_cutoff_ry=None, bdot=g.bdot, v_head_fn=None)[0]
    return v * g.cell_volume


def read_q_rows(src, iq, mesh):
    """``(V, W, ζ̃)`` of stored q ``iq``: V/W ``(1, μ_pad, μ_pad)`` at
    ``P(None,'x','y')`` (zero-padded), ζ̃ ``(1, μ_pad, ngkmax)`` μ-sharded."""
    from file_io.slab_io import SlabIO
    shape = (1, src.mu_pad, src.mu_pad)
    with SlabIO(src.bundle, mode="r", mesh=mesh) as io:
        V, W = (io.read_slab(name, shape=shape, dtype=np.complex128,
                             offset=(int(iq), 0, 0), mesh=mesh,
                             partition_spec=P(None, "x", "y"))
                for name in ("V_qmunu", "W0_qmunu"))
    Z = src.zeta.read_zeta_G_slab(q_offset=int(iq), q_count=1, mu_offset=0,
                                  mu_count=src.mu_pad, mesh=mesh)
    return V, W, Z


@functools.lru_cache(maxsize=None)
def _replicate_logical(mesh, n):
    """jit: a stored ``(1, μ_pad, μ_pad)`` face -> its logical ``(μ, μ)``, replicated."""
    return jax.jit(lambda a: a[0, :n, :n], out_shardings=NamedSharding(mesh, P()))


@functools.lru_cache(maxsize=None)
def _face_program(mesh, n, mu_pad):
    """jit: a replicated logical ``(μ, μ)`` -> the stored ``(1, μ_pad, μ_pad)`` face."""
    pad = int(mu_pad) - int(n)
    return jax.jit(lambda a: jnp.pad(a, ((0, pad), (0, pad)))[None],
                   out_shardings=NamedSharding(mesh, P(None, "x", "y")))


def q_algebra(src, iq, mesh, *, rcond, print_fn=print):
    """Read stored q ``iq`` and return ``(Z, B, K, V, Wc, v_sphere, diag)``; gates V and W.

    ``B``/``K`` are padded to ``μ_pad`` (zero pad rows/columns); ``V``/``Wc``
    are the logical replicated μ² matrices; ``v_sphere`` is LORRAX v on the
    stored sphere; ``diag`` holds the two gate figures and the spectrum
    range of K (−1 ≼ K ≼ 0 for a passive static response).
    """
    from isdf import cplus
    Vs, Ws, Z = read_q_rows(src, iq, mesh)
    n = src.n_mu
    rep = _replicate_logical(mesh, n)
    V, W = rep(Vs), rep(Ws)
    del Vs, Ws
    sphere = src.zeta.gvecs(q=[int(iq)])[0]
    ngk = int(src.zeta.ngk_valid(q=[int(iq)])[0])
    v_sphere = lorrax_v(src, src.q_stored[iq], sphere[:ngk])
    Vrec = coulomb_gram(Z, np.arange(ngk), v_sphere, mesh=mesh)[:n, :n]
    nV = float(jnp.linalg.norm(V))
    err_v = float(jnp.linalg.norm(Vrec - V)) / nV
    herm_w = float(jnp.linalg.norm(W - jnp.conj(W).T) / jnp.linalg.norm(W))
    if not np.isfinite(err_v) or err_v > V_RECONSTRUCTION_TOL:
        _refuse("epsmat_v_reconstruction", f"|Zᴴ v Z − V|/|V| = {err_v:.3e} at stored q {iq}",
                f"<= {V_RECONSTRUCTION_TOL:.0e}",
                "ε⁻¹ would be scaled by the ratio of two ζ normalizations",
                "use the bundle's own zeta_q.h5 and producer deck (Coulomb keys)")
    if herm_w > STATIC_W_HERMITIAN_TOL:
        _refuse("epsmat_static_w_hermitian", f"|W − Wᴴ|/|W| = {herm_w:.3e}",
                f"<= {STATIC_W_HERMITIAN_TOL:.0e}", "W(ω=0) is Hermitian, (E3) with a Hermitian static χ",
                "the bundle's W0_qmunu is not a static screened interaction")
    Vh = 0.5 * (V + jnp.conj(V).T)
    B, figures = coulomb_whitener(Vh, rcond=rcond)
    cplus.report(figures, rcond=float(rcond), n_log=n, where="epsmat V+")
    Wc = W - V
    K = whitened_correlation(0.5 * (Wc + jnp.conj(Wc).T), B)
    ev = np.linalg.eigvalsh(np.asarray(jax.device_get(K)))
    print_fn(f"  q {iq} {np.round(src.q_stored[iq], 6).tolist()}: |V_rec−V|/|V|={err_v:.2e} "
             f"|W−Wᴴ|/|W|={herm_w:.2e} K∈[{ev[0]:.4f}, {ev[-1]:.2e}]")
    pad = src.mu_pad - n
    Bp = jnp.pad(B, ((0, pad), (0, pad)))
    Kp = jnp.pad(K, ((0, pad), (0, pad)))
    diag = {"err_v": err_v, "herm_w": herm_w, "k_min": float(ev[0]), "k_max": float(ev[-1])}
    return Z, Bp, Kp, V, Wc, v_sphere, diag


def _check_projection(proj, V_e, B, K, Wc, n, iq, where):
    """GATE: the block projects back to V_e B K Bᴴ V_e (exact for any G list)."""
    ref = V_e @ B[:n, :n] @ K[:n, :n] @ jnp.conj(B[:n, :n]).T @ V_e
    err = float(jnp.linalg.norm(proj[:n, :n] - ref) / jnp.linalg.norm(Wc))
    if not np.isfinite(err) or err > BLOCK_PROJECTION_TOL:
        _refuse("epsmat_block_projection", f"relative error {err:.3e} at stored q {iq} ({where})",
                f"<= {BLOCK_PROJECTION_TOL:.0e}",
                "the written block is not the ISDF response (layout or sharding defect)",
                "report the deck; do not consume the file")
    rt = float(jnp.linalg.norm(proj[:n, :n] - Wc) / jnp.linalg.norm(Wc))
    return err, rt


# ---------------------------------------------------------------------------
# Export: LORRAX -> eps0mat.h5 / epsmat.h5
# ---------------------------------------------------------------------------

class QPlan(NamedTuple):
    """One output q: file row, stored row, umklapp, BerkeleyGW G list."""
    iq_file: int
    iq_store: int
    q_file: np.ndarray
    g0: np.ndarray
    gspace: object               # bgw_epsmat.QGspace


def plan_export(cfg, src, *, print_fn=print):
    """The q rows and G lists of each output file, refusing unmatched q.

    With ``bgw_reference`` the q list, q0, ecuts and G order are BerkeleyGW's
    own (the order is also recomputed and must agree); otherwise every
    stored q ≠ 0 goes to ``epsmat.h5`` and Γ to ``eps0mat.h5`` at
    ``q0_vector``.
    """
    import h5py
    from file_io import bgw_epsmat as be
    with h5py.File(src.wfn_file, "r") as w:
        comps = np.asarray(w["mf_header/gspace/components"][()], dtype=np.int64)
        bdot = np.asarray(w["mf_header/crystal/bdot"][()], dtype=np.float64)
    gamma = np.flatnonzero(np.all(np.abs(src.q_stored) < 1e-12, axis=1))
    plans = {}
    for name in cfg.files:
        ref = os.path.join(cfg.bgw_reference, f"{name}.h5") if cfg.bgw_reference else ""
        if ref:
            h = be.read_epsmat_header(ref)
            q_file, ecuts = h["qpts"], float(h["params/ecuts"])
        elif name == "eps0mat":
            q_file, ecuts = np.asarray([cfg.q0_vector]), cfg.epsilon_cutoff
        else:
            keep = np.setdiff1d(np.arange(src.q_stored.shape[0]), gamma)
            q_file, ecuts = np.mod(src.q_stored[keep], 1.0), cfg.epsilon_cutoff
            q_file[np.abs(q_file - 1.0) < 1e-12] = 0.0
        if src.config.head.bare_coulomb_cutoff is not None and ecuts > float(
                src.config.head.bare_coulomb_cutoff) + 1e-9:
            _refuse("epsmat_ecuts_beyond_coulomb", f"ecuts {ecuts} Ry",
                    f"<= bare_coulomb_cutoff {src.config.head.bare_coulomb_cutoff} Ry",
                    "LORRAX's V and W carry v = 0 beyond the bare cutoff",
                    "lower epsilon_cutoff or rerun GW with a larger bare_coulomb_cutoff")
        rows = []
        for i, q in enumerate(q_file):
            if name == "eps0mat":
                idx, g0 = (gamma[:1] if gamma.size else np.array([-1])), np.zeros((1, 3), int)
            else:
                idx, g0 = match_q_umklapp(src.q_stored, q[None])
            if int(idx[0]) < 0:
                _refuse("epsmat_q_unmatched", f"{name} q {np.round(q, 6).tolist()}",
                        "a stored LORRAX q equal up to a reciprocal lattice vector",
                        "a symmetry-rotated W is not reconstructed by this export",
                        "use a bundle stored on the full zone or a BerkeleyGW q list "
                        "of its stored wedge")
            qg = be.q_gspace(comps, bdot, q, ecuts, is_q0=(name == "eps0mat"))
            if ref:
                qref, _ = be.read_q_gspace(ref, i)
                if qref.nmtx != qg.nmtx or not np.array_equal(
                        qref.order[: qref.nmtx], qg.order[: qg.nmtx]):
                    _refuse("epsmat_bgw_gorder", f"{name} q {i}: nmtx {qg.nmtx} vs "
                            f"{qref.nmtx}, order equal={np.array_equal(qref.order[:qg.nmtx], qg.order[:qg.nmtx])}",
                            "BerkeleyGW's own sortrx order", "Sigma maps rows by gind",
                            "report: the sort emulation disagrees with the reference")
                qg = qref
            if name == "eps0mat" and np.any(comps[qg.order[0]] != 0):
                _refuse("epsmat_q0_head_first", f"first eps G {comps[qg.order[0]].tolist()}",
                        "G = 0 first at q → 0", "the head is row 0", "report")
            rows.append(QPlan(i, int(idx[0]), np.asarray(q), g0[0], qg))
        plans[name] = (np.asarray(q_file), ecuts, rows)
        print_fn(f"  {name}.h5: {len(rows)} q, ecuts {ecuts} Ry, nmtx "
                 f"{min(r.gspace.nmtx for r in rows)}..{max(r.gspace.nmtx for r in rows)}"
                 f"{'  (q list, q0 and G order from ' + ref + ')' if ref else ''}")
    return plans, comps


def _q0_head_value(cfg, src, q0):
    """ε⁻¹_00(q0) from the bundle's long-wavelength model (or 1 when asked)."""
    if cfg.q0_head == "unscreened":
        return 1.0 + 0.0j
    if src.head["S_cart"] is None:
        _refuse("epsmat_q0_head_source", "no S_cart_head in the bundle",
                "the Cartesian q² tensor of the producer's head",
                "the q → 0 head of ε⁻¹ is the long-wavelength model",
                "rerun GW with head_correction on, or set q0_head = unscreened")
    q0c = np.asarray(q0, dtype=np.float64) @ src.geometry.bvec
    v0 = float(bare_v_bgw_units(src, q0, np.zeros((1, 3), int))[0]) / src.geometry.cell_volume
    return q0_head(src.head["S_cart"], q0c, v0)


def run_epsmat_export(cfg, mesh, *, print_fn=print):
    """Write ``eps0mat.h5`` / ``epsmat.h5`` from a LORRAX bundle, one q at a time.

    Every q passes three gates before its block counts: V from ζ and v equals
    the bundle's V, W(0) is Hermitian, and the written block projects back to
    ``V_ε B K Bᴴ V_ε``.  Returns a dict of per-q figures for the report.
    """
    from common.collectives import barrier, process_rank
    from file_io import bgw_epsmat as be
    from file_io.slab_io import SlabIO

    src = open_export_source(cfg, mesh, print_fn=print_fn)
    plans, comps = plan_export(cfg, src, print_fn=print_fn)
    os.makedirs(cfg.output_dir, exist_ok=True)
    rank0 = process_rank() == 0
    _square_side(mesh)
    out = {}
    for name, (q_file, ecuts, rows) in plans.items():
        path = os.path.join(cfg.output_dir, f"{name}.h5")
        if rank0:
            be.create_epsmat_file(
                path, wfn_file=src.wfn_file, qpts=q_file, qgrid=src.kgrid,
                nmtx=[r.gspace.nmtx for r in rows], ecuts=ecuts,
                nband=int(src.config.nband),
                efermi_ry=be.midgap_ry(src.wfn_file, int(src.config.nband)),
                icutv=be.ICUTV_OF_SYS_DIM[int(src.config.sys_dim)])
            for r in rows:
                miller = comps[r.gspace.order[: r.gspace.nmtx]]
                be.write_q_gspace(path, r.iq_file, r.gspace,
                                  bare_v_bgw_units(src, r.q_file, miller))
        barrier(f"epsmat.{name}.skeleton")
        diags, figs = [], []
        with SlabIO(path, mode="a", mesh=mesh) as io:
            be.register_matrix(io, len(rows), max(r.gspace.nmtx for r in rows))
            for r in rows:
                Z, B, K, V, Wc, _, qd = q_algebra(src, r.iq_store, mesh, rcond=cfg.rcond,
                                              print_fn=print_fn)
                n = r.gspace.nmtx
                miller = comps[r.gspace.order[:n]] + r.g0[None, :]
                slots = zeta_slots(src.zeta.gvecs(q=[r.iq_store])[0],
                                   src.zeta.ngk_valid(q=[r.iq_store])[0], miller,
                                   fill=Z.shape[2])
                v_eps = lorrax_v(src, src.q_stored[r.iq_store], miller)
                head = _q0_head_value(cfg, src, r.q_file) if name == "eps0mat" else None
                blk, diag = epsinv_block(Z, slots, B, K, v_eps, mesh=mesh, n_valid=n,
                                         q0_head_value=head)
                offset, shape = be.matrix_block_window(r.iq_file, 0, blk.shape[0])
                blk6 = blk.reshape(shape)
                io.write_slab(be.MATRIX, blk6, offset=offset)
                proj = project_block(blk6, Z, slots, v_eps, mesh=mesh, n_valid=n,
                                     drop_head=name == "eps0mat")
                # v(G = 0) is already zero at Γ, so V_ε needs no head case.
                V_e = coulomb_gram(Z, slots, v_eps, mesh=mesh)[: src.n_mu, : src.n_mu]
                err, rt = _check_projection(proj, V_e, B, K, Wc, src.n_mu, r.iq_store, name)
                d = np.asarray(jax.device_get(diag))[:n]
                diags.append((r.iq_file, d))
                figs.append({"q": np.round(r.q_file, 6).tolist(), "iq_file": int(r.iq_file),
                             "iq_store": int(r.iq_store), "g0": np.asarray(r.g0).tolist(),
                             "nmtx": n, "projection_err": err, "roundtrip_vs_W": rt,
                             "head": (None if head is None else complex(head)), **qd})
                print_fn(f"    {name}[{r.iq_file}] nmtx {n}: block identity {err:.2e}, "
                         f"Zᴴ(ε⁻¹−1)vZ vs W−V {rt:.2e}"
                         + ("" if head is None else f", head ε⁻¹_00(q0)={complex(head):.6f}"))
        if rank0:
            for iq_file, d in diags:
                be.write_q_diagonal(path, iq_file, d)
        barrier(f"epsmat.{name}.done")
        out[name] = {"path": path, "ecuts": ecuts, "q": figs}
    return out


# ---------------------------------------------------------------------------
# Projection: BerkeleyGW eps files -> W_μν on the bundle's stored q
# ---------------------------------------------------------------------------

def run_epsmat_projection(cfg, mesh, *, print_fn=print):
    """Project BerkeleyGW's ε⁻¹ onto the bundle's ζ and write a variant bundle.

    For every stored q the matching BerkeleyGW q (an umklapp image; Γ maps
    to the eps0mat q0) is read block by block and ``W0_qmunu`` of a COPY of
    the bundle in ``output_dir/tmp`` is replaced by
    ``V + Z_εᴴ [(ε⁻¹_BGW − 1) diag(v)] Z_ε``; at Γ the G = 0 row and column are
    dropped (LORRAX carries the head as the rank-one ``whead`` term, kept).
    Returns per-q figures, including ``|W_BGW − W_LORRAX| / |W_LORRAX − V|``.
    """
    import shutil
    import h5py
    from common.collectives import barrier, process_rank
    from file_io import bgw_epsmat as be
    from file_io.slab_io import SlabIO

    src = open_export_source(cfg, mesh, print_fn=print_fn)
    files = {name: os.path.join(cfg.bgw_reference, f"{name}.h5") for name in ("eps0mat", "epsmat")}
    heads = {name: be.read_epsmat_header(p) for name, p in files.items() if os.path.isfile(p)}
    if "eps0mat" not in heads:
        _refuse("epsmat_project_inputs", f"no eps0mat.h5 under {cfg.bgw_reference}",
                "BerkeleyGW's eps0mat.h5 (and epsmat.h5 for q ≠ 0)",
                "every stored q needs its ε⁻¹", "set bgw_reference to the epsilon run dir")
    for name, h in heads.items():
        if int(h["params/matrix_type"]) != 0 or int(h["params/nmatrix"]) != 1:
            _refuse("epsmat_project_matrix_type", f"{name} matrix_type "
                    f"{int(h['params/matrix_type'])}, nmatrix {int(h['params/nmatrix'])}",
                    "ε⁻¹ (matrix_type 0), one spin matrix", "W = ε⁻¹ v",
                    "project an inverse dielectric file")
    rank0 = process_rank() == 0
    out_tmp = os.path.join(cfg.output_dir, "tmp")
    target = os.path.join(out_tmp, os.path.basename(src.bundle))
    if os.path.abspath(out_tmp) == os.path.dirname(src.bundle):
        _refuse("epsmat_project_output", out_tmp, "a directory other than the source's",
                "the source bundle is never mutated", "choose a new output_dir")
    if rank0:
        os.makedirs(out_tmp, exist_ok=True)
        shutil.copyfile(src.bundle, target)
        link = os.path.join(out_tmp, "zeta_q.h5")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(src.zeta._path), link)
    barrier("epsmat.project.copy")
    gamma = np.flatnonzero(np.all(np.abs(src.q_stored) < 1e-12, axis=1))
    figs = []
    comps = heads["eps0mat"]["components"]
    with SlabIO(target, mode="a", mesh=mesh) as io:
        # Register the copy's W0_qmunu (identical logical shape: reuse), the
        # parallel tier's precondition for writing into an existing dataset.
        io.create_dataset("W0_qmunu", shape=(src.q_stored.shape[0], src.n_mu, src.n_mu),
                          dtype=np.complex128)
        for iq in range(src.q_stored.shape[0]):
            is_g = iq in set(gamma.tolist())
            name = "eps0mat" if is_g else "epsmat"
            if name not in heads:
                _refuse("epsmat_project_inputs", f"no {name}.h5 for stored q {iq}",
                        "BerkeleyGW's file", "every stored q needs its ε⁻¹",
                        "add the file to bgw_reference")
            if is_g:
                iqf, g0 = 0, np.zeros(3, int)
            else:
                idx, g0s = match_q_umklapp(heads[name]["qpts"], src.q_stored[iq][None])
                if int(idx[0]) < 0:
                    _refuse("epsmat_q_unmatched", f"stored q {np.round(src.q_stored[iq], 6).tolist()}",
                            f"an umklapp image in {files[name]}",
                            "a symmetry-rotated ε⁻¹ is not reconstructed here",
                            "run BerkeleyGW epsilon on the full q list")
                iqf, g0 = int(idx[0]), -g0s[0]
            qg, _ = be.read_q_gspace(files[name], iqf)
            n = qg.nmtx
            Z, B, K, V, Wc, _, qd = q_algebra(src, iq, mesh, rcond=cfg.rcond, print_fn=print_fn)
            # q_BGW = q_store + g0, so ζ̃_{q_BGW}(G) = ζ̃_{q_store}(G + g0).
            miller = comps[qg.order[:n]] + g0[None, :]
            slots = zeta_slots(src.zeta.gvecs(q=[iq])[0], src.zeta.ngk_valid(q=[iq])[0],
                               miller, fill=Z.shape[2])
            v_eps = lorrax_v(src, src.q_stored[iq], miller)
            n_pad = _pad_to(n, int(mesh.shape["x"]) ** 2)
            offset, shape = be.matrix_block_window(iqf, 0, n_pad)
            with SlabIO(files[name], mode="r", mesh=mesh) as rd:
                blk = rd.read_slab(be.MATRIX, shape=shape, dtype=np.float64, offset=offset,
                                   mesh=mesh, partition_spec=P(None, None, None, "x", "y", None))
            Wc_bgw = project_block(blk, Z, slots, v_eps, mesh=mesh, n_valid=n,
                                   drop_head=is_g)[: src.n_mu, : src.n_mu]
            W_new = V + Wc_bgw
            rel = float(jnp.linalg.norm(Wc_bgw - Wc) / jnp.linalg.norm(Wc))
            # BerkeleyGW's W = ε⁻¹v is Hermitian only in the orientation read here.
            herm_bgw = float(jnp.linalg.norm(Wc_bgw - jnp.conj(Wc_bgw).T) / jnp.linalg.norm(Wc_bgw))
            Bl = B[: src.n_mu, : src.n_mu]
            Kb = whitened_correlation(0.5 * (Wc_bgw + jnp.conj(Wc_bgw).T), Bl)
            Kl = K[: src.n_mu, : src.n_mu]
            evb = np.linalg.eigvalsh(np.asarray(jax.device_get(Kb)))
            rel_k = float(np.linalg.norm(np.asarray(jax.device_get(Kb - Kl)), 2)
                          / np.linalg.norm(np.asarray(jax.device_get(Kl)), 2))
            io.write_slab("W0_qmunu", _face_program(mesh, src.n_mu, src.mu_pad)(W_new),
                          offset=(iq, 0, 0))
            head_bgw = None
            if is_g:
                with h5py.File(files[name], "r") as f:
                    head_bgw = complex(f[be.MATRIX][0, 0, 0, 0, 0, 0] + 1j * f[be.MATRIX][0, 0, 0, 0, 0, 1])
            figs.append({"q": np.round(src.q_stored[iq], 6).tolist(), "iq_store": int(iq),
                         "file": name, "iq_file": iqf, "g0": np.asarray(g0).tolist(), "nmtx": n,
                         "rel_Wc_bgw_vs_lorrax": rel, "herm_bgw": herm_bgw,
                         "rel_K_bgw_vs_lorrax": rel_k, "k_bgw_min": float(evb[0]),
                         "k_bgw_max": float(evb[-1]), "bgw_head": head_bgw, **qd})
            print_fn(f"    stored q {iq} <- {name}[{iqf}] nmtx {n}: "
                     f"|Wc_BGW − Wc_LX|/|Wc_LX| = {rel:.4e}, |Wc_BGW − h.c.|/|Wc_BGW| = {herm_bgw:.1e}, "
                     f"‖K_BGW − K‖₂/‖K‖₂ = {rel_k:.4e}, K_BGW∈[{evb[0]:.4f}, {evb[-1]:.2e}]"
                     + ("" if head_bgw is None else f", BGW ε⁻¹_00(q0) = {head_bgw:.6f}"))
        io.stamp_dataset_attrs("W0_qmunu", {
            "W0_source": "bgw_epsmat", "bgw_reference": os.path.abspath(cfg.bgw_reference),
            "bgw_ecuts_ry": float(heads["eps0mat"]["params/ecuts"]),
            "bgw_q0_head_epsinv": str(figs[gamma[0]]["bgw_head"]) if gamma.size else "",
            "q0_head_note": "whead kept from the LORRAX producer; body from BerkeleyGW"})
    barrier("epsmat.project.done")
    return {"path": target, "q": figs}
