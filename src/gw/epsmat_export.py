"""LORRAX screened interaction <-> BerkeleyGW ε⁻¹(G,G'); see docs/theory/bgw_epsmat_export.md.

Pair densities and χ are EXPANDED in ζ, V and W are PROJECTED on it
(``Z[G, μ] = ζ̃_qμ(G)``, ``V = Zᴴ diag(v) Z``, ``W = Zᴴ W(G,G') Z``), so

    ε⁻¹_q(G,G') − δ = v_q(G) [Z χ Zᴴ](G,G'),   χ = V⁺ (W − V) V⁺        (E4)

with ``V⁺ = B Bᴴ`` from the face eigensolve of V (``distrib_la``) and the cut
of ``isdf.cplus.factor_from_eigh``, ``K = Bᴴ (W − V) B`` (−1 ≼ K ≼ 0 at
ω = 0) and ``χ = B K Bᴴ``.  The reverse is ``W = V + Z_εᴴ [(ε⁻¹ − 1) diag(v)]
Z_ε`` on the file's own G list.  One q at a time, and every matrix on a
``P(None,'x','y')`` face: ζ̃_q arrives μ-sharded and becomes the ``(μ, G)``
face by one all-to-all over ``y``; V, W, B, K and χ are ``(μ, μ)`` faces;
the ``(G, G')`` block is a face, written tile by tile.  Products are
``distrib_la.panel_matmul`` (2-D SUMMA, live panels ≤ ``PANEL_BYTES`` per
rank), so per rank the μ-space state is ``16 μ_pad²/P`` B per matrix and the
block ``16 n²/P`` B; only vectors and scalars are replicated.
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
from common.log import get_logger

log = get_logger(__name__)


__all__ = [
    "match_q_umklapp", "zeta_slots", "coulomb_whitener", "q0_body_policy", "q0_dipole_terms",
    "whitened_correlation", "epsinv_block", "project_block", "coulomb_gram",
    "run_epsmat_export", "run_epsmat_projection", "projection_frequencies",
    "import_q_plan", "import_gvectors", "import_zeta_face", "W_SOURCE_BGW",
]

_XY = ("x", "y")
#: GATE thresholds.  V from ζ and v is an identity of the producer's own
#: arithmetic (measured 1e-14 on CrI3 978); the block projection is an
#: identity of this module's assembly (measured 1e-14 on the same bundle).
V_RECONSTRUCTION_TOL = 1.0e-9
BLOCK_PROJECTION_TOL = 1.0e-8
STATIC_W_HERMITIAN_TOL = 1.0e-8
#: The stored head tensor is the fold of the stored wings through W(Γ)
#: (``S_cart_head = S_direct + L W R / Ω``); the export re-forms that fold
#: with the FILE's W(Γ), so the gate certifies L's μ order and normalisation
#: where the export consumes them (relative to the fold term).
Q0_WING_FOLD_TOL = 1.0e-3


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


# ---------------------------------------------------------------------------
# μ-space algebra on P(None,'x','y') faces (distrib_la)
# ---------------------------------------------------------------------------
# Every N_μ²- and N_G²-class object below is a face: rank (x, y) holds the
# (x, y) tile and nothing else.  Products are distrib_la.panel_matmul (a 2-D
# SUMMA whose live panels are bounded by PANEL_BYTES), the eigensolve is the
# service's face plan, and masks and scalings are elementwise programs on
# the face.  Only vectors (eigenvalues, v, the block diagonal) and scalars
# are replicated.

#: Bytes per rank the two live operand panels of one product may hold.
PANEL_BYTES = 64 << 20


def _square_side(mesh):
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if px != py:
        _refuse("epsmat_square_mesh", f"mesh {px}x{py}", "a square mesh",
                "the face products are a SUMMA on a p x p grid",
                "run on 1, 4, 9, 16, ... devices")
    return px


def _pad_to(n, m):
    return -(-int(n) // int(m)) * int(m)


def _face(mesh):
    return NamedSharding(mesh, P(None, "x", "y"))


def eigh_plan(mesh, n):
    """The face eigensolver of one ``(n, n)`` Hermitian matrix at ``P('x','y')``.

    ``distributed`` (cuSOLVERMp on CUDA, ScaLAPACK on CPU) whenever each
    device has its own process; with one process every device's data is in
    that process already, and the native solve costs the same memory.
    distrib_la's ``auto`` resolves eigh to native everywhere, which would
    solve a gathered matrix on every rank.  Eigenvalues come back ascending
    and replicated, eigenvectors as columns on the input's face.
    """
    import distrib_la as dla
    backend = "distributed" if jax.process_count() > 1 else "native"
    return dla.plan("eigh", mesh, backend=backend, n=int(n))


def _pm(a, b, mesh, transa="N", transb="N"):
    """``op(a) @ op(b)`` for ``(1, ·, ·)`` faces (distrib_la.panel_matmul)."""
    import distrib_la as dla
    return dla.panel_matmul(a, b, mesh=mesh, panel_bytes=PANEL_BYTES,
                            transa=transa, transb=transb)


@functools.lru_cache(maxsize=None)
def _zeta_face_program(mesh, n_pad, mu_pad, ng_disk):
    """jit: ζ̃ ``(1, μ, ngkmax)`` μ-sharded + slots ``(n,)`` -> ``Z_f (1, μ, n)`` face.

    ``Z_f[μ, G] = ζ̃_μ(G)`` on the requested G list (slot ``ngkmax`` reads
    zero).  One all-to-all over ``y``: rank (x, y) sends its μ/P rows' column
    block y' to (x, y') and receives the μ/p rows of its x block.
    """
    def tile(zt, slots):
        z = jnp.pad(zt, ((0, 0), (0, 0), (0, 1)))
        z = jnp.take(z, slots, axis=2)                       # (1, μ/P, n)
        return jax.lax.all_to_all(z, "y", 2, 1, tiled=True)  # (1, μ/p, n/p)

    smap = shard_map(tile, mesh=mesh, in_specs=(P(None, _XY, None), P()),
                     out_specs=P(None, "x", "y"), check_vma=False)
    return jax.jit(smap, in_shardings=(NamedSharding(mesh, P(None, _XY, None)),
                                       NamedSharding(mesh, P())),
                   out_shardings=_face(mesh))


def zeta_face(zeta_q, slots, *, mesh, n_pad):
    """``Z_f (1, μ_pad, n_pad)`` at ``P(None,'x','y')``: ζ̃ on the G list ``slots``.

    ``zeta_q`` ``(1, μ_pad, ngkmax)`` μ-sharded (``ZetaLoader.read_zeta_G_slab``);
    ``slots`` ``(n,)`` the stored-sphere slot of each G (``ngkmax`` outside it);
    columns past ``n`` are zero.
    """
    p = _square_side(mesh)
    mu_pad, ngk = int(zeta_q.shape[1]), int(zeta_q.shape[2])
    if n_pad % p:
        _refuse("epsmat_block_extent", f"G extent {n_pad}", f"a multiple of {p}",
                "the G axis is a face axis", "pad the G list to the mesh")
    sl = np.full(int(n_pad), ngk, dtype=np.int32)
    sl[: len(slots)] = np.asarray(slots, dtype=np.int32)
    return _zeta_face_program(mesh, int(n_pad), mu_pad, ngk)(zeta_q, jnp.asarray(sl))


@functools.lru_cache(maxsize=None)
def _scale_columns(mesh):
    """jit: face ``(1, m, n)`` times replicated ``(n,)``, conjugated or not."""
    rep = NamedSharding(mesh, P())
    return (jax.jit(lambda a, v: a * v[None, None, :], in_shardings=(_face(mesh), rep),
                    out_shardings=_face(mesh)),
            jax.jit(lambda a, v: jnp.conj(a) * v[None, None, :], in_shardings=(_face(mesh), rep),
                    out_shardings=_face(mesh)))


def coulomb_gram(Zf, v, *, mesh):
    """``V_s = Zᴴ diag(v) Z`` (Ry), ``(1, μ_pad, μ_pad)`` face, from ``Z_f`` and v ``(n_pad,)``.

    With every stored-sphere slot this is the bundle's own V_q (the
    normalization gate); on the eps G list it is ``V_ε``.
    """
    _, conj_scale = _scale_columns(mesh)
    return _pm(conj_scale(Zf, jnp.asarray(v, jnp.float64)), Zf, mesh, transb="T")


def hermitian(a):
    """``(a + aᴴ)/2`` on ``a``'s face (distrib_la.hermitian_part)."""
    import distrib_la as dla
    return dla.hermitian_part(a)


def coulomb_whitener(V_face, *, rcond, plan):
    """``(B, figures)`` with ``B Bᴴ = V⁺`` on the retained Coulomb span, B on V's face.

    ``V_face`` ``(1, μ_pad, μ_pad)`` Hermitian-PSD V_q (Ry; zero pad rows,
    which the eigensolve returns as exact zeros below the cut).  The cut keeps
    ``λ > rcond λ_max``, closed over degenerate multiplets and certified
    (``isdf.cplus.factor_from_eigh``); dropped modes are zero columns, so
    shapes do not depend on the rank.
    """
    from isdf import cplus
    mesh = V_face.sharding.mesh
    lam, U = plan(hermitian(V_face)[0])
    B, figures = cplus.factor_from_eigh(lam[None], _on_face(mesh)(U)[None],
                                        rcond=float(rcond), rank_log=True, where="epsmat V+")
    return _on_face(mesh)(B), figures


@functools.lru_cache(maxsize=None)
def _on_face(mesh):
    """jit: ``(n, n)`` or ``(1, n, n)`` -> the same values on the ``(x, y)`` face.

    The eigensolve may hand its vectors back on another layout (the native
    plan in one process returns them whole); this lays them, and B, on V's
    face inside a program."""
    def run(a):
        spec = P("x", "y") if a.ndim == 2 else P(None, "x", "y")
        return jax.lax.with_sharding_constraint(a, NamedSharding(mesh, spec))
    return jax.jit(run)


def whitened_correlation(Wc_face, B, *, mesh):
    """K = Bᴴ (W − V) B, Hermitian part, on the face: the symmetrized
    response ``v^{1/2} χ v^{1/2}`` in the Coulomb span (−1 ≼ K ≼ 0 at ω = 0)."""
    return hermitian(_pm(B, _pm(hermitian(Wc_face), B, mesh), mesh, transa="C"))


def response(B, K, *, mesh):
    """``χ^μ = B K Bᴴ`` (= V⁺(W − V)V⁺ on the retained span), on the face."""
    return _pm(_pm(B, K, mesh), B, mesh, transb="C")


@functools.lru_cache(maxsize=None)
def _block_finish(mesh, n_pad, is_q0):
    """jit: ``T1 = conj(Z_f)ᵀ χᵀ Z_f`` face -> the file block and its diagonal.

    ``T[a, b] = ε⁻¹(G_b, G_a) = δ_ab + v_b T1[a, b]`` (BerkeleyGW orientation);
    rows and columns past ``n_valid`` are zero.  At q → 0 the wings are zero
    and the head is the long-wavelength model.  Returns the
    ``(1, 1, 1, n, n, 2)`` (Re, Im) block at ``P(None,None,None,'x','y',None)``
    and the diagonal ``(n,)`` replicated (every process reads it).
    """
    rep = NamedSharding(mesh, P())
    bsh = NamedSharding(mesh, P(None, None, None, "x", "y", None))

    def run(t1, v, n_valid, head, w0b, wa0):
        a = jnp.arange(n_pad)[:, None]
        b = jnp.arange(n_pad)[None, :]
        t = t1[0] * v[None, :] + ((a == b) & (a < n_valid)).astype(t1.dtype)
        if is_q0:
            # T[0, b] = ε⁻¹(G_b, 0) (E11) and T[a, 0] = ε⁻¹(0, G_a) (E12);
            # zero vectors give the q0_wings = zero file.
            t = jnp.where((a == 0) | (b == 0), 0.0, t)
            t = jnp.where((a == 0) & (b > 0) & (b < n_valid), w0b[None, :], t)
            t = jnp.where((b == 0) & (a > 0) & (a < n_valid), wa0[:, None], t)
            t = jnp.where((a == 0) & (b == 0), head, t)
        blk = jnp.stack([t.real, t.imag], axis=-1)[None, None, None]
        return blk, jnp.diagonal(t)

    return jax.jit(run, in_shardings=(_face(mesh), rep, rep, rep, rep, rep),
                   out_shardings=(bsh, rep))


def epsinv_block(Zf, chi, v_eps, *, mesh, n_valid, q0_head_value=None, q0_wings=None):
    """ε⁻¹_q(G,G') on one BerkeleyGW G list, in the file's block orientation (E6).

    Parameters
    ----------
    Zf : (1, μ_pad, n_pad) complex128 face — ζ̃ on the eps G list (:func:`zeta_face`).
    chi : (1, μ_pad, μ_pad) complex128 face — :func:`response`.
    v_eps : (n,) float — LORRAX v(q+G)/Ω in Ry at the eps G list.
    n_valid : int — nmtx; rows and columns past it are zero (mesh pad).
    q0_head_value : complex | None — at q → 0 the head ε⁻¹_00(q0).
    q0_wings : (w0b, wa0) | None — at q → 0 the two wing vectors ``(n_pad,)``,
        ``w0b[b] = ε⁻¹(G_b, 0)`` and ``wa0[a] = ε⁻¹(0, G_a)`` (:func:`q0_dipole_terms`);
        ``None`` writes zero wings.

    Returns
    -------
    block : (1, 1, 1, n_pad, n_pad, 2) float64 at ``P(None,None,None,'x','y',None)``,
        ``[a, b]`` = (Re, Im) ε⁻¹(G_b, G_a); diagonal : (n_pad,) complex, replicated.
    """
    n_pad = int(Zf.shape[2])
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[: len(v_eps)] = np.asarray(v_eps, dtype=np.float64)
    t1 = _pm(jnp.conj(Zf), _pm(chi, Zf, mesh, transa="T"), mesh, transa="T")
    head = 1.0 + 0.0j if q0_head_value is None else complex(q0_head_value)
    if q0_wings is None:
        w0b = wa0 = jnp.zeros(n_pad, jnp.complex128)
    else:
        w0b, wa0 = (jnp.asarray(w, jnp.complex128) for w in q0_wings)
    return _block_finish(mesh, n_pad, q0_head_value is not None)(
        t1, jnp.asarray(vv), jnp.int32(n_valid), jnp.complex128(head), w0b, wa0)


# ---------------------------------------------------------------------------
# q → 0: the dipole wings (docs/theory/bgw_epsmat_export.md §5, E9–E14)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _face_matvec(mesh):
    """jit: face ``(1, m, m)`` A, replicated x, y ``(m,)`` -> ``(A x, yᵀ A)`` replicated."""
    rep = NamedSharding(mesh, P())
    return jax.jit(lambda A, x, y: (jnp.einsum("ij,j->i", A[0], x),
                                    jnp.einsum("i,ij->j", y, A[0])),
                   in_shardings=(_face(mesh), rep, rep), out_shardings=(rep, rep))


@functools.lru_cache(maxsize=None)
def _add_outer(mesh):
    """jit: face ``χ + c u ũᵀ`` (an elementwise program; u, ũ replicated vectors)."""
    rep = NamedSharding(mesh, P())
    return jax.jit(lambda A, u, ut, c: A + c * u[None, :, None] * ut[None, None, :],
                   in_shardings=(_face(mesh), rep, rep, rep), out_shardings=_face(mesh))


@functools.lru_cache(maxsize=None)
def _zeta_contract(mesh):
    """jit: ζ̃ face ``Z_f (1, μ, n)`` -> ``(Z u)(G) = Σ_μ Z_f[μ,G] u_μ`` and
    ``(ũᵀZᴴ)(G) = Σ_μ ũ_μ conj(Z_f[μ,G])``, replicated ``(n,)``."""
    rep = NamedSharding(mesh, P())
    return jax.jit(lambda Z, u, ut: (jnp.einsum("mg,m->g", Z[0], u),
                                     jnp.einsum("mg,m->g", jnp.conj(Z[0]), ut)),
                   in_shardings=(_face(mesh), rep, rep), out_shardings=(rep, rep))


def q0_dipole_terms(Zf, chi, V, W, wl, wr, v_eps, *, mesh, v0, chi00):
    """The q → 0 block of ε⁻¹ from the Γ body and the dipole wings (E10–E14).

    Parameters
    ----------
    Zf : (1, μ_pad, n_pad) face — ζ̃ of Γ on the eps0mat G list.
    chi, V, W : (1, μ_pad, μ_pad) faces — Γ's χ^μ = B K Bᴴ, V and W = V + VχV (Ry).
    wl, wr : (μ_pad,) complex — the q-contracted wings of χ₀, −q0·L and −R q0 (E9).
    v_eps : (n,) — v(q0 + G)/Ω (Ry), zero in the G = 0 slot.
    v0 : float — v(q0)/Ω, the bare (truncated) Coulomb of the head slot (Ry).
    chi00 : complex — Ω q0ᵀ S_direct q0, the direct head of χ₀.

    Returns
    -------
    chi_q0 : face — χ^μ + c u ũᵀ (E13), the body's response with its head–wing term.
    head : complex — ε⁻¹_00 = 1/ε_M (E10).
    wings : (w0b, wa0) replicated (n_pad,) — T[0, b] = ε⁻¹(G_b, 0) (E11),
        T[a, 0] = ε⁻¹(0, G_a) (E12).
    figures : dict — ε_M, the fold ũᵀVω = wlᵀ W wr, |u|, |ũ|.
    """
    mv = _face_matvec(mesh)
    wl = jnp.asarray(wl, jnp.complex128)
    wr = jnp.asarray(wr, jnp.complex128)
    Vr, lV = mv(V, wr, wl)                    # V ω, ω̃ᵀ V
    cVr, lVc = mv(chi, Vr, lV)                # χ V ω, ω̃ᵀ V χ
    u, ut = wr + cVr, wl + lVc                # (1 + χV) ω, ω̃ᵀ(1 + Vχ)       (E11, E12)
    Ww, _ = mv(W, wr, wl)
    fold = complex(jnp.dot(wl, Ww))           # ω̃ᵀ W ω = ũᵀ V ω                (E10)
    eps_m = 1.0 - float(v0) * (complex(chi00) + fold)
    c = float(v0) / eps_m
    chi_q0 = _add_outer(mesh)(chi, u, ut, jnp.complex128(c))
    zu, zut = _zeta_contract(mesh)(Zf, u, ut)
    n_pad = int(Zf.shape[2])
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[: len(v_eps)] = np.asarray(v_eps, dtype=np.float64)
    w0b = jnp.asarray(vv) * zu / eps_m
    wa0 = c * zut
    figures = {"eps_M": eps_m, "fold": fold, "chi00": complex(chi00), "v0": float(v0),
               "u_norm": float(jnp.linalg.norm(u)), "ut_norm": float(jnp.linalg.norm(ut))}
    return chi_q0, 1.0 / eps_m, (w0b, wa0), figures


@functools.lru_cache(maxsize=None)
def _block_minus_one(mesh, n_pad, drop_head):
    """jit: file block ``(1,1,1,n,n,2)`` -> ``T − 1`` face ``(1, n, n)``, masked.

    Rows/columns past ``n_valid`` are zero; ``drop_head`` also zeroes the
    G = 0 row and column (the q → 0 head and wings LORRAX carries outside
    the body)."""
    rep = NamedSharding(mesh, P())
    bsh = NamedSharding(mesh, P(None, None, None, "x", "y", None))

    def run(blk, n_valid):
        a = jnp.arange(n_pad)[:, None]
        b = jnp.arange(n_pad)[None, :]
        live = (a < n_valid) & (b < n_valid)
        t = jnp.where(live, blk[0, 0, 0, ..., 0] + 1j * blk[0, 0, 0, ..., 1], 0.0)
        t = t - ((a == b) & live).astype(t.dtype)
        if drop_head:
            t = jnp.where((a == 0) | (b == 0), 0.0, t)
        return t[None]

    return jax.jit(run, in_shardings=(bsh, rep), out_shardings=_face(mesh))


def project_block(block, Zf, v_eps, *, mesh, n_valid, drop_head):
    """W_c(q) = Z_εᴴ [(ε⁻¹ − 1) diag(v)] Z_ε (E7) from one block in file orientation.

    ``block`` ``(1, 1, 1, n_pad, n_pad, 2)`` at ``P(None,None,None,'x','y',None)``,
    ``Zf`` the ζ̃ face on the same G list.  Returns ``(1, μ_pad, μ_pad)``
    complex128 on the face, Ry.  With ``T − 1 = (ε⁻¹ − 1)ᵀ``:
    ``W_c = conj(Z_f) [(Z_f diag v)(T − 1)]ᵀ``.
    """
    p = _square_side(mesh)
    n_pad = int(block.shape[3])
    if n_pad % p or n_pad != int(Zf.shape[2]):
        _refuse("epsmat_block_extent", f"block extent {n_pad}, ζ̃ face {int(Zf.shape[2])}",
                f"one extent, a multiple of {p}", "the G axis is a face axis",
                "read the block through bgw_epsmat.matrix_block_window")
    vv = np.zeros(n_pad, dtype=np.float64)
    vv[: len(v_eps)] = np.asarray(v_eps, dtype=np.float64)
    scale, _ = _scale_columns(mesh)
    tm1 = _block_minus_one(mesh, n_pad, bool(drop_head))(block, jnp.int32(n_valid))
    S = _pm(scale(Zf, jnp.asarray(vv)), tm1, mesh)
    return _pm(jnp.conj(Zf), S, mesh, transb="T")


def face_spectrum(A_face, plan):
    """Eigenvalues ``(n,)`` (ascending, host) of the Hermitian part of a face."""
    lam, _ = plan(hermitian(A_face)[0])
    return np.asarray(jax.device_get(lam))


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
    eigh: object                 # distrib_la face eigensolver, n = mu_pad (eigh_plan)
    sym: object                  # SymMaps: the shared reciprocal/antiunitary action
    q_full: np.ndarray           # (nk_full, 3), the complete unshifted q grid
    q_tables: object             # validated persisted QirrTables, or None for full storage
    w0_ready: bool = True        # the bundle carries LORRAX's own ready W0_qmunu
    storage: str = "full"        # V_qmunu's q storage: "ibz" (the wedge) or "full"


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


def export_q_metadata(bundle, wfn, kgrid, storage, *, with_w0=True):
    """Full-q geometry and the persisted, authenticated parent/action tables.

    Metadata only: neither V nor W is read or unfolded here.  ``with_w0``:
    W0's tables must be V's (a bundle without W0 checks V alone).
    """
    from symmetry_maps import bgw_integer_q_to_fractional, read_tables, read_tensor
    if not np.array_equal(np.asarray(wfn.kgrid), np.asarray(kgrid)):
        _refuse("epsmat_q_grid", f"WFN grid {wfn.kgrid}, bundle grid {kgrid}",
                "the same full q grid", "q row labels depend on the grid",
                "use the bundle's own WFN")
    sym = wfn.symmetry()
    q_full = bgw_integer_q_to_fractional(sym.kvecs_asints, np.asarray(kgrid))
    tables = None
    if storage == "ibz":
        _, vh = read_tensor(bundle, "V_qmunu", metadata_only=True)
        _, wh = (read_tensor(bundle, "W0_qmunu", metadata_only=True) if with_w0 else (None, vh))
        tables = read_tables(bundle, "V_qmunu")
        if (vh.table_hash != wh.table_hash or tables.n_q_full != len(q_full)
                or tables.n_sym_spatial != len(sym.sym_matrices)):
            _refuse("epsmat_q_tables", "inconsistent V/W or WFN q-action tables",
                    "the same persisted parent/action map for V, W and the WFN",
                    "the output q images must be the producer's own",
                    "use the bundle's own WFN and restart pair")
        S, _, _ = sym.operation_rows(np.asarray(tables.sym_idx_q, dtype=np.int32))
        moved = np.einsum("qij,qj->qi", S,
                          np.asarray(tables.q_irr_frac)[np.asarray(tables.irr_idx_q)])
        delta = moved - q_full
        if not np.allclose(delta, np.rint(delta), rtol=0.0, atol=1e-10):
            _refuse("epsmat_q_action", "persisted rows do not map parents to the WFN q grid",
                    "exact shared reciprocal actions modulo lattice vectors",
                    "the stored and current row order must agree",
                    "use the bundle's own WFN symmetry header")
    return sym, np.asarray(q_full), tables


def open_export_source(cfg, mesh, *, print_fn=None, require_w0=True):
    """Open the producer's bundle, ζ and Coulomb geometry; refuse unsupported decks.

    ``require_w0`` (the export): the bundle must carry a ready ``W0_qmunu``.
    The projection reads V and ζ only and compares with LORRAX's W(0) when
    the bundle has one (``ExportSource.w0_ready``).
    """
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
        w0_ready = "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False))
        if require_w0 and not w0_ready:
            _refuse("epsmat_requires_w0", f"{bundle} without a ready W0_qmunu",
                    "a GW bundle with persisted static W", "ε⁻¹ is built from W(0)",
                    "rerun the producer with write_restart_tensors = true")
        storage = str(f["V_qmunu"].attrs.get("q_storage", "full"))
        q_stored = (np.asarray(f["V_qmunu__qirr/q_irr_frac"][()]) if storage == "ibz"
                    else bgw_integer_q_to_fractional(
                        wfn.symmetry().kvecs_asints, np.asarray(kgrid)))
        n_mu = int(f["V_qmunu"].shape[-1])
        head = {"S_cart": (np.asarray(f["S_cart_head"][()]) if "S_cart_head" in f else None),
                "S_direct": (np.asarray(f["S_cart_head_direct"][()])
                             if "S_cart_head_direct" in f else None),
                "L": np.asarray(f["head_wing_L"][()]) if "head_wing_L" in f else None,
                "R": np.asarray(f["head_wing_R"][()]) if "head_wing_R" in f else None,
                "G0": (np.asarray(f["G0_mu_nu"][()]) if "G0_mu_nu" in f else None),
                "whead": (np.asarray(f["whead"][()]) if "whead" in f else None),
                "head_correction": (str(f["whead"].attrs.get("head_correction", ""))
                                    if "whead" in f else str(gw.head.correction.value))}
        if (w0_ready and storage == "ibz"
                and str(f["W0_qmunu"].attrs.get("q_storage", "full")) != "ibz"):
            _refuse("epsmat_q_storage", "V on the wedge and W0 on the full zone",
                    "one stored q set", "V and W rows are read pairwise",
                    "regenerate the bundle")
    zl = ZetaLoader(zeta_path, mesh=mesh)
    if int(zl.n_q_on_disk) != q_stored.shape[0] or int(zl.n_rmu) != n_mu:
        _refuse("epsmat_zeta_bundle_pair", f"ζ (q {zl.n_q_on_disk}, μ {zl.n_rmu})",
                f"bundle (q {q_stored.shape[0]}, μ {n_mu})",
                "V and W are projected on this ζ", "point zeta_file at the bundle's own ζ")
    mu_pad = int(padded_mu_axis(n_mu, mesh).carrier)
    sym, q_full, q_tables = export_q_metadata(bundle, wfn, kgrid, storage, with_w0=w0_ready)
    eigh = eigh_plan(mesh, mu_pad)
    n_dev = int(mesh.devices.size)
    log.info(f"  eigh        {eigh.describe()}")
    log.info(f"  μ faces     {16 * mu_pad**2 / n_dev / 2**30:.3f} GiB per device per matrix "
             f"(16 μ_pad²/P, P = {n_dev})")
    if jax.process_count() == 1 and n_dev > 1:
        log.info(f"  NOTE        eigh native in one process: each solve gathers "
                 f"{16 * mu_pad**2 / 2**30:.3f} GiB per device; one process per device "
                 f"takes the distributed solver")
    log.info(f"  bundle      {bundle}\n  zeta        {zeta_path}\n"
             f"  stored q    {q_stored.shape[0]} ({storage}) on k-grid {kgrid}, "
             f"μ {n_mu} (carrier {mu_pad})\n  head        {head['head_correction']}"
             + (f"; q0 wings L/R (3 x {n_mu}) and S_direct stored" if head["L"] is not None
                else "; no q0 wings stored"))
    return ExportSource(bundle=bundle, zeta=zl, q_stored=q_stored, kgrid=kgrid,
                        n_mu=n_mu, mu_pad=mu_pad, geometry=geometry, config=gw,
                        head=head, wfn_file=gw.paths.wfn_file, eigh=eigh,
                        sym=sym, q_full=q_full, q_tables=q_tables, w0_ready=w0_ready,
                        storage=storage)


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


def _file_q_v(src, q_file, gvecs, is_q0):
    """LORRAX v(q+G)/Ω at the file's own q and G list, with v(G = 0) = 0 at q0.

    ``q_file + G`` is the same vector as the stored ``q + (G + G0)``, so for
    q ≠ 0 these are V_q's own values.  At q0 the rows are v(q0 + G), as
    BerkeleyGW forms ``W = ε⁻¹ v(q0 + G')``, so W stays Hermitian; the G = 0
    entry is zero because the head is the long-wavelength model (E8) and the
    wings are zero, and ``χ`` and ζ̃ are the stored Γ ones (an O(q0) step).
    """
    v = np.array(lorrax_v(src, q_file, gvecs), dtype=np.float64)
    if is_q0:
        v[0] = 0.0
    return v


def read_q_rows(src, iq, mesh, *, with_w=True):
    """``(V, W, ζ̃)`` of stored q ``iq``: V/W ``(1, μ_pad, μ_pad)`` at
    ``P(None,'x','y')`` (zero-padded), ζ̃ ``(1, μ_pad, ngkmax)`` μ-sharded.
    ``with_w = False`` reads no W (``None``)."""
    from file_io.slab_io import SlabIO
    shape = (1, src.mu_pad, src.mu_pad)
    names = ("V_qmunu", "W0_qmunu") if with_w else ("V_qmunu",)
    with SlabIO(src.bundle, mode="r", mesh=mesh) as io:
        rows = [io.read_slab(name, shape=shape, dtype=np.complex128,
                             offset=(int(iq), 0, 0), mesh=mesh,
                             partition_spec=P(None, "x", "y"))
                for name in names]
    V, W = rows[0], (rows[1] if with_w else None)
    Z = src.zeta.read_zeta_G_slab(q_offset=int(iq), q_count=1, mu_offset=0,
                                  mu_count=src.mu_pad, mesh=mesh)
    return V, W, Z


@functools.lru_cache(maxsize=None)
def _fourier_partner_program(mesh):
    """Exact scalar Fourier action on μ-owned ζ; G gathers remain rank-local."""
    owner = NamedSharding(mesh, P(None, _XY, None))
    rep = NamedSharding(mesh, P())

    def apply(zeta, slots, phase):
        padded = jnp.pad(zeta, ((0, 0), (0, 0), (0, 1)))
        return jnp.conj(jnp.take(padded, slots, axis=2)) * phase[None, None, :]
    return jax.jit(apply, in_shardings=(owner, rep, rep), out_shardings=owner)


def fixed_q_zeta_basis(src, iq, Z, *, mesh):
    """Use the producer's measured Θ-even Fourier basis at fixed-q parents.

    The producer group-averages V on these rows. Realize the same symmetry
    on ζ itself before every normalization/export contraction. The unchanged
    V reconstruction gate still rejects a non-negligible odd-basis metric.
    """
    if src.q_tables is None:
        return Z, None
    from .qgrid_symmetry import qgrid_trs_policy_for
    tables = src.q_tables
    policy = qgrid_trs_policy_for(sym=src.sym, irr_idx_q=tables.irr_idx_q,
        sym_idx_q=tables.sym_idx_q, kgrid=src.kgrid,
        n_sym_spatial=tables.n_sym_spatial, context="epsilon Fourier basis")
    full, _ = match_q_umklapp(src.q_full, src.q_stored[iq][None])
    if int(full[0]) < 0:
        raise ValueError("Stored epsilon parent must name an exact full-grid q")

    def partner():
        rows = np.arange(2*tables.n_sym_spatial, dtype=np.int32)
        action, translation, anti = src.sym.operation_rows(rows)
        pure = np.flatnonzero(np.asarray(anti) &
            np.all(np.asarray(action) == -np.eye(3, dtype=int), axis=(1, 2)) &
            np.all(np.abs(translation) < 1e-12, axis=1))
        if pure.size != 1:
            raise ValueError("Measured TRS requires one exact identity-time-reversal row")
        sphere = src.zeta.gvecs(q=[iq])[0]
        ngk = int(src.zeta.ngk_valid(q=[iq])[0])
        row = QPlan(0, iq, src.q_stored[iq], np.zeros(3, int), None, int(pure[0]))
        miller, phase, conjugate = parent_gvectors(src, row, sphere[:ngk])
        slots = zeta_slots(sphere, ngk, miller, fill=Z.shape[2])
        if not conjugate or np.any(slots >= ngk):
            _refuse("epsmat_theta_sphere", f"stored q {iq} has missing Θ G partners",
                    "an exactly Θ-closed stored sphere", "a Fourier leg needs its exact image",
                    "use the producer's own ζ sphere")
        padded = np.full(Z.shape[2], Z.shape[2], dtype=np.int32)
        padded[:ngk] = slots
        phases = np.zeros(Z.shape[2], dtype=np.complex128)
        phases[:ngk] = 1.0 if phase is None else phase
        return _fourier_partner_program(mesh)(Z, jnp.asarray(padded), jnp.asarray(phases))

    return policy.project_fixed_q(Z, full, antiunitary_partner=partner)


def q_algebra(src, iq, mesh, *, rcond, print_fn=None):
    """Stored q ``iq`` on faces: ``(Z, B, K, chi, V, Wc, diag)``; gates V and W.

    ``Z`` is ζ̃ ``(1, μ_pad, ngkmax)`` μ-sharded; ``B``, ``K``, ``chi = B K Bᴴ``,
    ``V`` and ``Wc = W − V`` are ``(1, μ_pad, μ_pad)`` at ``P(None,'x','y')``
    (zero pad rows); ``diag`` holds the two gate figures and the range of
    K (−1 ≼ K ≼ 0 for a passive static response).
    """
    from isdf import cplus
    Z, V, W, basis = q_basis(src, iq, mesh)
    err_v, theta_removed = basis["err_v"], basis["zeta_anti_theta_removed"]
    herm_w = 2.0 * float(jnp.linalg.norm(W - hermitian(W))) / float(jnp.linalg.norm(W))
    if herm_w > STATIC_W_HERMITIAN_TOL:
        _refuse("epsmat_static_w_hermitian", f"|W − Wᴴ|/|W| = {herm_w:.3e}",
                f"<= {STATIC_W_HERMITIAN_TOL:.0e}", "W(ω=0) is Hermitian, (E3) with a Hermitian static χ",
                "the bundle's W0_qmunu is not a static screened interaction")
    B, figures = coulomb_whitener(V, rcond=rcond, plan=src.eigh)
    cplus.report(figures, rcond=float(rcond), n_log=src.n_mu, where="epsmat V+")
    Wc = W - V
    K = whitened_correlation(Wc, B, mesh=mesh)
    chi = response(B, K, mesh=mesh)
    ev = face_spectrum(K, src.eigh)
    log.info(f"  q {iq} {np.round(src.q_stored[iq], 6).tolist()}: |V_rec−V|/|V|={err_v:.2e} "
             f"|W−Wᴴ|/|W|={herm_w:.2e} K∈[{ev[0]:.4f}, {ev[-1]:.2e}]")
    diag = {"err_v": err_v, "herm_w": herm_w, "k_min": float(ev[0]), "k_max": float(ev[-1]),
            "zeta_anti_theta_removed": theta_removed}
    return Z, B, K, chi, V, Wc, diag


def q_basis(src, iq, mesh, *, with_w=True):
    """Stored q ``iq``: ``(Z, V, W, figures)`` with V certified against ζ and v.

    ``Z`` ζ̃ ``(1, μ_pad, ngkmax)`` μ-sharded in the producer's fixed-q basis
    (:func:`fixed_q_zeta_basis`); V and W ``(1, μ_pad, μ_pad)`` faces (W
    ``None`` with ``with_w = False``).  GATE ``epsmat_v_reconstruction``:
    ``Zᴴ diag(v) Z`` on the stored sphere is the bundle's V, so ε⁻¹ and W
    are converted with the producer's own ζ normalization.
    """
    V, W, Z = read_q_rows(src, iq, mesh, with_w=with_w)
    Z, theta_removed = fixed_q_zeta_basis(src, iq, Z, mesh=mesh)
    p = _square_side(mesh)
    sphere = src.zeta.gvecs(q=[int(iq)])[0]
    ngk = int(src.zeta.ngk_valid(q=[int(iq)])[0])
    v_sphere = lorrax_v(src, src.q_stored[iq], sphere[:ngk])
    n_s = _pad_to(max(ngk, 1), p)
    vs = np.zeros(n_s); vs[:ngk] = v_sphere
    Zs = zeta_face(Z, np.arange(ngk), mesh=mesh, n_pad=n_s)
    Vrec = coulomb_gram(Zs, vs, mesh=mesh)
    del Zs
    err_v = float(jnp.linalg.norm(Vrec - V)) / float(jnp.linalg.norm(V))
    if not np.isfinite(err_v) or err_v > V_RECONSTRUCTION_TOL:
        _refuse("epsmat_v_reconstruction", f"|Zᴴ v Z − V|/|V| = {err_v:.3e} at stored q {iq}",
                f"<= {V_RECONSTRUCTION_TOL:.0e}",
                "ε⁻¹ would be scaled by the ratio of two ζ normalizations",
                "use the bundle's own zeta_q.h5 and producer deck (Coulomb keys)")
    return Z, V, W, {"err_v": err_v, "zeta_anti_theta_removed": (
        None if theta_removed is None else np.asarray(theta_removed).tolist())}


def _check_projection(proj, V_e, chi, Wc, iq, where, mesh):
    """GATE: the block projects back to ``V_ε χ V_ε`` (exact for any G list); faces."""
    ref = _pm(_pm(V_e, chi, mesh), V_e, mesh)
    nwc = float(jnp.linalg.norm(Wc))
    err = float(jnp.linalg.norm(proj - ref)) / nwc
    if not np.isfinite(err) or err > BLOCK_PROJECTION_TOL:
        _refuse("epsmat_block_projection", f"relative error {err:.3e} at stored q {iq} ({where})",
                f"<= {BLOCK_PROJECTION_TOL:.0e}",
                "the written block is not the ISDF response (layout or sharding defect)",
                "report the deck; do not consume the file")
    rt = float(jnp.linalg.norm(proj - Wc)) / nwc
    return err, rt


class QPlan(NamedTuple):
    """One output q: file row, parent row, umklapp, G list and typed action."""
    iq_file: int
    iq_store: int
    q_file: np.ndarray
    g0: np.ndarray
    gspace: object               # bgw_epsmat.QGspace
    sym_row: int = 0             # identity for direct stored-q/umklapp images


def export_q_parent(src, q):
    """Stored parent, umklapp and typed action of one requested full-grid q."""
    idx, g0 = match_q_umklapp(src.q_stored, np.asarray(q)[None])
    if int(idx[0]) >= 0:
        return int(idx[0]), g0[0], 0
    full, shift = match_q_umklapp(src.q_full, np.asarray(q)[None])
    if int(full[0]) < 0 or src.q_tables is None:
        _refuse("epsmat_q_unmatched", f"q {np.round(q, 6).tolist()}",
                "a q on the producer's full grid",
                "a q outside that grid has no stored response",
                "use a BerkeleyGW q list on the producer's grid")
    iq = int(full[0])
    return (int(src.q_tables.irr_idx_q[iq]), shift[0],
            int(src.q_tables.sym_idx_q[iq]))


def parent_gvectors(src, row, miller):
    """Parent G rows, scalar Fourier phase and TR bit, by the SymMaps action.

    Centroid permutations and wrap phases cancel between Z, chi and Z^H;
    the plane-wave response therefore needs only the reciprocal action.
    """
    if row.sym_row == 0:
        return np.asarray(miller) + row.g0[None, :], None, False
    from symmetry_maps import unfold_reciprocal_carriers
    S, _, anti = src.sym.operation_rows(row.sym_row)
    inverse = np.rint(np.linalg.inv(S)).astype(np.int64)
    if not np.array_equal(inverse @ S, np.eye(3, dtype=np.int64)):
        raise ValueError("epsmat parent G action must be unimodular")
    q_parent = src.q_stored[row.iq_store]
    gp_float = unfold_reciprocal_carriers(inverse, row.q_file[None] + miller, q_parent)
    gp = np.rint(gp_float).astype(np.int64)
    if not np.allclose(gp_float, gp, rtol=0.0, atol=1e-6):
        _refuse("epsmat_parent_g", f"noninteger parent G at q {row.q_file.tolist()}",
                "an exact reciprocal relabel under the persisted action",
                "the q action and the G sphere must describe the same carriers",
                "use the bundle's own WFN and q grid")
    phase = src.sym.reciprocal_phase(row.sym_row, q_parent[None] + gp)
    return gp, phase, bool(anti)


def resolve_ecut(epsmat_ecut, full_ecuts, *, source):
    """The written G cutoff (Ry): ``epsmat_ecut``, or the full sphere when it is 0.

    ``full_ecuts`` is the largest cutoff the export can write, ``source``
    names it for the receipt (``bgw_reference``'s ecuts, or the producer's
    bare-Coulomb sphere). A cutoff below it selects the leading G of the same
    BerkeleyGW order, so the file is the sub-block of the converged ε⁻¹
    (docs/theory/bgw_epsmat_export.md#ecut); one above it refuses.
    Returns ``(ecut, receipt)``.
    """
    full = float(full_ecuts)
    e = float(epsmat_ecut)
    if e <= 0.0:
        return full, f"{full} Ry (full sphere: {source})"
    if e > full + 1e-9:
        _refuse("epsmat_ecut_beyond_sphere", f"epsmat_ecut {e} Ry",
                f"<= {full} Ry ({source})",
                "the export writes a sub-block of the stored ε⁻¹; G outside the full sphere "
                "carry no response", "lower epsmat_ecut, or leave it 0 for the full sphere")
    return e, f"{e} Ry (sub-block of the full sphere {full} Ry: {source})"


def plan_export(cfg, src, *, print_fn=None):
    """The q rows and G lists of each output file, refusing unmatched q.

    With ``bgw_reference`` the q list, q0, ecuts and G order are BerkeleyGW's
    own (the order is also recomputed and must agree); otherwise every
    full-grid q ≠ 0 goes to ``epsmat.h5`` and Γ to ``eps0mat.h5`` at
    ``q0_vector``. ``epsmat_ecut`` (0: the full sphere) keeps the leading
    ``gcutoff`` G of that order at every q (:func:`resolve_ecut`).
    """
    import h5py
    from file_io import bgw_epsmat as be
    with h5py.File(src.wfn_file, "r") as w:
        comps = np.asarray(w["mf_header/gspace/components"][()], dtype=np.int64)
        bdot = np.asarray(w["mf_header/crystal/bdot"][()], dtype=np.float64)
        ecutwfc = float(w["mf_header/kpoints/ecutwfc"][()])
    bare_set = src.config.head.bare_coulomb_cutoff
    bare = ecutwfc if bare_set is None else float(bare_set)    # gw_init's resolution of the V sphere
    gamma = np.flatnonzero(np.all(np.abs(src.q_stored) < 1e-12, axis=1))
    plans = {}
    for name in cfg.files:
        ref = os.path.join(cfg.bgw_reference, f"{name}.h5") if cfg.bgw_reference else ""
        if ref:
            h = be.read_epsmat_header(ref)
            q_file, full = h["qpts"], float(h["params/ecuts"])
            src_name = f"ecuts of {ref}"
        elif name == "eps0mat":
            q_file, full = np.asarray([cfg.q0_vector]), bare
            src_name = "the producer's bare_coulomb_cutoff"
        else:
            keep = np.flatnonzero(np.any(np.abs(src.q_full) > 1e-12, axis=1))
            q_file, full = np.mod(src.q_full[keep], 1.0), bare
            q_file[np.abs(q_file - 1.0) < 1e-12] = 0.0
            src_name = "the producer's bare_coulomb_cutoff"
        if bare_set is not None and full > bare + 1e-9:
            _refuse("epsmat_ecuts_beyond_coulomb", f"ecuts {full} Ry",
                    f"<= bare_coulomb_cutoff {bare} Ry",
                    "LORRAX's V and W carry v = 0 beyond the bare cutoff",
                    "lower epsmat_ecut or rerun GW with a larger bare_coulomb_cutoff")
        ecuts, ecut_receipt = resolve_ecut(cfg.epsmat_ecut, full, source=src_name)
        rows = []
        for i, q in enumerate(q_file):
            if name == "eps0mat":
                idx, g0 = (gamma[:1] if gamma.size else np.array([-1])), np.zeros((1, 3), int)
                sym_row = 0
            else:
                parent, shift, sym_row = export_q_parent(src, q)
                idx, g0 = np.asarray([parent]), shift[None]
            if int(idx[0]) < 0:
                _refuse("epsmat_q_unmatched", f"{name} q {np.round(q, 6).tolist()}",
                        "the stored Gamma response", "eps0mat uses the long-wavelength model",
                        "use a bundle containing Gamma")
            qg = be.q_gspace(comps, bdot, q, full, is_q0=(name == "eps0mat"))
            if ref:
                qref, _ = be.read_q_gspace(ref, i)
                if qref.nmtx != qg.nmtx or not np.array_equal(
                        qref.order[: qref.nmtx], qg.order[: qg.nmtx]):
                    _refuse("epsmat_bgw_gorder", f"{name} q {i}: nmtx {qg.nmtx} vs "
                            f"{qref.nmtx}, order equal={np.array_equal(qref.order[:qg.nmtx], qg.order[:qg.nmtx])}",
                            "BerkeleyGW's own sortrx order", "Sigma maps rows by gind",
                            "report: the sort emulation disagrees with the reference")
                qg = qref
            if ecuts < full:                       # the leading G of the same order: a sub-block
                qg = qg._replace(nmtx=be.gcutoff(qg.ekin[qg.order], ecuts))
            if name == "eps0mat" and np.any(comps[qg.order[0]] != 0):
                _refuse("epsmat_q0_head_first", f"first eps G {comps[qg.order[0]].tolist()}",
                        "G = 0 first at q → 0", "the head is row 0", "report")
            rows.append(QPlan(i, int(idx[0]), np.asarray(q), g0[0], qg, sym_row))
        plans[name] = (np.asarray(q_file), ecuts, rows)
        log.info(f"  {name}.h5: {len(rows)} q, ecuts {ecut_receipt}, nmtx "
                 f"{min(r.gspace.nmtx for r in rows)}..{max(r.gspace.nmtx for r in rows)}"
                 f"{'  (q list, q0 and G order from ' + ref + ')' if ref else ''}")
    return plans, comps


def _q0_head_value(cfg, src, q0):
    """ε⁻¹_00(q0) = W_h(q0) / v(q0), the direct head of the bundle's S tensor (E8).

    ``W_h = v / (1 − v q0ᵀSq0)`` is the owner's pointwise head
    (``gw.head_densify``: slab or bulk), with ``v`` the truncated
    8π/|q0|² in Ry bohr³ — no 1/Ω: ``q0ᵀSq0`` is χ₀₀ per volume
    (``docs/theory/s-tensor-convention.md``).  ``q0`` crystal, ``(3,)``.
    """
    if cfg.q0_head == "unscreened":
        return 1.0 + 0.0j
    if src.head["S_cart"] is None:
        _refuse("epsmat_q0_head_source", "no S_cart_head in the bundle",
                "the Cartesian q² tensor of the producer's head",
                "the q → 0 head of ε⁻¹ is the long-wavelength model",
                "rerun GW with head_correction on, or set q0_head = unscreened")
    from .head_densify import head_scalar_pointwise, slab_head_scalar_pointwise
    q0c = (np.asarray(q0, dtype=np.float64) @ src.geometry.bvec)[None, :]
    sys_dim = int(src.config.sys_dim)
    if sys_dim == 2:
        w_h = slab_head_scalar_pointwise(src.geometry, q0c, src.head["S_cart"])[0]
    elif sys_dim == 3:
        w_h = head_scalar_pointwise(q0c, src.head["S_cart"])[0]
    else:
        _refuse("epsmat_q0_head_source", f"sys_dim {sys_dim}", "2 (slab) or 3 (bulk)",
                "the pointwise head exists for the slab and bulk kernels",
                "set q0_head = unscreened")
    v_h = float(bare_v_bgw_units(src, q0, np.zeros((1, 3), int))[0])
    return complex(float(w_h) / v_h)


def q0_wing_inputs(cfg, src, q0, W_face, *, mesh):
    """``(wl, wr, v0, chi00, figures)`` of (E9) at ``q0`` from the bundle's wings, or refuse.

    ``wl = −q0·L``, ``wr = −R q0`` padded to μ_pad (replicated), ``v0 = v(q0)/Ω``
    and ``chi00 = Ω q0ᵀ S_direct q0``.  GATE ``epsmat_q0_wing_fold``: the
    stored head tensor must be the fold of the stored wings through this
    file's W(Γ), ``Ω q0ᵀ(S_cart_head − S_direct)q0 = wlᵀ W wr`` (sign-invariant).
    """
    h = src.head
    if h["L"] is None or h["R"] is None or h["S_direct"] is None:
        _refuse("epsmat_q0_wings_source", "a bundle without head_wing_L/R and S_cart_head_direct",
                "the GW run's ω = 0 head wings (written by gw_jax since this branch)",
                "the q → 0 wings, local-field head and 3-D body come from them",
                "rerun the producer GW on this branch, or set q0_wings = zero")
    if cfg.q0_head != "s_tensor":
        _refuse("epsmat_q0_wings_head", f"q0_head = {cfg.q0_head}",
                "q0_head = s_tensor with q0_wings = dipole",
                "the dipole head is 1/ε_M of the same wings", "set q0_head = s_tensor or q0_wings = zero")
    q0c = np.asarray(q0, dtype=np.float64) @ src.geometry.bvec
    L = np.zeros((3, src.mu_pad), np.complex128); L[:, : src.n_mu] = h["L"]
    R = np.zeros((src.mu_pad, 3), np.complex128); R[: src.n_mu] = h["R"]
    # The q-linear G = 0 density enters with a minus sign against ζ̃'s Fourier
    # convention (E9): χ₀(0, G) = −Σ_μ (q·L)_μ conj(ζ̃_μ(G)), χ₀(G, 0) = −Σ_μ ζ̃_μ(G)(R q)_μ.
    # Fixed on BerkeleyGW's own eps0mat (Si 4³ and MoS₂ 9×9): with +L every wing entry is
    # −1.0000 × BerkeleyGW's (residual 1.4e-3), the conjugated assignment fails (residual 1);
    # the head and the fold L W R are sign-invariant.
    wl, wr = -(q0c @ L), -(R @ q0c)
    omega = float(src.geometry.cell_volume)
    v0 = float(bare_v_bgw_units(src, q0, np.zeros((1, 3), int))[0]) / omega
    chi00 = omega * complex(q0c @ h["S_direct"] @ q0c)
    # Host vectors (3·μ-class) enter the replicated jit argument as epsinv_block's v does.
    wl_d, wr_d = jnp.asarray(wl, jnp.complex128), jnp.asarray(wr, jnp.complex128)
    Ww, _ = _face_matvec(mesh)(W_face, wr_d, wl_d)
    fold = complex(np.dot(wl, np.asarray(Ww)))
    stored = (omega * complex(q0c @ (h["S_cart"] - h["S_direct"]) @ q0c)
              if h["S_cart"] is not None else None)
    err = abs(stored - fold) / max(abs(fold), 1e-300) if stored is not None else float("nan")
    if stored is not None and not err <= Q0_WING_FOLD_TOL:
        _refuse("epsmat_q0_wing_fold",
                f"|Ω q0(S_cart − S_direct)q0 − wl W wr| / |wl W wr| = {err:.3e}",
                f"<= {Q0_WING_FOLD_TOL:.0e}", "the wings are not in this file's μ basis or normalisation",
                "report the bundle; export with q0_wings = zero")
    return wl_d, wr_d, v0, chi00, {"fold_file_W": fold, "fold_stored_S": stored, "fold_gate": err}


def q0_body_policy(files, sys_dim, q0_body_3d, q0_wings="zero"):
    """How ``eps0mat``'s body is written (§5), or the refusal; a receipt string.

    In 2D the head–wing term ε_b0 ε_00⁻¹ ε_0b of the block inverse is O(q)
    and the bundle's body is the q → 0 limit.  In 3D it is O(1)
    (v ∝ q⁻², χ_0G ∝ q) and absent from the bundle's body, so the file is
    written only when the deck names that approximation
    (``q0_body_3d = omit_head_wing``).  No other ``sys_dim`` has a derived body.
    """
    if "eps0mat" not in files:
        return "no eps0mat"
    sys_dim = int(sys_dim)
    if q0_wings == "dipole" and sys_dim in (2, 3):
        return "complete: head-wing term c u ũᵀ from the dipole wings (E13)"
    if sys_dim == 2:
        return "q -> 0 limit (head-wing term O(q))"
    if sys_dim != 3:
        _refuse("epsmat_q0_body_sys_dim", f"sys_dim {sys_dim} with eps0mat in files",
                "sys_dim 2 or 3", "the q → 0 body is derived for slab and bulk only",
                "export files = epsmat")
    if q0_body_3d != "omit_head_wing":
        _refuse("epsmat_q0_body_3d", "sys_dim 3 with eps0mat in files",
                "q0_body_3d = omit_head_wing, or files = epsmat",
                "in 3D the head–wing term ε_b0 ε_00⁻¹ ε_0b of the q → 0 body is O(1) "
                "and the bundle's body omits it",
                "set q0_body_3d = omit_head_wing to write that approximation (the receipt "
                "records it), or export files = epsmat")
    return "omit_head_wing (3D: the O(1) head-wing term is absent)"


def run_epsmat_export(cfg, mesh, *, print_fn=None):
    """Write ``eps0mat.h5`` / ``epsmat.h5`` from a LORRAX bundle, one q at a time.

    Every q passes three gates before its block counts: V from ζ and v equals
    the bundle's V, W(0) is Hermitian, and the written block projects back to
    ``V_ε B K Bᴴ V_ε``.  Returns a dict of per-q figures for the report.
    """
    from common.collectives import barrier, process_rank
    from file_io import bgw_epsmat as be
    from file_io.slab_io import SlabIO

    src = open_export_source(cfg, mesh)
    q0_body = q0_body_policy(cfg.files, src.config.sys_dim, cfg.q0_body_3d, cfg.q0_wings)
    log.info(f"  q0 body     {q0_body}")
    plans, comps = plan_export(cfg, src)
    os.makedirs(cfg.output_dir, exist_ok=True)
    rank0 = process_rank() == 0
    p = _square_side(mesh)
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
            algebra, parent_iq = None, -1
            # File offsets retain the requested q order. Group the computation
            # by parent so its eigensolves run once, with only one parent live.
            for r in sorted(rows, key=lambda row: row.iq_store):
                if parent_iq != r.iq_store:
                    algebra = None
                    Z, B, K, chi, V, Wc, qd = q_algebra(src, r.iq_store, mesh, rcond=cfg.rcond)
                    del B, K
                    algebra, parent_iq = (Z, chi, V, Wc, qd), r.iq_store
                Z, chi, V, Wc, qd = algebra
                n = r.gspace.nmtx
                miller, phase, anti = parent_gvectors(src, r, comps[r.gspace.order[:n]])
                slots = zeta_slots(src.zeta.gvecs(q=[r.iq_store])[0],
                                   src.zeta.ngk_valid(q=[r.iq_store])[0], miller,
                                   fill=Z.shape[2])
                v_eps = _file_q_v(src, r.q_file, comps[r.gspace.order[:n]], name == "eps0mat")
                head = _q0_head_value(cfg, src, r.q_file) if name == "eps0mat" else None
                n_pad = _pad_to(max(n, 1), p * p)
                Zf = zeta_face(Z, slots, mesh=mesh, n_pad=n_pad)
                # Same parent-centroid basis on both legs: the centroid action
                # cancels in Z chi Z^H. Only the scalar Fourier action remains.
                if r.sym_row != 0:
                    ph = np.ones(n_pad, np.complex128)
                    if phase is not None:
                        ph[:n] = phase
                    Zf = _scale_columns(mesh)[int(anti)](Zf, jnp.asarray(ph))
                    if anti:
                        chi, Wc = jnp.conj(chi), jnp.conj(Wc)
                wings, qw = None, {}
                if name == "eps0mat" and cfg.q0_wings == "dipole":
                    W_face = V + Wc
                    wl, wr, v0, chi00, qw = q0_wing_inputs(cfg, src, r.q_file, W_face, mesh=mesh)
                    head_e8 = head
                    chi, head, wings, tf = q0_dipole_terms(Zf, chi, V, W_face, wl, wr, v_eps,
                                                           mesh=mesh, v0=v0, chi00=chi00)
                    del W_face
                    qw.update(tf, head_E8=complex(head_e8), head_E10=complex(head))
                    log.info(f"    eps0mat q0 wings (dipole): ε_M = {complex(tf['eps_M']).real:.6f} "
                             f"(E8 from S_cart_head: {(1 / complex(head_e8)).real:.6f}); fold gate "
                             f"{qw['fold_gate']:.2e}; |T[a,0]|max = {float(jnp.max(jnp.abs(wings[1]))):.4e}, "
                             f"|T[0,b]|max = {float(jnp.max(jnp.abs(wings[0]))):.4e}")
                blk6, diag = epsinv_block(Zf, chi, v_eps, mesh=mesh, n_valid=n,
                                          q0_head_value=head, q0_wings=wings)
                offset, _ = be.matrix_block_window(r.iq_file, 0, n_pad)
                io.write_slab(be.MATRIX, blk6, offset=offset)
                proj = project_block(blk6, Zf, v_eps, mesh=mesh, n_valid=n,
                                     drop_head=name == "eps0mat")
                # v_eps[0] = 0 at q0, so V_ε carries no head term (the head is (E8)).
                vv = np.zeros(n_pad); vv[:n] = v_eps
                V_e = coulomb_gram(Zf, vv, mesh=mesh)
                err, rt = _check_projection(proj, V_e, chi, Wc, r.iq_store, name, mesh)
                del Zf, blk6, proj, V_e
                d = np.asarray(jax.device_get(diag))[:n]
                diags.append((r.iq_file, d))
                figs.append({"q": np.round(r.q_file, 6).tolist(), "iq_file": int(r.iq_file),
                             "iq_store": int(r.iq_store), "g0": np.asarray(r.g0).tolist(),
                             "sym_row": int(r.sym_row), "antiunitary": anti,
                             "nmtx": n, "projection_err": err, "roundtrip_vs_W": rt,
                             "head": (None if head is None else complex(head)),
                             **({"q0_wings": qw} if qw else {}), **qd})
                log.info(f"    {name}[{r.iq_file}] nmtx {n}: block identity {err:.2e}, "
                         f"Zᴴ(ε⁻¹−1)vZ vs W−V {rt:.2e}"
                         + ("" if head is None else f", head ε⁻¹_00(q0)={complex(head):.6f}"))
                del Z, chi, V, Wc
            del algebra
        if rank0:
            for iq_file, d in diags:
                be.write_q_diagonal(path, iq_file, d)
        barrier(f"epsmat.{name}.done")
        out[name] = {"path": path, "ecuts": ecuts, "q": figs,
                     **({"q0_body": q0_body} if name == "eps0mat" else {})}
    return out


# ---------------------------------------------------------------------------
# Projection: BerkeleyGW eps files -> W_μν on the bundle's stored q
# ---------------------------------------------------------------------------

#: The ``W0_source`` stamp of a projected screening: the GW driver's
#: ``screening_source = bgw_epsmat`` admits only datasets carrying it.
W_SOURCE_BGW = "bgw_epsmat"
#: GATE epsmat_project_passivity.  The whitened correlation K = Bᴴ W_c B of a
#: passive ε⁻¹ at ω = 0 or ω = iω_p lies in [−1, 0] (a compression of
#: v^½(ε⁻¹ − 1)v^½); an eigenvalue above this fraction of |K_min| is a sign,
#: orientation or basis defect, not physics.
PASSIVITY_TOL = 1.0e-3
#: The two files' frequency tables must agree to this (eV).
FREQ_MATCH_TOL_EV = 1.0e-9
#: BerkeleyGW ``icutv`` by LORRAX ``sys_dim`` (``TRUNC_NONE``, ``TRUNC_SLAB``).
ICUTV_OF_SYS_DIM_BGW = {3: 0, 2: 6}


class ProjectFrequency(NamedTuple):
    """One BerkeleyGW frequency row LORRAX consumes, and the dataset it becomes."""
    ifreq: int
    omega_ry: complex
    role: str                    # "static" (W(0)) or "probe" (the GN-PPM W(iω_p))
    dataset: str                 # "W0_qmunu" or "Wp_qmunu"


def projection_frequencies(heads):
    """The frequency rows of BerkeleyGW's files that become W_μν datasets, or refuse.

    ``freq_dep 0`` (epsilon ``frequency_dependence 0``) gives W(0) →
    ``W0_qmunu``; ``freq_dep 3`` (``frequency_dependence 3``, the Godby–Needs
    pair ω = 0 and iω_p) gives W(0) and W(iω_p) → ``Wp_qmunu``, the two
    supports of LORRAX's GN-PPM fit.  BerkeleyGW stores ``freqs (nfreq, 2)``
    as (Re, Im) in eV.  The full-frequency files (``freq_dep 2``) and two
    files with different tables refuse.
    """
    from common.units import RYD_TO_EV
    ref = None
    for name, h in heads.items():
        dep = int(h["freqs/freq_dep"])
        f = np.asarray(h["freqs/freqs"], dtype=np.float64).reshape(-1, 2)
        if dep not in (0, 3):
            _refuse("epsmat_project_freq_dep", f"{name}.h5 freq_dep {dep}",
                    "0 (static) or 3 (Godby-Needs: omega = 0 and i omega_p)",
                    "the static and GN-PPM screening read W at those frequencies only",
                    "run BerkeleyGW epsilon with frequency_dependence 0 or 3")
        tol = FREQ_MATCH_TOL_EV
        shape_ok = f.shape[0] == (1 if dep == 0 else 2) and bool(np.all(np.abs(f[0]) <= tol))
        if dep == 3 and shape_ok:
            shape_ok = abs(f[1, 0]) <= tol and f[1, 1] > tol
        if not shape_ok:
            _refuse("epsmat_project_freqs", f"{name}.h5 freqs {f.tolist()} eV at freq_dep {dep}",
                    "[(0, 0)] or [(0, 0), (0, omega_p > 0)]",
                    "row 0 is W(0) and row 1 the imaginary-axis GN probe",
                    "project an epsilon file of frequency_dependence 0 or 3")
        if ref is None:
            ref = (dep, f, name)
        elif ref[0] != dep or ref[1].shape != f.shape or np.max(np.abs(ref[1] - f)) > tol:
            _refuse("epsmat_project_freqs_differ",
                    f"{ref[2]}.h5 {ref[1].tolist()} vs {name}.h5 {f.tolist()} eV",
                    "one frequency table", "every q of a W row is one frequency",
                    "project eps0mat.h5 and epsmat.h5 of one epsilon run")
    rows = [ProjectFrequency(0, 0j, "static", "W0_qmunu")]
    if ref[0] == 3:
        rows.append(ProjectFrequency(1, 1j * float(ref[1][1, 1]) / RYD_TO_EV, "probe", "Wp_qmunu"))
    return rows


def check_bgw_crystal(src, heads):
    """GATE ``epsmat_project_crystal``: BerkeleyGW's files describe the bundle's crystal.

    Same q grid, the same cell volume, and BerkeleyGW's Coulomb truncation
    (``icutv``) equal to the producer's ``sys_dim`` (0 bulk, 6 slab); the q
    rows and G lists are matched one by one later.
    """
    sys_dim = int(src.config.sys_dim)
    want_icutv = ICUTV_OF_SYS_DIM_BGW.get(sys_dim)
    for name, h in heads.items():
        grid = tuple(int(x) for x in np.asarray(h["qpoints/qgrid"]).reshape(-1))
        celvol = float(h["celvol"])
        icutv = int(h["params/icutv"])
        bad = []
        if grid != tuple(int(k) for k in src.kgrid):
            bad.append(f"qgrid {grid} vs the bundle's {tuple(src.kgrid)}")
        if abs(celvol - float(src.geometry.cell_volume)) > 1e-8 * float(src.geometry.cell_volume):
            bad.append(f"cell volume {celvol} vs {float(src.geometry.cell_volume)} bohr^3")
        if want_icutv is None or icutv != want_icutv:
            bad.append(f"icutv {icutv} vs sys_dim {sys_dim} (want icutv {want_icutv})")
        if bad:
            _refuse("epsmat_project_crystal", f"{name}.h5: " + "; ".join(bad),
                    "the producer's crystal, q grid and Coulomb truncation",
                    "W = ε⁻¹ v is projected with LORRAX's v on the bundle's ζ",
                    "run BerkeleyGW epsilon on the bundle's WFN with the matching truncation")


class ImportRow(NamedTuple):
    """One stored q and the BerkeleyGW block that gives it (theory §4)."""
    iq_store: int
    name: str                    # "eps0mat" (Γ, at BerkeleyGW's q0) or "epsmat"
    iq_file: int
    g0: np.ndarray               # sym_row 0: ζ̃_{q_file}(G) = ζ̃_{q_store}(G + g0)
    sym_row: int                 # 0, or the SymMaps row with q_store ≡ S q_file


def import_q_plan(src, heads):
    """The BerkeleyGW block of every stored q, or refuse naming every uncovered q.

    Γ is ``eps0mat`` row 0.  Any other stored q takes an ``epsmat`` row that
    is the same q up to a reciprocal lattice vector, else one that a
    symmetry of the WFN maps onto it (``SymMaps.operation_rows``; the
    antiunitary rows only on a measured time-reversal-invariant deck), the
    way BerkeleyGW's Sigma unfolds its own irreducible q list.
    """
    sym = src.sym
    n_sym = int(np.asarray(sym.sym_matrices).shape[0])
    rows = np.arange(2 * n_sym if bool(getattr(sym, "trs_allowed", False)) else n_sym,
                     dtype=np.int32)
    S_all = np.asarray(sym.operation_rows(rows)[0], dtype=np.float64)
    q_file = heads["epsmat"]["qpts"] if "epsmat" in heads else np.zeros((0, 3))
    gamma = set(np.flatnonzero(np.all(np.abs(src.q_stored) < 1e-12, axis=1)).tolist())
    plan, missing = [], []
    for iq, q in enumerate(np.asarray(src.q_stored, dtype=np.float64)):
        if iq in gamma:
            plan.append(ImportRow(iq, "eps0mat", 0, np.zeros(3, np.int64), 0))
            continue
        if q_file.shape[0] == 0:
            missing.append(q)
            continue
        idx, g0s = match_q_umklapp(q_file, q[None])
        if int(idx[0]) >= 0:
            plan.append(ImportRow(iq, "epsmat", int(idx[0]), -g0s[0], 0))
            continue
        d = q[None, None, :] - np.einsum("rij,fj->rfi", S_all, q_file)
        hit = np.all(np.abs(d - np.rint(d)) < 1e-6, axis=-1)
        if not hit.any():
            missing.append(q)
            continue
        r, f = (int(x) for x in np.argwhere(hit)[0])
        plan.append(ImportRow(iq, "epsmat", f, np.zeros(3, np.int64), int(rows[r])))
    if missing:
        _refuse("epsmat_project_q_coverage",
                f"{len(missing)} stored q without a BerkeleyGW row: "
                f"{np.round(np.asarray(missing), 6).tolist()[:8]}",
                "every stored q an umklapp or symmetry image of an epsmat.h5 q (Γ: eps0mat.h5)",
                "W is projected per stored q; a missing q has no screening",
                "run BerkeleyGW epsilon on the irreducible q list of the bundle's WFN and grid")
    return plan


def import_gvectors(src, row, q_file, g_file):
    """Stored-q G of each file G, the Fourier phase and the TR bit (theory §4).

    ``G_s = S(q_file + G_f) − q_store`` for the row's action S; then
    ``ε⁻¹_{q_store}(G_s, G_s') = φ_s ε̃(G_f, G_f') φ̄_s'`` with ε̃ the file's
    block (conjugated on an antiunitary row) and φ the export's own phase
    (:func:`parent_gvectors`, which must map every G_s back to its G_f).
    Returns ``(G_s (n, 3) int, φ (n,) or None, anti)``.
    """
    from types import SimpleNamespace
    if row.sym_row == 0:
        return np.asarray(g_file) + np.asarray(row.g0)[None, :], None, False
    S = np.asarray(src.sym.operation_rows(int(row.sym_row))[0], dtype=np.float64)
    q_s = np.asarray(src.q_stored[row.iq_store], dtype=np.float64)
    gs_f = (np.asarray(q_file)[None, :] + np.asarray(g_file)) @ S.T - q_s[None, :]
    g_s = np.rint(gs_f).astype(np.int64)
    plan = QPlan(0, 0, q_s, np.zeros(3, np.int64), None, int(row.sym_row))
    back, phase, anti = parent_gvectors(
        SimpleNamespace(sym=src.sym, q_stored=np.asarray(q_file)[None]), plan, g_s)
    if (not np.allclose(gs_f, g_s, rtol=0.0, atol=1e-6)
            or not np.array_equal(np.asarray(back), np.asarray(g_file))):
        _refuse("epsmat_project_image", f"stored q {np.round(q_s, 6).tolist()} from file q "
                f"{np.round(q_file, 6).tolist()} by row {row.sym_row}",
                "an exact, invertible G relabel", "the rotated block would mix G vectors",
                "report the WFN symmetry header")
    return g_s, phase, bool(anti)


def import_zeta_face(Z, slots, phase, anti, *, mesh, n, n_pad):
    """ζ̃ on the file's G list in the file block's frame: ``(φ̄ ζ̃)^(*)``, a face.

    With ε⁻¹_s = Φ ε̃ Φᴴ (:func:`import_gvectors`) and W_c = Zᴴ(ε⁻¹_s − 1)vZ,
    the phase moves onto ζ̃ as ``Z' = Φᴴ Z``; on an antiunitary row the
    block is ε̃ = conj(ε_f), so ``W_c = conj(Z''ᴴ (ε_f − 1) v Z'')`` with
    ``Z'' = conj(Z')`` (the caller conjugates the projected W_c).
    """
    Zf = zeta_face(Z, slots, mesh=mesh, n_pad=n_pad)
    if phase is not None:
        ph = np.ones(n_pad, np.complex128)
        ph[:n] = np.conj(np.asarray(phase))
        Zf = _scale_columns(mesh)[0](Zf, jnp.asarray(ph))
    return jnp.conj(Zf) if anti else Zf


def projection_price(mu_pad, n_pad, ranks):
    """Per-rank device bytes of one stored q of the projection (``lxkit.memory.Price``).

    Resident: V, W_c, V + W_c, B and K (five μ faces, 16 μ_pad²/P each), the
    file block (16 n_pad²/P) and ζ̃ on the eps G list with its phased copy
    (2 × 16 μ_pad n_pad/P); fixed: the two live SUMMA panels
    (``PANEL_BYTES`` each).  Theory §8.
    """
    from lxkit import memory
    spread = 5 * 16.0 * mu_pad ** 2 + 16.0 * n_pad ** 2 + 2 * 16.0 * mu_pad * n_pad
    return memory.priced(2.0 * PANEL_BYTES, "epsmat projection: two SUMMA panels",
                         spread=spread, ranks=int(ranks))


def bgw_file_fingerprint(path, head):
    """Provenance of one BerkeleyGW file: path, size, mtime and a header digest."""
    import hashlib
    import json
    st = os.stat(path)
    h = hashlib.sha256()
    for key in ("qpts", "nmtx", "freqs/freqs", "params/ecuts", "params/icutv", "qpoints/qgrid"):
        h.update(np.ascontiguousarray(np.asarray(head[key])).tobytes())
    return json.dumps({"path": os.path.abspath(path), "bytes": int(st.st_size),
                       "mtime": int(st.st_mtime), "header_sha256": h.hexdigest()})


def _copy_qirr_stamp(f, src_name, dst_name):
    """Give ``dst_name`` the q-wedge stamp of ``src_name`` (same q rows and centroids).

    The projected W rows are written on V's stored rows, so V's unfold tables
    and closure receipt describe them exactly; the table group is copied,
    not rebuilt, so the two digests agree by construction.
    """
    from symmetry_maps import QIRR_TABLE_SUFFIX
    if src_name + QIRR_TABLE_SUFFIX not in f:
        return
    if dst_name + QIRR_TABLE_SUFFIX in f:
        del f[dst_name + QIRR_TABLE_SUFFIX]
    f.copy(f[src_name + QIRR_TABLE_SUFFIX], dst_name + QIRR_TABLE_SUFFIX)
    for key, val in f[src_name].attrs.items():
        if str(key) == "q_storage" or str(key).startswith("qirr_"):
            f[dst_name].attrs[key] = val
    f[dst_name].attrs["qirr_data_ready"] = True


def run_epsmat_projection(cfg, mesh, *, print_fn=None):
    """Project BerkeleyGW's ε⁻¹ onto the bundle's ζ and write a variant bundle.

    For every stored q the BerkeleyGW block that gives it (an umklapp or a
    symmetry image, :func:`import_q_plan`; Γ is the eps0mat q0) is read tile
    by tile, at every frequency the files carry (:func:`projection_frequencies`),
    and a COPY of the bundle in ``output_dir/tmp`` receives
    ``W_ω = V + Z_εᴴ [(ε⁻¹_ω − 1) diag(v)] Z_ε`` (E7): ``W0_qmunu`` at ω = 0
    and, from a Godby–Needs file, ``Wp_qmunu`` at iω_p.  At Γ the G = 0 row
    and column are dropped; ``bgw_epsinv_head`` keeps BerkeleyGW's
    ε⁻¹_00(q0, ω) for the GW driver's head (``screening_source = bgw_epsmat``),
    and the producer's ``whead`` stays for the BSE.  LORRAX's own W(0) is
    not needed; when the bundle has one the receipt compares the two.
    Returns the receipt dict.
    """
    import shutil
    import h5py
    from common.collectives import barrier, process_rank
    from common.gpu_utils import device_budget_bytes
    from lxkit import memory
    from file_io import bgw_epsmat as be
    from file_io.slab_io import SlabIO

    src = open_export_source(cfg, mesh, require_w0=False)
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
        undone = np.flatnonzero(np.asarray(h["qpoints/qpt_done"]).reshape(-1) == 0)
        if undone.size:
            _refuse("epsmat_project_incomplete", f"{name}.h5 q rows {undone.tolist()} not done",
                    "qpt_done = 1 on every row", "an unfinished row holds zeros, not ε⁻¹",
                    "finish the BerkeleyGW epsilon run")
    check_bgw_crystal(src, heads)
    freqs = projection_frequencies(heads)
    plan = import_q_plan(src, heads)
    n_rot = sum(r.sym_row != 0 for r in plan)
    n_g = sum(r.name == "eps0mat" for r in plan)
    log.info("  frequencies " + ", ".join(
        f"{f.dataset} <- {f.role} ω = {f.omega_ry:.6f} Ry" for f in freqs))
    log.info(f"  q plan      {len(plan)} stored q: {n_g} from eps0mat, "
             f"{len(plan) - n_rot - n_g} umklapp and {n_rot} symmetry images from epsmat")
    log.info(f"  LORRAX W(0) {'in the bundle: compared' if src.w0_ready else 'absent: not compared'}")

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

    p = _square_side(mesh)
    n_max = max(int(heads[r.name]["nmtx"][r.iq_file]) for r in plan)
    device_budget_bytes()
    price = projection_price(src.mu_pad, _pad_to(n_max, p * p), int(mesh.devices.size))
    memory.admit("epsmat.project", (("faces", price),), memory.room(bucketed=False))
    log.info(f"  memory      {price.bytes() / 2**30:.3f} GiB per device per q "
             f"(nmtx_max {n_max}, μ carrier {src.mu_pad}; {price.describe()})")

    with h5py.File(files["eps0mat"], "r") as f:
        heads_bgw = [complex(f[be.MATRIX][0, 0, fr.ifreq, 0, 0, 0],
                             f[be.MATRIX][0, 0, fr.ifreq, 0, 0, 1]) for fr in freqs]
    figs = []
    with SlabIO(target, mode="a", mesh=mesh) as io:
        # Register (W0) or create (Wp, or W0 of a bundle without screening):
        # the parallel tier writes only datasets its handle created.
        for fr in freqs:
            io.create_dataset(fr.dataset, shape=(src.q_stored.shape[0], src.n_mu, src.n_mu),
                              dtype=np.complex128)
        for row in plan:
            iq, is_g = row.iq_store, row.name == "eps0mat"
            h = heads[row.name]
            Z, V, W_lx, basis = q_basis(src, iq, mesh, with_w=src.w0_ready)
            B, _ = coulomb_whitener(V, rcond=cfg.rcond, plan=src.eigh)
            qg, _ = be.read_q_gspace(files[row.name], row.iq_file)
            n = qg.nmtx
            g_file = h["components"][qg.order[:n]]
            q_file = np.asarray(h["qpts"][row.iq_file], dtype=np.float64)
            g_s, phase, anti = import_gvectors(src, row, q_file, g_file)
            sphere = src.zeta.gvecs(q=[iq])[0]
            ngk = int(src.zeta.ngk_valid(q=[iq])[0])
            slots = zeta_slots(sphere, ngk, g_s, fill=Z.shape[2])
            outside = int(np.count_nonzero(slots >= ngk))
            if outside:
                _refuse("epsmat_project_sphere",
                        f"{outside} of {n} BerkeleyGW G at stored q {iq} outside the ζ sphere",
                        "BerkeleyGW's ecuts within the producer's bare-Coulomb sphere",
                        "ζ̃ is zero there, so that part of ε⁻¹ would be dropped silently",
                        "rerun epsilon with a smaller epsilon_cutoff, or GW with a larger "
                        "bare_coulomb_cutoff")
            v_eps = (_file_q_v(src, q_file, g_file, True) if is_g
                     else lorrax_v(src, src.q_stored[iq], g_s))
            n_pad = _pad_to(n, p * p)
            Zf = import_zeta_face(Z, slots, phase, anti, mesh=mesh, n=n, n_pad=n_pad)
            row_figs = {"q": np.round(src.q_stored[iq], 6).tolist(), "iq_store": int(iq),
                        "file": row.name, "iq_file": int(row.iq_file),
                        "q_file": np.round(q_file, 6).tolist(), "g0": np.asarray(row.g0).tolist(),
                        "sym_row": int(row.sym_row), "antiunitary": anti, "nmtx": n,
                        "err_v": basis["err_v"]}
            for fr in freqs:
                offset, shape = be.matrix_block_window(row.iq_file, fr.ifreq, n_pad)
                with SlabIO(files[row.name], mode="r", mesh=mesh) as rd:
                    blk = rd.read_slab(be.MATRIX, shape=shape, dtype=np.float64, offset=offset,
                                       mesh=mesh, partition_spec=P(None, None, None, "x", "y", None))
                Wc = project_block(blk, Zf, v_eps, mesh=mesh, n_valid=n, drop_head=is_g)
                del blk
                if anti:
                    Wc = jnp.conj(Wc)
                herm = (2.0 * float(jnp.linalg.norm(Wc - hermitian(Wc)))
                        / max(float(jnp.linalg.norm(Wc)), 1e-300))
                ev = face_spectrum(whitened_correlation(Wc, B, mesh=mesh), src.eigh)
                if not np.all(np.isfinite(ev)) or ev[-1] > PASSIVITY_TOL * max(abs(ev[0]), 1e-300):
                    _refuse("epsmat_project_passivity",
                            f"K = Bᴴ W_c B in [{ev[0]:.4e}, {ev[-1]:.4e}] at stored q {iq}, "
                            f"{fr.role} ω = {fr.omega_ry:.4f} Ry",
                            f"K_max <= {PASSIVITY_TOL:.0e} |K_min| (−1 ≼ K ≼ 0)",
                            "a passive ε⁻¹ projects to a non-positive correlation; a positive "
                            "eigenvalue is a sign, orientation or basis defect",
                            "report the files and the bundle; do not consume the output")
                rel = None
                if fr.role == "static" and W_lx is not None:
                    Wc_lx = W_lx - V
                    rel = float(jnp.linalg.norm(Wc - Wc_lx)) / float(jnp.linalg.norm(Wc_lx))
                io.write_slab(fr.dataset, V + Wc, offset=(iq, 0, 0))
                row_figs[fr.role] = {"omega_ry": fr.omega_ry, "herm": herm,
                                     "k_min": float(ev[0]), "k_max": float(ev[-1]),
                                     "rel_Wc_bgw_vs_lorrax": rel}
                log.info(f"    stored q {iq} <- {row.name}[{row.iq_file}]"
                         + (f" by row {row.sym_row}{' (TR)' if anti else ''}" if row.sym_row else "")
                         + f" nmtx {n} {fr.role}: K∈[{ev[0]:.4f}, {ev[-1]:.1e}], "
                         f"|Wc − h.c.|/|Wc| = {herm:.1e}"
                         + ("" if rel is None else f", |Wc_BGW − Wc_LX|/|Wc_LX| = {rel:.4e}"))
                del Wc
            figs.append(row_figs)
            del Z, V, W_lx, B, Zf
        stamp = {"W0_source": W_SOURCE_BGW,
                 "bgw_reference": os.path.abspath(cfg.bgw_reference),
                 "bgw_files": "[" + ", ".join(bgw_file_fingerprint(files[k], heads[k])
                                              for k in sorted(heads)) + "]",
                 "bgw_ecuts_ry": float(heads["eps0mat"]["params/ecuts"]),
                 "bgw_q0": np.asarray(heads["eps0mat"]["qpts"][0], dtype=np.float64),
                 "q0_head_note": "G = 0 row/column dropped at Gamma; BerkeleyGW's head is "
                                 "bgw_epsinv_head, the producer's whead is kept for the BSE"}
        for fr in freqs:
            io.stamp_dataset_attrs(fr.dataset, {**stamp, "omega_ry_re": fr.omega_ry.real,
                                                "omega_ry_im": fr.omega_ry.imag,
                                                "screening_role": fr.role,
                                                **({"W0_ready": True} if fr.role == "static" else {})})
    if rank0:
        with h5py.File(target, "a") as f:
            for fr in freqs:
                _copy_qirr_stamp(f, "V_qmunu", fr.dataset)
            if "bgw_epsinv_head" in f:
                del f["bgw_epsinv_head"]
            d = f.create_dataset("bgw_epsinv_head", data=np.asarray(heads_bgw, np.complex128))
            d.attrs["omega_ry_re"] = np.asarray([fr.omega_ry.real for fr in freqs])
            d.attrs["omega_ry_im"] = np.asarray([fr.omega_ry.imag for fr in freqs])
            d.attrs["roles"] = ",".join(fr.role for fr in freqs)
            d.attrs["bgw_q0"] = np.asarray(heads["eps0mat"]["qpts"][0], dtype=np.float64)
    barrier("epsmat.project.done")
    log.info("  BerkeleyGW ε⁻¹_00(q0): " + ", ".join(
        f"{fr.role} {hb:.6f}" for fr, hb in zip(freqs, heads_bgw)))
    return {"path": target, "frequencies": [{"role": fr.role, "dataset": fr.dataset,
                                              "omega_ry": fr.omega_ry} for fr in freqs],
            "bgw_epsinv_head": heads_bgw, "q": figs}
