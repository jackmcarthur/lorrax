"""The TD-aGW interaction model: ΔΣ_k[δρ] on the ISDF basis.

``build_td_agw_apply(mesh_xy, model)`` returns ``apply(drho) -> dsigma``,

    ΔΣ_{k,nm} = Σ_{k'n'm'} (K^H − K^W)_{knm,k'n'm'} δρ_{k',n'm'}        (Ry)

with K^W the screened direct term of the TDA BSE matvec
(``bse_stack_matvec.make_w_term``) on both orbital legs = the TD band window,
and K^H the Hartree term in G space through the q = 0 interpolation vectors,
G = 0 excluded (``docs/theory/td_agw.md``).

The model bundle
----------------
A plain dict, the same whatever produced it (a GW restart, a densified BSE
bundle, a synthetic test).  ``assemble_td_model`` is its one constructor.

``psi_X``  (nk, nb_pad, ns, nmu_pad) c128, ``P(None,None,None,'x')`` —
           ψ_{nk,s}(r_μ) of the window, k in flat C order of ``kgrid``.
``psi_Y``  the same values at ``P(None,None,None,'y')``.
``M``      (nk, nb_pad, nb_pad, nmu_pad) c128, ``P(None,'x','y',None)`` —
           M_{k,nm}(μ) = Σ_s ψ*_{nk,s}(μ) ψ_{mk,s}(μ).
``W_R``    (nmu_pad, nmu_pad, nkx, nky, nkz) c128, ``P('x','y',None,None,None)``
           — static W(q) with its q = 0 head, inverse-FFT'd to R
           (``bse_feast.ensure_W_R``), Ry.
``zeta_G`` (nmu_pad, nG_pad) c128, ``P(None,('x','y'))`` — ζ̃_{0μ}(G).
``v_G``    (nG_pad,) f64, ``P(('x','y'))`` — v(G) at q = 0, Ry, zero at G = 0
           and on pad slots.
``kgrid``, ``nspinor``, ``n_bands`` (logical window), ``band_lo`` (absolute
0-based first band), ``n_rmu`` (logical centroids), ``k_frac`` (nk, 3) — the
k of each row as sampled, ``receipts`` — provenance and checks.

The state ``drho`` / ``dsigma`` is (b, nk, nb_pad, nb_pad) c128 at
``P(None,None,'x','y')``: b density matrices (trials), k whole, row band on
'x', column band on 'y'.  Pad rows and columns are inert (ψ = 0 there).

Per-rank memory (P ranks, √P × √P mesh; see the theory page's cost table):
``16·[2·nk·nb·ns·Nμ/√P + nk·nb²·Nμ/P + nk·Nμ²/P + Nμ·N_G/P + b·nk·nb²]`` B,
plus ``16·nk·ns²·Nμ²/P`` for one trial's T on the routes that store it.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import shard_map as _shard_map_fn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse.bse_preconditioner import compute_pair_amplitude, exchange_spin_weight
from bse.bse_ring_comm import make_bse_shardings
from bse.bse_stack_matvec import exchange_decode, exchange_encode, make_w_term
from runtime.padding import pad_axis, padded_mu_extent, round_up

__all__ = [
    "TD_STATE_SPEC", "assemble_td_model", "build_td_agw_apply",
    "dense_td_kernel", "flat_k_rows", "hartree_operator_from_zeta",
    "load_td_model", "pad_td_state", "sample_window_psi", "strip_td_state",
    "td_state_sharding",
]

#: (b, k, n, m): trials and k whole, row band on 'x', column band on 'y'.
TD_STATE_SPEC = P(None, None, "x", "y")
#: ζ̃_0(μ, G): G over every rank.
ZETA_G_SPEC = P(None, ("x", "y"))
V_G_SPEC = P(("x", "y"))


#: Why each gate refuses: one reason per gate name, read by :func:`_refuse`.
_WHY = {
    "td_model_terms": "an operator with neither term is identically zero",
    "td_model_k_axis": "the W term reshapes the flat k axis onto the grid, so a partial k list "
                       "pairs the wrong k - k'",
    "td_model_mu_axis": "every centroid-indexed operand must share one padded extent, or the "
                        "mesh shards misalign their rows",
    "td_model_w_grid": "the k-convolution pairs each psi row with the W_R cell of the same grid",
    "td_k_on_grid": "the W convolution needs k on the unshifted grid, and a nearest-k fallback "
                    "would mislabel rows",
    "td_k_full_grid": "the k-convolution's FFT needs every grid point exactly once",
    "td_wfn_full_grid": "psi is sampled at every grid k, and an IBZ WFN would need an unfolding "
                        "this path does not do",
    "td_zeta_q0_row": "the Hartree term drops exactly the G = 0 slot of the q = 0 sphere",
    "td_band_origin": "bands are absolute indices, and a stored window that starts elsewhere "
                      "would offset psi against W",
    "td_window_gap": "rho_eq = diag(f) needs occupied and empty bands in the window",
    "td_window_resolved": "the loader moved a window edge to close a degenerate multiplet, so "
                          "psi would not be the requested bands",
    "td_wfn_densified": "the densified psi come from the htransform, and TDGW's coarse WFN is "
                        "another basis",
    "td_centroids_md5": "psi sampled at other centroids does not match the restart's W and zeta",
    "td_wfn_geometry": "psi(r_mu) from another FFT grid or cell falls on other points than the "
                       "restart's centroids",
    "td_velocity_full_window": "a full-window velocity is read from a velocity file",
    "td_velocity_loader": "this loader cannot carry the velocity, and dropping the request would "
                          "return no dipoles",
    "td_velocity_route": "the loader reads the velocity on the densified route only, so elsewhere "
                         "the request would be ignored",
    "td_velocity_full_window_loader": "this loader returns the window's cv dipoles only, so the "
                                      "vv' and cc' pairs would be missing",
    "td_eqp_n_occ": "psi hold the DFT window, and a QP spectrum with another n_occ changes which "
                    "bands are occupied",
    "td_coulomb_policy": "the Hartree term must use the Coulomb kernel that V and W were built "
                         "with, which the stamp names",
    "td_coulomb_bgw_vcoul": "compute_v_q_per_G does not reproduce BerkeleyGW's vcoul overlay, so "
                            "Vbar would differ from the stored V",
    "td_zeta_gamma_row": "the Hartree term reads the q = 0 row, and another row is a finite-q "
                         "operator",
}


def _refuse(gate: str, got, want, fix: str):
    raise ValueError(f"GATE {gate}: got: {got}; want: {want}; why: {_WHY[gate]}; fix: {fix} "
                     f"(docs/theory/td_agw.md)")


def td_state_sharding(mesh_xy: Mesh) -> NamedSharding:
    """The ``(b, k, n, m)`` density-matrix sharding, ``P(None,None,'x','y')``."""
    return NamedSharding(mesh_xy, TD_STATE_SPEC)


# ===========================================================================
# The operator
# ===========================================================================

def _make_hartree_term(mesh_xy: Mesh):
    """Hartree term through ζ̃_0(G), one ``shard_map`` over ('x','y').

    ``h(X, M, zeta_G, v_G, enc, dec) -> (w_s/N_k) Σ_μ M[k,n,m,μ] U[b,μ]`` with

        δn_ν = Σ_{knm} M*[k,n,m,ν] X[b,n,m,k]                     (encode)
        U_μ  = Σ_{G≠0} ζ̃*_μ(G) v(G) Σ_ν ζ̃_ν(G) δn_ν              (G space)

    ``X`` and ``M`` on the transition layout ``P(None,'x','y',None)``;
    ``zeta_G`` (Nμ, G) and ``v_G`` (G,) with G over every rank, so δn(G) is
    local per G slab.  Two psums of a (b, Nμ) vector; no Nμ² object exists.
    ``enc`` carries w_s/√N_k and ``dec`` 1/√N_k (the BSE exchange split).
    """
    xspec = P(None, "x", "y", None)

    def _h(X, M, zeta_G, v_G, enc, dec):
        S = lax.psum(exchange_encode(X, M) * enc, ("x", "y"))       # (b, μ)
        dn = jnp.einsum("NG,bN->bG", zeta_G, S)                     # this rank's G slab
        U = lax.psum(jnp.einsum("MG,bG->bM", jnp.conj(zeta_G),
                                dn * v_G[None, :]), ("x", "y"))     # (b, μ)
        return exchange_decode(M, U) * dec

    return _shard_map_fn(
        _h, mesh=mesh_xy,
        in_specs=(xspec, xspec, ZETA_G_SPEC, V_G_SPEC, P(), P()),
        out_specs=xspec, check_vma=False)


@functools.lru_cache(maxsize=None)
def _td_program(mesh_xy: Mesh, kgrid: tuple, nspinor: int, sex: bool, hartree: bool,
                backend: str):
    """The jitted ``(drho, psi_X, psi_Y, M, W_R, zeta_G, v_G) -> dsigma``, one per static signature.

    ``backend`` is the k-convolution router's choice when the program is
    built (``ffi.fft.kconv_backend``): it is part of the key so a program
    built inside ``ffi.fft.xla_reference()`` is never served outside it.
    """
    if not (sex or hartree):
        _refuse("td_model_terms", "sex=False, hartree=False", "at least one term",
                "request the SEX term, the Hartree term or both")
    nkx, nky, nkz = (int(n) for n in kgrid)
    nk = nkx * nky * nkz
    sh = make_bse_shardings(mesh_xy)
    st = td_state_sharding(mesh_xy)
    w_term = make_w_term(mesh_xy, nkx, nky, nkz) if sex else None
    h_term = _make_hartree_term(mesh_xy) if hartree else None
    w_s = exchange_spin_weight(nspinor)

    def _apply(drho, psi_X, psi_Y, M, W_R, zeta_G, v_G):
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=drho.real.dtype))
        X = lax.with_sharding_constraint(jnp.moveaxis(drho, 1, -1), sh.X)   # (b, n, m, k)
        out = jnp.zeros_like(X)
        if hartree:
            out = out + h_term(X, M, zeta_G, v_G, w_s / sqrt_nk, 1.0 / sqrt_nk)
        if sex:
            out = out - w_term(X, psi_X, psi_Y, W_R)
        return jnp.moveaxis(out, -1, 1)

    zsh = NamedSharding(mesh_xy, ZETA_G_SPEC)
    vsh = NamedSharding(mesh_xy, V_G_SPEC)
    return jax.jit(_apply,
                   in_shardings=(st, sh.psi_x, sh.psi_y, sh.M, sh.W, zsh, vsh),
                   out_shardings=st)


def build_td_agw_apply(mesh_xy: Mesh, model: dict, *, sex: bool = True,
                       hartree: bool = True):
    """``apply(drho) -> dsigma``: the TD-aGW self-energy update on ``model``.

    Computes ΔΣ_{k,nm} = Σ_{k'n'm'} (s_H K^H − s_W K^W)_{knm,k'n'm'} δρ_{k',n'm'}
    (``docs/theory/td_agw.md``, kernel), s_H = ``hartree``, s_W = ``sex``.

    Parameters
    ----------
    mesh_xy : Mesh
        The run's square ('x','y') mesh; every operand of ``model`` lives on it.
    model : dict
        The bundle of :func:`assemble_td_model`.
    sex, hartree : bool
        Keep the screened-exchange (−W) and the Hartree (+w_s V̄) terms.

    Returns
    -------
    apply : callable
        ``apply(drho)``: ``drho`` (b, nk, nb_pad, nb_pad) complex128 at
        :data:`TD_STATE_SPEC`, k in the model's flat k order; returns
        ΔΣ of the same shape and sharding, Ry.  One compiled program per
        static signature (mesh, k grid, spinor extent, terms) and per shape
        of ``drho``; the model's arrays are runtime operands.
    """
    from ffi.fft import kconv_backend
    prog = _td_program(mesh_xy, tuple(int(n) for n in model["kgrid"]),
                       int(model["nspinor"]), bool(sex), bool(hartree), kconv_backend(mesh_xy))
    ops = (model["psi_X"], model["psi_Y"], model["M"], model["W_R"],
           model["zeta_G"], model["v_G"])

    def apply(drho):
        return prog(drho, *ops)
    return apply


# ===========================================================================
# The bundle
# ===========================================================================

def assemble_td_model(psi, W_R, zeta_G, v_G, *, mesh_xy: Mesh, kgrid, k_frac,
                      band_lo: int, n_bands: int, n_rmu: int, receipts=None) -> dict:
    """The model bundle from its four physical operands.

    Parameters
    ----------
    psi : jax.Array
        (nk, n_bands or more, ns, nmu_pad) c128 — ψ of the window at the
        centroids, k in flat C order of ``kgrid``; μ already padded with
        zeros to ``padded_mu_extent(n_rmu, mesh_xy)``.  Bands past
        ``n_bands`` are dropped, then the band axis is zero-padded to the mesh.
    W_R : jax.Array
        (nmu_pad, nmu_pad, nkx, nky, nkz) at ``P('x','y',None,None,None)``, Ry.
    zeta_G, v_G : jax.Array
        (nmu_pad, nG_pad) at ``P(None,('x','y'))`` and (nG_pad,) at
        ``P(('x','y'))``, v in Ry with v(G=0) = 0
        (:func:`hartree_operator_from_zeta`).
    kgrid : (3,) ints;  k_frac : (nk, 3) float — k of each psi row.
    band_lo, n_bands, n_rmu : int — absolute first band (0-based), window
        width and logical centroid count.
    """
    kgrid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(kgrid))
    if int(psi.shape[0]) != nk:
        _refuse("td_model_k_axis", f"psi k extent {psi.shape[0]}",
                f"prod(kgrid) = {nk}", "sample ψ on the full k grid")
    nmu_pad = padded_mu_extent(int(n_rmu), mesh_xy)
    for name, a, ax in (("psi", psi, 3), ("W_R", W_R, 0), ("zeta_G", zeta_G, 0)):
        if int(a.shape[ax]) != nmu_pad:
            _refuse("td_model_mu_axis", f"{name} μ extent {a.shape[ax]}",
                    f"{nmu_pad} (n_rmu={n_rmu} padded to the mesh)",
                    "pad μ with zeros to padded_mu_extent before assembly")
    if tuple(int(n) for n in W_R.shape[2:]) != kgrid:
        _refuse("td_model_w_grid", f"W_R k grid {tuple(W_R.shape[2:])}",
                f"{kgrid}", "densify W to the model's k grid first")
    sh = make_bse_shardings(mesh_xy)
    px = int(mesh_xy.shape["x"])
    psi_X = jax.jit(lambda a: pad_axis(a[:, :int(n_bands)], px, axis=1).array,
                    out_shardings=sh.psi_x)(psi)
    psi_Y = jax.jit(lambda a: a, out_shardings=sh.psi_y)(psi_X)
    M = jax.jit(compute_pair_amplitude, out_shardings=sh.M)(psi_X, psi_X)
    return dict(psi_X=psi_X, psi_Y=psi_Y, M=M, W_R=W_R, zeta_G=zeta_G, v_G=v_G,
                kgrid=kgrid, nspinor=int(psi.shape[2]), n_bands=int(n_bands),
                nb_pad=int(psi_X.shape[1]), band_lo=int(band_lo),
                n_rmu=int(n_rmu), k_frac=np.asarray(k_frac, dtype=np.float64),
                units="Ry", receipts=dict(receipts or {}))


def pad_td_state(drho: np.ndarray, model: dict) -> np.ndarray:
    """Host (b, nk, n_bands, n_bands) → the carrier (b, nk, nb_pad, nb_pad), zero pad."""
    nb, nbp = int(model["n_bands"]), int(model["nb_pad"])
    out = np.zeros(drho.shape[:2] + (nbp, nbp), dtype=np.complex128)
    out[:, :, :nb, :nb] = drho
    return out


def strip_td_state(x: np.ndarray, model: dict) -> np.ndarray:
    """Carrier (b, nk, nb_pad, nb_pad) → the logical (b, nk, n_bands, n_bands)."""
    nb = int(model["n_bands"])
    return np.asarray(x)[:, :, :nb, :nb]


def flat_k_rows(k_frac, kgrid, *, tol: float = 1e-6) -> np.ndarray:
    """Flat C-order grid row of each fractional k; refuses off-grid or repeated k.

    ``rows[i] = ix·nky·nkz + iy·nkz + iz`` with ``k_frac[i] ≡ (ix/nkx, iy/nky,
    iz/nkz) mod 1``.  The k-convolution of the W term runs on this order, so
    a shifted or incomplete grid cannot be mapped and refuses (TASTE 4: no
    nearest-k fallback).
    """
    kgrid = np.asarray([int(n) for n in kgrid])
    k = np.asarray(k_frac, dtype=np.float64)
    idx = k * kgrid[None, :]
    rnd = np.round(idx)
    if np.max(np.abs(idx - rnd), initial=0.0) > tol:
        _refuse("td_k_on_grid", f"max |k·N - round| = {np.max(np.abs(idx - rnd)):.2e}",
                f"an unshifted {tuple(kgrid)} grid (tol {tol})",
                "use a Γ-centred full-grid WFN")
    ijk = rnd.astype(np.int64) % kgrid[None, :]
    rows = (ijk[:, 0] * kgrid[1] + ijk[:, 1]) * kgrid[2] + ijk[:, 2]
    if not np.array_equal(np.sort(rows), np.arange(int(np.prod(kgrid)))):
        _refuse("td_k_full_grid", f"{len(set(rows.tolist()))} distinct grid rows of {len(rows)} k",
                f"every one of the {int(np.prod(kgrid))} grid points once",
                "use a full-grid (nosym) WFN")
    return rows


def dense_td_kernel(apply, model: dict, *, mesh_xy: Mesh, batch: int = 64) -> np.ndarray:
    """K_{knm,k'n'm'} (Ry, model k order, logical bands) by applying ``apply`` to unit δρ.

    Column (k', n', m') of K is ``apply(e_{k'n'm'})``: the dense kernel IS the
    model's operator, sampled, so a propagator using K and one calling
    ``apply`` differ only by summation order.  ``N = nk·n_bands²`` columns in
    batches of ``batch``; the result is host (nk, nb, nb, nk, nb, nb)
    complex128 on every process.  For small windows only (N² ·16 B).
    """
    nk = int(np.prod(model["kgrid"]))
    nb = int(model["n_bands"])
    n = nk * nb * nb
    st = td_state_sharding(mesh_xy)
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    K = np.zeros((n, n), dtype=np.complex128)
    for j0 in range(0, n, batch):
        cols = np.arange(j0, min(j0 + batch, n))
        e = np.zeros((len(cols), n), dtype=np.complex128)
        e[np.arange(len(cols)), cols] = 1.0
        x = pad_td_state(e.reshape(len(cols), nk, nb, nb), model)
        xd = jax.make_array_from_callback(x.shape, st, lambda i: x[i])
        y = np.asarray(rep(apply(xd)).addressable_data(0))
        K[:, cols] = strip_td_state(y, model).reshape(len(cols), n).T
    return K.reshape(nk, nb, nb, nk, nb, nb)


# ===========================================================================
# Producers
# ===========================================================================

def sample_window_psi(wfn_file: str, centroid_indices, bands, mesh_xy: Mesh, *,
                      kgrid, n_rmu_pad: int):
    """ψ_{nk,s}(r_μ) of ``bands`` from a full-grid WFN.h5, in its own gauge.

    The plane-wave coefficients are the file's own (``k='ibz'``: no unfold,
    no symmetry rotation), sampled exactly as the GW run samples them
    (``wfn_transforms.gflat_to_rmu``, ortho norm, Bloch phase from the
    loader's k table), so the band gauge is the file's — the gauge of every
    dipole and energy a TDGW run reads from that file.

    Parameters
    ----------
    wfn_file : str — a BerkeleyGW WFN.h5 holding every point of ``kgrid``.
    centroid_indices : (n_rmu, 3) int — FFT-grid indices, canonical order.
    bands : (b_lo, b_hi) — absolute 0-based, half open.
    kgrid : (3,) ints;  n_rmu_pad : int — μ carrier extent.

    Returns
    -------
    psi : jax.Array
        (nk, b_hi-b_lo, ns, n_rmu_pad) c128 at ``P(None,None,None,'x')``,
        rows in flat C order of ``kgrid``.
    k_frac : (nk, 3) — the file's k of each row.
    rows : (nk_file,) — flat row of each file k (:func:`flat_k_rows`).
    """
    from wfn_loader import WfnLoader
    from common.wfn_layout import band_sphere_spec
    from common.wfn_transforms import gflat_to_rmu
    kgrid = tuple(int(n) for n in kgrid)
    wfn = WfnLoader(wfn_file, mesh=mesh_xy)
    try:
        if int(wfn.nkpts) != int(np.prod(kgrid)):
            _refuse("td_wfn_full_grid", f"{wfn_file}: {int(wfn.nkpts)} k points",
                    f"the full {kgrid} grid ({int(np.prod(kgrid))} k)",
                    "give TDGW's own full-grid (nosym) WFN.h5")
        k_file = np.asarray(wfn.kvecs(k="ibz"))
        rows = flat_k_rows(k_file, kgrid)
        b_lo, b_hi = int(bands[0]), int(bands[1])
        psi_G = wfn.load(bands=(b_lo, b_hi), k="ibz", sharding=band_sphere_spec())
        g_index = wfn.box_index_dev(k="ibz", mesh=mesh_xy)
        r_mu = np.asarray(centroid_indices, dtype=np.int32)
        psi_b = gflat_to_rmu(psi_G, g_index, r_mu, mesh=mesh_xy,
                             fft_grid=tuple(int(s) for s in wfn.fft_grid),
                             kvecs_frac=jnp.asarray(k_file), norm="ortho")
        nb, n_rmu = b_hi - b_lo, int(r_mu.shape[0])
        order = np.argsort(rows)                 # file row of each flat grid row
        out = NamedSharding(mesh_xy, P(None, None, None, "x"))

        @functools.partial(jax.jit, out_shardings=out)
        def _finish(a, order_):
            a = jnp.take(a[:, :nb], order_, axis=0)
            return jnp.pad(a, ((0, 0), (0, 0), (0, 0), (0, n_rmu_pad - n_rmu)))
        psi = _finish(psi_b, jnp.asarray(order, dtype=jnp.int32))
        return psi, k_file[order], rows
    finally:
        wfn.close()


def hartree_operator_from_zeta(zeta_file: str, mesh_xy: Mesh, n_rmu_pad: int, *,
                               coulomb: dict):
    """ζ̃_0(μ, G) and v(G) for the Hartree term, G = 0 excluded.

    ζ̃ is the q = 0 row of ``zeta_q.h5`` (``ZetaLoader.read_zeta_G_slab``);
    v is the V_q producer's own kernel, ``gw.compute_vcoul.compute_v_q_per_G``
    with the restart's Coulomb policy (``coulomb``: sys_dim, bvec,
    cell_volume, bdot, bare_coulomb_cutoff_ry), at the q = 0 sphere of the
    file, with the G = 0 slot and the sphere's pad slots set to zero.

    Returns
    -------
    zeta_G : (n_rmu_pad, nG_pad) c128 at ``P(None,('x','y'))``.
    v_G : (nG_pad,) f64 at ``P(('x','y'))``, Ry.
    receipt : dict — q = 0 row, sphere size, the zeroed slot, v range.
    """
    from file_io.restart_bundle import open_zeta
    from gw.compute_vcoul import compute_v_q_per_G
    q0 = 0                                     # Γ: proven by the caller (``_gamma_row``)
    with open_zeta(zeta_file, mesh=mesh_xy) as zl:
        n_q_disk = int(zl.n_q_on_disk)
        comps = np.asarray(zl.gvec_components, dtype=np.int32)          # (n_q, 3, ngkmax)
        ngk = np.asarray(zl.ngk_per_q, dtype=np.int64)
        g0 = comps[q0, :, :int(ngk[q0])]
        zero = np.flatnonzero(np.all(g0 == 0, axis=0))
        if zero.size != 1:
            _refuse("td_zeta_q0_row", f"{zero.size} G=0 slots in on-disk q row {q0}",
                    "exactly one", "check the ζ file's G list")
        zeta = zl.read_zeta_G_slab(q_offset=q0, q_count=1, mu_offset=0,
                                   mu_count=int(n_rmu_pad), mesh=mesh_xy)
        ngkmax = int(zeta.shape[-1])
    v = np.real(np.asarray(compute_v_q_per_G(
        np.zeros((1, 3)), comps[q0:q0 + 1], bvec=coulomb["bvec"],
        cell_volume=coulomb["cell_volume"], sys_dim=int(coulomb["sys_dim"]),
        vcoul_cutoff_ry=coulomb.get("bare_coulomb_cutoff_ry"),
        bdot=coulomb.get("bdot"), v_head_fn=None))[0])
    v_raw_g0 = float(v[int(zero[0])])
    v[int(zero[0])] = 0.0
    v[int(ngk[q0]):] = 0.0
    p = int(np.prod(mesh_xy.devices.shape))
    nG_pad = round_up(ngkmax, p)
    v_pad = np.zeros(nG_pad)
    v_pad[:ngkmax] = v

    @functools.partial(jax.jit, out_shardings=NamedSharding(mesh_xy, ZETA_G_SPEC))
    def _zeta(z):
        return jnp.pad(z[0], ((0, 0), (0, nG_pad - ngkmax)))
    from common.collectives import device_put_process_local
    v_G = device_put_process_local(v_pad, NamedSharding(mesh_xy, V_G_SPEC))
    return _zeta(zeta), v_G, dict(
        zeta_file=os.path.abspath(zeta_file), q_row=q0, n_G=int(ngk[q0]),
        g0_slot=int(zero[0]), v_g0_raw=v_raw_g0, v_max=float(v.max()),
        sys_dim=int(coulomb["sys_dim"]), q_rows_on_disk=n_q_disk)


def load_td_model(input_file: str, mesh_xy: Mesh, *, bands, wfn_file: str | None = None,
                  zeta_file: str | None = None, n_occ: int | None = None,
                  degeneracy_mode: str | None = None, eqp_file: str | None = None,
                  velocity_file: str | None = None, velocity_full_window: bool = False,
                  keep_window: bool = False) -> dict:
    """The model bundle from a GW restart: W from the BSE loader, ψ from TDGW's WFN.

    W(q) (with its q = 0 head) comes from the BSE restart loader
    (``bse_loading.load_bse_data_from_restart_sharded``, so a deck with
    ``bse_k_grid`` delivers the densified W and ψ), ψ from ``wfn_file`` when
    given (:func:`sample_window_psi`, TDGW's gauge) and from the restart
    otherwise, and the Hartree operator from the restart's ``zeta_q.h5``.

    Parameters
    ----------
    input_file : str — the GW deck of the restart.
    bands : (b_lo, b_hi) — the TD window, absolute 0-based, half open; it
        must straddle n_occ and close degenerate multiplets at both edges.
    wfn_file : str, optional — TDGW's full-grid WFN.h5.
    zeta_file : str, optional — default ``<restart dir>/zeta_q.h5``.
    eqp_file : str, optional — BerkeleyGW ``eqp.dat``, the BSE run's ``--eqp``.
        On a ``bse_k_grid`` deck it enters the loader's htransform, so the
        fine ψ are the quasiparticle f(H) states (the states of a
        ``bse_jax --eqp`` run on the same deck) and the fine ε are QP.  On a
        coarse deck ψ stay the restart's (or ``wfn_file``'s) and the file
        only sets ``window``'s energies (``keep_window``).
    velocity_file : str, optional — the coarse WFN's ``dipole.h5``
        (``bse_jax --dipole``).  The loader reads it on the densified route
        only; refuses on a coarse deck and with a loader that has no
        ``velocity_file``.
    velocity_full_window : bool — ask the loader for the velocity over every
        window pair (vv', cc' and cv); refuses when the loader has no such
        option.  Needs ``velocity_file``.
    keep_window : bool — keep the loader's window as ``model["window"]``:
        ``eps_v``/``eps_c`` (Ry, the loader's padded (nk, n_pad) layout,
        window bands ascending, pad slots the loader's sentinel), ``n_occ``,
        ``n_val``, ``n_cond``, ``dipoles`` (the loader's, or None),
        ``densified``, ``eqp_file`` and ``energy_source``.

    All four are off by default, and then the loader call and the bundle are
    those of the previous signature.

    Returns
    -------
    model : dict — :func:`assemble_td_model`'s bundle; ``receipts`` carries
        provenance and the ψ gauge comparison against the restart; the
        cross-check operands are ``V_q0_stored`` (the restart's q = 0
        exchange tile), ``V_q0_bse`` (the BSE loader's, stored + the vhead
        head) and ``g0_X`` (the head vector), all (nmu_pad, …) on the mesh;
        ``window`` with ``keep_window``.
    """
    from bse.bse_feast import ensure_W_R
    from bse.bse_loading import load_bse_data_from_restart_sharded
    from bse.bse_window import resolve_n_occ
    from common.band_degeneracy import DEFAULT_MODE
    from file_io.restart_bundle import (_find_restart_file, read_coulomb_policy_from_h5,
                                        read_interaction, read_metadata)
    from gw.gw_config import LorraxConfig
    from file_io.centroids import load_centroid_basis
    from file_io.wfn_basis import centroid_table_md5
    from wfn_loader import WfnLoader

    restart_file = _find_restart_file(input_file)
    header = read_metadata(restart_file)
    bw = header.get("band_window")
    if bw is not None and int(np.asarray(bw).reshape(-1)[0]) != 0:
        _refuse("td_band_origin", f"band_window[0] = {int(np.asarray(bw).reshape(-1)[0])}",
                "0 (absolute band indexing)", "use a restart whose window starts at band 0")
    n_occ = resolve_n_occ(header["energies"], n_occ=n_occ, input_file=input_file)
    b_lo, b_hi = int(bands[0]), int(bands[1])
    if not b_lo < n_occ < b_hi:
        _refuse("td_window_gap", f"bands [{b_lo}, {b_hi}) with n_occ={n_occ}",
                "b_lo < n_occ < b_hi", "give a window that straddles the gap")
    n_val, n_cond = n_occ - b_lo, b_hi - n_occ
    densifies = None
    if velocity_file is not None:
        densifies = _deck_densifies(input_file, header)
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val, n_cond, mesh_xy=mesh_xy, input_file=input_file,
        n_occ=n_occ, degeneracy_mode=degeneracy_mode or DEFAULT_MODE,
        **_loader_options(load_bse_data_from_restart_sharded, eqp_file=eqp_file,
                          velocity_file=velocity_file,
                          velocity_full_window=velocity_full_window, densifies=densifies))
    if (int(data["n_val"]), int(data["n_cond"])) != (n_val, n_cond):
        _refuse("td_window_resolved", f"loader window {data['n_val']}v x {data['n_cond']}c",
                f"{n_val}v x {n_cond}c", "close the multiplets at both window edges")
    ensure_W_R(data, True, mesh_xy)
    kgrid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    psi_restart = jax.jit(lambda v, c: jnp.concatenate([v[:, :n_val], c[:, :n_cond]], axis=1),
                          out_shardings=NamedSharding(mesh_xy, P(None, None, None, "x")))(
        data["psi_v_X"], data["psi_c_X"])
    cfg = LorraxConfig.from_input_file(input_file)
    deck_wfn = WfnLoader(cfg.paths.wfn_file)
    try:
        receipts = dict(restart_file=os.path.abspath(restart_file), n_occ=int(n_occ),
                        bands=(b_lo, b_hi), densified=bool(data.get("densified", False)))
        if wfn_file is not None:
            if data.get("densified", False):
                _refuse("td_wfn_densified", "wfn_file with a bse_k_grid deck",
                        "ψ from one source", "drop wfn_file (the densified ψ is the htransform's)")
            centroids = load_centroid_basis(cfg.paths.centroids_file, deck_wfn.fft_grid,
                                            sym=deck_wfn.symmetry())
            idx = np.asarray(centroids.centroid_indices)
            digest = header["centroid_hashes"].get("charge")
            if digest is not None and str(np.asarray(digest).astype(str)) != centroid_table_md5(idx):
                _refuse("td_centroids_md5", "deck centroids differ from the restart's",
                        "the restart's centroid table", "point the deck at the restart's centroids")
            tdgw_wfn = WfnLoader(wfn_file)
            same_grid = tuple(tdgw_wfn.fft_grid) == tuple(deck_wfn.fft_grid)
            same_cell = np.allclose(np.asarray(tdgw_wfn.bvec), np.asarray(deck_wfn.bvec), atol=1e-8)
            tdgw_wfn.close()
            if not (same_grid and same_cell):
                _refuse("td_wfn_geometry", f"fft_grid {tuple(tdgw_wfn.fft_grid)}, same_cell={same_cell}",
                        f"the restart WFN's fft_grid {tuple(deck_wfn.fft_grid)} and cell",
                        "generate TDGW's WFN from the GW run's SCF and cutoff")
            psi, k_frac, _ = sample_window_psi(wfn_file, idx, (b_lo, b_hi), mesh_xy,
                                               kgrid=kgrid, n_rmu_pad=n_rmu_pad)
            receipts["psi_source"] = os.path.abspath(wfn_file)
            receipts["psi_vs_restart"] = _gauge_receipt(psi, psi_restart, mesh_xy)
        else:
            psi = psi_restart
            # The loader's flat k axis is reshaped to (nkx, nky, nkz) by the W
            # convolution, so its rows are the C-order grid, coarse or densified.
            k_frac = _c_order_k(kgrid)
            if not data.get("densified", False):
                flat_k_rows(np.asarray(deck_wfn.kvecs(k="full_bz")), kgrid)   # full, unshifted grid
            receipts["psi_source"] = "restart"
        coulomb = _coulomb_from_policy(read_coulomb_policy_from_h5(restart_file), deck_wfn,
                                       restart_file)
    finally:
        deck_wfn.close()
    zeta_file = zeta_file or os.path.join(os.path.dirname(restart_file), "zeta_q.h5")
    _gamma_row(restart_file, kgrid)
    zeta_G, v_G, zr = hartree_operator_from_zeta(zeta_file, mesh_xy, n_rmu_pad, coulomb=coulomb)
    receipts["hartree"] = zr
    receipts["coulomb"] = {k: v for k, v in coulomb.items() if np.isscalar(v) or v is None}
    model = assemble_td_model(psi, data["W_R"], zeta_G, v_G, mesh_xy=mesh_xy, kgrid=kgrid,
                              k_frac=k_frac, band_lo=b_lo, n_bands=b_hi - b_lo,
                              n_rmu=n_rmu, receipts=receipts)
    model["V_q0_stored"] = read_interaction(restart_file, "bare", mesh_xy)[0]
    model["V_q0_bse"] = data["V_q0"]
    model["g0_X"] = data["g0_X"]
    for name, path in (("eqp_file", eqp_file), ("velocity_file", velocity_file)):
        if path is not None:
            receipts[name] = os.path.abspath(path)
    if velocity_full_window:
        receipts["velocity_full_window"] = True
    if keep_window:
        model["window"] = _window_record(
            data, restart_file=restart_file, input_file=input_file, n_occ=n_occ,
            eqp_file=eqp_file, mesh_xy=mesh_xy,
            degeneracy_mode=degeneracy_mode or DEFAULT_MODE)
    return model


def _deck_densifies(input_file: str, header: dict) -> bool:
    """True when the deck's ``bse_k_grid`` differs from the restart's grid (the loader's own test)."""
    from bse.bse_densify import _resolve_bse_k_grid
    fine = _resolve_bse_k_grid(None, input_file)
    coarse = tuple(int(n) for n in np.asarray(header["grid"]).reshape(-1)[:3])
    return fine is not None and tuple(int(n) for n in fine) != coarse


def _loader_options(loader, *, eqp_file=None, velocity_file=None,
                    velocity_full_window: bool = False, densifies=None) -> dict:
    """The optional keywords :func:`load_td_model` forwards to the BSE loader.

    Empty at the defaults, so the default call is the loader's own.  The
    velocity keywords are forwarded only to a loader that declares them, and
    only on the densified route (``densifies``), the one route on which the
    loader reads the velocity; elsewhere the request refuses by name instead
    of being dropped.
    """
    import inspect
    params = inspect.signature(loader).parameters
    out = {}
    if eqp_file is not None:
        out["eqp_file"] = os.path.abspath(eqp_file)
    if velocity_full_window and velocity_file is None:
        _refuse("td_velocity_full_window", "velocity_full_window=True, velocity_file=None",
                "a velocity_file", "pass the coarse WFN's dipole.h5 (bse_jax --dipole)")
    if velocity_file is not None:
        if "velocity_file" not in params:
            _refuse("td_velocity_loader", f"{loader.__module__}.{loader.__name__} has no "
                    "velocity_file", "a loader that carries the velocity through the "
                    "htransform (BSEDIP feat/bse-kgrid-dipoles-2026-10-16)",
                    "run from a checkout that merges the BSEDIP loader")
        if not densifies:
            _refuse("td_velocity_route", "velocity_file on a deck without a densifying "
                    "bse_k_grid", "a bse_k_grid deck (the loader reads the velocity there "
                    "only)", "slice the coarse dipole.h5 with absorption_common.window_dipoles")
        out["velocity_file"] = os.path.abspath(velocity_file)
    if velocity_full_window:
        if "velocity_full_window" not in params:
            _refuse("td_velocity_full_window_loader", f"{loader.__module__}."
                    f"{loader.__name__} has no velocity_full_window",
                    "a loader that returns the velocity over every window pair "
                    "(vv', cc' and cv)", "add velocity_full_window to the BSE loader")
        out["velocity_full_window"] = True
    return out


def _window_record(data: dict, *, restart_file: str, input_file: str, n_occ: int,
                   eqp_file, mesh_xy: Mesh, degeneracy_mode: str) -> dict:
    """``model["window"]``: the loader's window energies, occupation and dipoles.

    On the densified route the loader's ε already carry ``eqp_file`` (its
    htransform).  On a coarse deck with ``eqp_file`` they are re-sliced as
    ``bse_jax --eqp`` does (``bse_window.apply_eqp_and_reslice_bands``); a
    QP gap that moves ``n_occ`` refuses, since ψ hold the DFT window.
    """
    densified = bool(data.get("densified", False))
    n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
    eps_v, eps_c = data["eps_v"], data["eps_c"]
    if densified:
        source = "htransform (eqp)" if eqp_file is not None else "htransform (dft)"
    elif eqp_file is not None:
        from bse.bse_window import apply_eqp_and_reslice_bands
        grid_x, grid_y = mesh_xy.devices.shape
        eps_v, eps_c, n_occ_eff = apply_eqp_and_reslice_bands(
            restart_file, eqp_file, input_file, n_val, n_cond, n_occ, grid_x, grid_y,
            degeneracy_mode=degeneracy_mode)
        if int(n_occ_eff) != int(n_occ):
            _refuse("td_eqp_n_occ", f"n_occ {n_occ_eff} on the QP spectrum of {eqp_file}",
                    f"the DFT window's n_occ {n_occ}",
                    "pass n_occ, or a window whose QP and DFT occupations agree")
        source = "eqp re-slice (bse_window.apply_eqp_and_reslice_bands)"
    else:
        source = "restart"
    return dict(eps_v=eps_v, eps_c=eps_c, n_occ=int(n_occ), n_val=n_val, n_cond=n_cond,
                dipoles=data.get("dipoles"), densified=densified,
                eqp_file=None if eqp_file is None else os.path.abspath(eqp_file),
                energy_source=source)


def _c_order_k(kgrid) -> np.ndarray:
    """(nk, 3) fractional k of the flat C-order grid, row i = (ix, iy, iz)/N."""
    return np.stack(np.meshgrid(*[np.arange(int(n)) / int(n) for n in kgrid], indexing="ij"),
                    -1).reshape(-1, 3)


def _coulomb_from_policy(policy, wfn, restart_file) -> dict:
    """The V_q producer's Coulomb geometry from the restart's ``coulomb_policy`` stamp.

    The Hartree term must use the kernel the stored V and W were built with,
    so an unstamped restart or one whose V carries BerkeleyGW's vcoul overlay
    (which ``compute_v_q_per_G`` does not reproduce) refuses.
    """
    if not policy or not str(policy.get("sys_dim", "")).strip():
        _refuse("td_coulomb_policy", f"{restart_file}: no coulomb_policy sys_dim stamp",
                "a stamped restart", "regenerate the restart with a current GW run")
    if str(policy.get("use_bgw_vcoul", "")).strip().lower() in ("true", "1", "yes"):
        _refuse("td_coulomb_bgw_vcoul", "use_bgw_vcoul = true", "the native v(q+G)",
                "run the GW with the native Coulomb kernel")
    from vcoul import CoulombGeometry
    geo = CoulombGeometry.from_wfn(wfn)          # the V_q writer's geometry (gw_init)
    cut = str(policy.get("bare_coulomb_cutoff", "")).strip()
    return dict(sys_dim=int(policy["sys_dim"]), bvec=np.asarray(geo.bvec, dtype=np.float64),
                cell_volume=float(geo.cell_volume), bdot=None,
                bare_coulomb_cutoff_ry=float(cut) if cut else None)


def _gamma_row(restart_file: str, kgrid) -> int:
    """On-disk q row of Γ in the restart's q layout; refuses when it is not row 0.

    An IBZ-stored restart carries its q list (``V_qmunu__qirr/q_irr_frac``);
    full storage is the flat C-order grid, whose row 0 is Γ.  ``zeta_q.h5``
    shares that row order.  (A G-sphere at q ≠ 0 also holds Miller index 0,
    so the G list alone cannot identify Γ.)
    """
    import h5py
    with h5py.File(restart_file, "r") as f:
        if "V_qmunu__qirr" in f and "q_irr_frac" in f["V_qmunu__qirr"]:
            q = np.asarray(f["V_qmunu__qirr/q_irr_frac"][()], dtype=np.float64).reshape(-1, 3)
        else:
            q = np.zeros((1, 3))
    if np.max(np.abs(q[0] - np.rint(q[0]))) > 1e-12:
        _refuse("td_zeta_gamma_row", f"q row 0 = {q[0].tolist()}", "Γ at row 0",
                "read the Γ row of this restart's q list")
    return 0


def _gauge_receipt(psi_a, psi_b, mesh_xy) -> dict:
    """max |ψ_a − ψ_b| and max |ψ_b| over the whole window, replicated scalars."""
    @functools.partial(jax.jit, out_shardings=NamedSharding(mesh_xy, P()))
    def _m(a, b):
        return jnp.max(jnp.abs(a - b)), jnp.max(jnp.abs(b)), jnp.max(
            jnp.abs(jnp.abs(a) - jnp.abs(b)))
    d, n, dmod = (float(np.asarray(x)) for x in _m(psi_a, psi_b))
    return dict(max_abs_diff=d, max_abs=n, max_modulus_diff=dmod)
