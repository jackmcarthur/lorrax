"""tdagw/self_energy.py — the instantaneous-state self-energy update dSigma[rho].

THE CONTRACT.  The propagator treats the self-energy as one pure function of
the instantaneous state,

    dSigma(t) = apply(params, rho(t), t),

with ``rho`` the ``(ntr, nk, nb, nb)`` density matrix of
:mod:`tdagw.eom` (trial axis first, k on the whole mesh) and the result of the
same shape, layout and units (Hartree), Hermitian per k.  ``params`` is a
pytree of device arrays the propagator passes as jit operands (never
captured as constants).  The propagator never assumes more than this:

* it calls ``apply`` only inside its jitted step, as often as the
  integrator needs (counted and reported per run);
* ``linear = True`` declares dSigma affine in rho - rho^eq; no integrator
  relies on it today (it is reported, and a consumer that wants a Krylov
  exponential of the full Liouvillian may use it);
* a model is free to diagonalize H(t) or build G(tau) inside ``apply`` —
  the instantaneous GN-PPM self-energy planned next is one more model with
  the same signature.

Four models live here: :func:`isdf_model`, the ISDF TD-aGW interaction of
:mod:`tdagw.model` (Hartree + screened exchange on the centroid basis) behind
this contract; :func:`dense_kernel_model`, TDGW's own update
(``aGWDriver.get_dSigma``) from a dense kernel K_{k n m, k' n' m'},

    TDA : dSigma_{k,cv} = sum_{k'c'v'} K_{kcv,k'c'v'} rho_{k',c'v'},
          dSigma_{k,vc} = conj(dSigma_{k,cv})                          (1)
    full: dSigma_{k,nm} = sum_{k'n'm'} K_{knm,k'n'm'} (rho - rho^eq)_{k',n'm'}  (2)

(TDGW writes -i K G^< with G^< = i rho, which is (1));
:func:`lowrank_kernel_model`, Eq. (1) with K = U Lambda U^H of rank R
(cost 2 N R per apply; the stand-in for a separable kernel at sizes where a
dense K cannot exist); and :func:`zero_model`, the independent-particle run
(``calc_type = "IP"``).
"""
from __future__ import annotations

import functools
from typing import Optional, Callable, NamedTuple

import numpy as np
import jax.numpy as jnp

__all__ = ["SelfEnergyModel", "zero_model", "dense_kernel_model",
           "lowrank_kernel_model", "isdf_model", "dense_kernel_from_tdgw_layout"]


class SelfEnergyModel(NamedTuple):
    """A pluggable instantaneous-state self-energy update.

    apply : callable (params, rho, t) -> dSigma, see the module contract.
    params : pytree of device arrays, the model's operands.
    linear : dSigma affine in rho - rho^eq.
    name : what the run header prints.
    """
    apply: Callable
    params: object
    linear: bool
    name: str


def _zero_apply(params, rho, t):
    del params, t
    return jnp.zeros_like(rho)


def zero_model() -> SelfEnergyModel:
    """dSigma = 0: TDGW ``calc_type = "IP"``."""
    return SelfEnergyModel(apply=_zero_apply, params=(), linear=True,
                           name="independent particle (dSigma = 0)")


def _tda_apply(params, rho, t):
    """Eq. (1).  params = (K, (c0, c1), (v0, v1)) with K (nk*nc*nv)^2."""
    del t
    kmat, (c0, c1), (v0, v1) = params[0], params[1], params[2]
    ntr, nk = rho.shape[0], rho.shape[1]
    x = rho[:, :, c0:c1, v0:v1]
    nc, nv = x.shape[2], x.shape[3]
    d = jnp.einsum("ij,tj->ti", kmat, x.reshape(ntr, nk * nc * nv))
    d = d.reshape(ntr, nk, nc, nv)
    out = jnp.zeros_like(rho)
    out = out.at[:, :, c0:c1, v0:v1].set(d)
    return out.at[:, :, v0:v1, c0:c1].set(jnp.conj(jnp.swapaxes(d, -1, -2)))


def _full_apply(params, rho, t):
    """Eq. (2).  params = (K, rho_eq) with K (nk*nb*nb)^2."""
    del t
    kmat, rho_eq = params
    ntr = rho.shape[0]
    x = (rho - rho_eq[None]).reshape(ntr, -1)
    return jnp.einsum("ij,tj->ti", kmat, x).reshape(rho.shape)


def dense_kernel_model(kernel, *, tda: bool, cslice=None, vslice=None,
                       rho_eq=None) -> SelfEnergyModel:
    """TDGW's dense-kernel self-energy update, Eq. (1) or (2).

    Parameters
    ----------
    kernel : (nk*nc*nv, nk*nc*nv) [TDA] or (nk*nb*nb, nk*nb*nb) [full]
        complex device array, Ha, row index (k, c, v) / (k, n, m) row-major
        on the PADDED k axis (pad rows and columns zero), row-sharded over
        the mesh.  K acts on rho directly (TDGW's sign: K = -kernel.h5 / 2).
    cslice, vslice : (start, stop) positions of the conduction and valence
        kernel bands inside the propagated window (TDA only).
    rho_eq : (nk, nb, nb) equilibrium density (full kernel only).
    """
    if tda:
        if cslice is None or vslice is None:
            raise ValueError(
                "GATE td_dense_kernel_slices: got: a TDA kernel without "
                "cslice/vslice; want: the window positions of its c and v "
                "bands; fix: pass the slices TDGW's BSEKernel derives "
                "(bands_b.mapsto(c_b), mapsto(v_b)).")
        apply = functools.partial(_tda_apply_static,
                                  tuple(map(int, cslice)),
                                  tuple(map(int, vslice)))
        return SelfEnergyModel(apply=apply, params=(kernel,), linear=True,
                               name=f"dense TDA kernel {tuple(kernel.shape)}")
    if rho_eq is None:
        raise ValueError(
            "GATE td_dense_kernel_rho_eq: got: a full kernel without rho_eq; "
            "want: the equilibrium density it is applied about (Eq. 2); fix: "
            "pass ops.rho_eq.")
    return SelfEnergyModel(apply=_full_apply, params=(kernel, rho_eq),
                           linear=True,
                           name=f"dense full kernel {tuple(kernel.shape)}")


def _tda_apply_static(cslice, vslice, params, rho, t):
    """Eq. (1) with the band slices static (they set shapes)."""
    return _tda_apply((params[0], cslice, vslice), rho, t)


def dense_kernel_from_tdgw_layout(kernel_knn_knn, nk_pad: int) -> np.ndarray:
    """TDGW's kernel tensor (k, n, m, k', n', m') -> the padded square matrix.

    ``kernel_knn_knn`` is the host array TDGW's ``read_kernel`` builds (after
    its sign, Ry->Ha, valence reversal and transpose), shape
    (nk, n1, n2, nk, n1, n2).  Pad k rows and columns are zero.
    """
    k = np.asarray(kernel_knn_knn)
    nk, n1, n2 = k.shape[:3]
    out = np.zeros((nk_pad, n1, n2, nk_pad, n1, n2), dtype=np.complex128)
    out[:nk, :, :, :nk] = k
    return out.reshape(nk_pad * n1 * n2, nk_pad * n1 * n2)


def _lowrank_apply(cslice, vslice, params, rho, t):
    """Eq. (1) with K = U diag(lam) U^H: d = U lam (U^H x).

    params = (u (nk, nc, nv, R) complex, k-sharded; lam (R,) float, Ha).
    The projection U^H x sums over k: one all-reduce of (ntr, R).
    """
    del t
    u, lam = params
    (c0, c1), (v0, v1) = cslice, vslice
    x = rho[:, :, c0:c1, v0:v1]
    a = jnp.einsum("kcvr,tkcv->tr", jnp.conj(u), x)
    d = jnp.einsum("kcvr,tr->tkcv", u, a * lam.astype(a.dtype)[None])
    out = jnp.zeros_like(rho)
    out = out.at[:, :, c0:c1, v0:v1].set(d)
    return out.at[:, :, v0:v1, c0:c1].set(jnp.conj(jnp.swapaxes(d, -1, -2)))


def lowrank_kernel_model(u, lam, *, cslice, vslice) -> SelfEnergyModel:
    """Eq. (1) with a rank-R Hermitian TDA kernel K = U diag(lam) U^H.

    u : (nk, nc, nv, R) complex device array, k-sharded (pad k rows zero).
    lam : (R,) float device array, Ha.  cslice, vslice : window positions.
    """
    apply = functools.partial(_lowrank_apply, tuple(map(int, cslice)),
                              tuple(map(int, vslice)))
    return SelfEnergyModel(apply=apply, params=(u, lam), linear=True,
                           name=f"low-rank TDA kernel, rank {int(u.shape[-1])}")


# ---------------------------------------------------------------------------
# The ISDF model of tdagw.model behind the dSigma contract
# ---------------------------------------------------------------------------

_ISDF_OPERANDS = ("psi_X", "psi_Y", "M", "W_R", "zeta_G", "v_G")


def _isdf_apply(static, params, rho, t):
    """dSigma = (1/2) P^T Delta Sigma^ISDF[P (rho - rho^eq)] (Ry -> Ha).

    P permutes TDGW's k order into the model's flat C order
    (``tdagw.model.flat_k_rows``) and pads the band axes to the model's
    carrier; the model's operator runs on its own layout
    ``P(None, None, 'x', 'y')`` (k whole, bands on the mesh), so one
    all-to-all moves rho there and one moves dSigma back.
    """
    del t
    from jax.lax import with_sharding_constraint
    from tdagw.model import build_td_agw_apply, td_state_sharding
    mesh, kgrid, nspinor, nb_pad, rows, perm, nk, sex, hartree, nv = static
    ops, rho_eq = params
    d = rho - rho_eq[None]
    nb, nk_pad = d.shape[-1], d.shape[1]
    if nv is not None:                   # TDGW TDA: only delta rho_cv drives, only dSigma_cv kept
        d = jnp.zeros_like(d).at[:, :, nv:, :nv].set(d[:, :, nv:, :nv])
    d = jnp.pad(d, ((0, 0), (0, 0), (0, nb_pad - nb), (0, nb_pad - nb)))
    d = with_sharding_constraint(d, td_state_sharding(mesh))
    x = jnp.take(d, jnp.asarray(perm), axis=1)
    model = dict(zip(_ISDF_OPERANDS, ops), kgrid=kgrid, nspinor=nspinor)
    y = build_td_agw_apply(mesh, model, sex=sex, hartree=hartree)(x)
    y = jnp.take(y, jnp.asarray(rows), axis=1)[..., :nb, :nb]
    if nv is not None:
        y_cv = y[:, :, nv:, :nv]
        y = (jnp.zeros_like(y).at[:, :, nv:, :nv].set(y_cv)
             .at[:, :, :nv, nv:].set(jnp.conj(jnp.swapaxes(y_cv, -1, -2))))
    return 0.5 * jnp.pad(y, ((0, 0), (0, nk_pad - nk), (0, 0), (0, 0)))


def isdf_model(model: dict, mesh, *, k_frac, rho_eq, sex: bool = True,
               hartree: bool = True, tda_n_val: Optional[int] = None) -> SelfEnergyModel:
    """The ISDF TD-aGW self-energy update of :mod:`tdagw.model`, in Ha.

    Parameters
    ----------
    model : the bundle of ``tdagw.model.assemble_td_model`` /
        ``load_td_model`` (Ry, flat C-order k rows).
    k_frac : (nk, 3) the propagated k list (TDGW's order); each point must
        be a model row (``flat_k_rows``) and the model's own ``k_frac`` of
        that row must be the same point mod G.
    rho_eq : (nk_pad, nb, nb) equilibrium density on the propagator's mesh.
    sex, hartree : keep the screened-exchange and Hartree terms.
    tda_n_val : TDGW's TDA semantics with the window's first ``tda_n_val`` rows
        as valence: dSigma_cv = K_{cv,c'v'} delta rho_{c'v'}, dSigma_vc =
        dSigma_cv^H, every other block 0 (``tdgw_bridge.make_tdgw_dsigma``'s
        projection; it drops the resonant/antiresonant coupling).
    """
    from tdagw.model import flat_k_rows
    rows = np.asarray(flat_k_rows(k_frac, model["kgrid"]), np.int64)
    d = np.asarray(model["k_frac"])[rows] - np.asarray(k_frac)
    if np.max(np.abs(d - np.rint(d)), initial=0.0) > 1e-8:
        raise ValueError(
            "GATE td_isdf_kpoints: got: model rows whose sampled k differs from "
            "the propagated k list; want: the same points (mod G); fix: build "
            "the model from the TDGW run's own WFN.h5")
    nb = int(rho_eq.shape[-1])
    if int(model["n_bands"]) != nb:
        raise ValueError(
            f"GATE td_isdf_window: got: model window {model['n_bands']} bands; "
            f"want: the propagated {nb}; fix: load the model with the deck's "
            f"[system] ibndmin..ibndmax")
    static = (mesh, tuple(int(n) for n in model["kgrid"]), int(model["nspinor"]),
              int(model["nb_pad"]), tuple(rows.tolist()),
              tuple(np.argsort(rows).tolist()), int(rows.size), bool(sex),
              bool(hartree), None if tda_n_val is None else int(tda_n_val))
    apply = functools.partial(_isdf_apply, static)
    params = (tuple(model[k] for k in _ISDF_OPERANDS), rho_eq)
    terms = "+".join(n for n, on in (("Hartree", hartree), ("SEX", sex)) if on)
    tda = "" if tda_n_val is None else f", TDA ({int(tda_n_val)} valence rows)"
    return SelfEnergyModel(apply=apply, params=params, linear=True,
                           name=f"ISDF TD-aGW ({terms}){tda}, N_mu = {model['n_rmu']}, "
                                f"k grid {tuple(model['kgrid'])}")
