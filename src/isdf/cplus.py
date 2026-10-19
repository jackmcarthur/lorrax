"""Charge ISDF conditioning, the one seam: ζ = C⁻¹Z by a certified Cholesky.

``factor`` Cholesky-factors the Hermitian-PD charge C_q stack ``(nq, n, n)`` on the face
(``distrib_la.cholesky_factor``: cuSOLVERMp, SLATE or native2d) and ``solve`` is ζ = C⁻¹Z there;
``dense`` is the same arithmetic on one whole local tile (the BSE refit), its figures read by
``report`` after the program. The previous seam truncated the spectrum at λ > rcond·λ_max
(an eigh pseudo-inverse) and so silently absorbed an over-complete centroid set (Cr:Al2O3
Γ: pivoted-Cholesky rank 18,779 of 29,998 centroids). This one REFUSES BY NAME wherever that
cut would have bound: κ(C_q) > 1/rcond (the certificate: λ_max by power iteration, λ_min by
inverse iteration through the factor, on the logical rows) or a factor that breaks down,
naming the numerical rank against N_μ and the remedy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: Probe columns and iterations of the κ certificate (λ_max by power iteration on C,
#: λ_min by inverse iteration through the factor).
PROBES, ITERS = 8, 24
REMEDY = ("set LORRAX_CENTROID_SELECT=strict, or use fewer centroids, or more bands "
          "(the deck must represent the pair densities of the bands it fits)")


def _probes(nq, n, n_log, dtype):
    x = jax.random.normal(jax.random.key(20261019), (nq, n, PROBES), jnp.float64).astype(dtype)
    return jnp.where((jnp.arange(n) < n_log)[None, :, None], x, 0)   # the logical rows only


def _orth(x):
    return jnp.linalg.qr(x)[0]


def _extreme(apply, x, iters):
    """Largest |eigenvalue| of ``apply`` (Hermitian PSD on the probes' span) per q: block power iteration."""
    for _ in range(iters):
        x = _orth(apply(x))
    y = apply(x + 0)                 # cuSOLVERMp's potrs consumes its right-hand side
    return jnp.max(jnp.real(jnp.einsum("qnk,qnk->qk", jnp.conj(x), y)), axis=-1)


def _refusal(where, n_log, rcond, kappa=None, rank=None, detail=""):
    figure = ("" if kappa is None else f"kappa(C_q) up to {float(np.max(kappa)):.3e} > 1/rcond = {1 / rcond:.3e}; "
              if np.all(np.isfinite(kappa)) else "the factor is not finite (C_q is not positive definite); ")
    ranks = "" if rank is None else f"numerical rank/q {np.asarray(rank).tolist()} (eigenvalues above rcond·λ_max) of "
    return (f"GATE zeta_charge_conditioning: {where}: the charge Gram C_q is not certified: {figure}"
            f"{ranks}N_mu = {n_log}{(' (' + detail + ')') if detail else ''}. The centroid set "
            f"over-completes the pair-density rank; {REMEDY}.")


def _rank(C, mesh, n_log, rcond):
    """Refusal path only: per q, the eigenvalues of the logical block above rcond·λ_max (the
    old cut's kept rank); the pads are the tail rows (core._identity_pad_block_diagonal)."""
    from distrib_la import plan
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    eigh, rank = plan("eigh", mesh, n=n_log), []
    for q in range(C.shape[0]):
        a = jax.device_put(C[q, :n_log, :n_log], NamedSharding(mesh, P("x", "y")))
        w = np.asarray(multihost_utils.process_allgather(eigh(a)[0], tiled=True))
        rank.append(int(np.sum(w > rcond * w.max())))
    return rank


def factor(C, mesh, *, rcond: float, n_log: int, where="zeta charge"):
    """The certified face Cholesky of ``C``; refuses by name (``_refusal``)."""
    from distrib_la import cholesky_factor, cholesky_solve
    nq, n, _ = C.shape
    x = _probes(nq, n, n_log, C.dtype)
    lam_max = np.asarray(jax.jit(lambda c, v: _extreme(lambda y: c @ y, v, ITERS))(C, x))
    try:     # the library's info (not positive definite) surfaces where the factor is first read
        F = cholesky_factor(C, mesh)
        lam_inv = np.asarray(_extreme(lambda y: cholesky_solve(F, y), x, ITERS))
    except (RuntimeError, ValueError) as error:
        raise ValueError(_refusal(where, n_log, rcond, rank=_rank(C, mesh, n_log, rcond),
                                  detail=f"the Cholesky broke down: {error}")) from None
    kappa = lam_max * lam_inv
    if not (np.all(np.isfinite(kappa)) and np.max(kappa) <= 1 / rcond):
        del F
        raise ValueError(_refusal(where, n_log, rcond, kappa, _rank(C, mesh, n_log, rcond)))
    if jax.process_index() == 0:
        print(f"[zeta charge cholesky] n_log={n_log} kappa/q={np.array2string(kappa, precision=3)} "
              f"(cap 1/rcond={1 / rcond:.1e})", flush=True)
    return F


def solve(F, Z):
    """ζ = C⁻¹Z for the factor of :func:`factor`, ``Z`` (nq, n, m) on the face."""
    from distrib_la import cholesky_solve
    return cholesky_solve(F, Z)


def dense(C, Z, *, rcond: float):
    """``(ζ, figures)`` on ONE whole local tile ``C`` (n, n): the same factor and certificate,
    traceable; :func:`report` refuses on the host."""
    n = C.shape[-1]
    L = jnp.linalg.cholesky(C)
    cho = lambda y: jax.scipy.linalg.cho_solve((L, True), y[0])[None]
    x = _probes(1, n, n, C.dtype)
    kappa = _extreme(lambda y: C @ y, x, ITERS) * _extreme(cho, x, ITERS)
    return cho(Z[None])[0], {"kappa": kappa[0]}


def report(figures, *, rcond: float, n_log: int) -> None:
    """The host half of :func:`dense`: refuse by name when the tile was not certified."""
    kappa = float(np.asarray(jax.device_get(figures["kappa"])))
    if not (np.isfinite(kappa) and kappa <= 1 / rcond):
        raise ValueError(_refusal("zeta charge (dense refit)", n_log, rcond, np.asarray([kappa])))
