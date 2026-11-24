"""Diagnostic-branch observers for the ISDF conditioning study (lane isdf_cond, 2026-11-24).

Everything here is inert unless ``LXDIAG_DIR`` is set in the environment:

* ``save(name, array)``: gather a (possibly sharded, multi-process) array and
  write ``$LXDIAG_DIR/<name>.npy`` on process 0.  Used for small quantities:
  the charge Gram spectrum per q, and the q = 0 slices of V, W(0), W(i w_p).
* ``perturb_gram(C)``: when ``LXDIAG_CPERT`` = eps is set, return
  ``C + eps * diag(diag(C) * r)`` with a fixed r_i = cos(1.7 i + 0.3): a
  Hermitian relative perturbation of size eps of the charge Gram before its
  factor (the sensitivity test).  Pad slots have a zero diagonal and stay zero.
"""
from __future__ import annotations

import os

import numpy as np

_COUNT: dict = {}


def _p(msg: str) -> None:
    """Diagnostic line on stderr (production stdout is /dev/null)."""
    import sys
    print(f"[isdf_diag] {msg}", file=sys.stderr, flush=True)


def enabled() -> bool:
    return bool(os.environ.get("LXDIAG_DIR"))


def _gather(a):
    import jax
    if getattr(a, "is_fully_addressable", True):
        return np.asarray(jax.device_get(a))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(a, tiled=True))


def save(name: str, a) -> None:
    """Write ``a`` to ``$LXDIAG_DIR/<name>[_n].npy`` on process 0 (every process must call)."""
    d = os.environ.get("LXDIAG_DIR")
    if not d:
        return
    import jax
    arr = _gather(a)
    if jax.process_index() == 0:
        os.makedirs(d, exist_ok=True)
        n = _COUNT.get(name, 0)
        _COUNT[name] = n + 1
        np.save(os.path.join(d, f"{name}_{n}.npy" if n else f"{name}.npy"), arr)
        _p(f"saved {name} shape={arr.shape} dtype={arr.dtype}")


def gram_spectrum(C, name: str, n_logical: int) -> None:
    """Eigenvalues (ascending) of C[q] for q in {0, nq//2, nq-1}: ``$LXDIAG_DIR/<name>.npy`` as (3, n)."""
    if not enabled() or os.environ.get("LXDIAG_GRAM", "1") == "0":
        return
    import jax
    import jax.numpy as jnp
    nq = int(C.shape[0])
    qs = sorted({0, nq // 2, nq - 1})
    out = []
    _p(f"gram_spectrum enter: C {tuple(C.shape)} {C.dtype}")
    for q in qs:
        cq = _gather(C[q])
        # Every process runs the same single-device eigvalsh (no rank-conditional
        # device work: a rank-0-only compile deadlocked the run, 2026-11-24).
        dev = jax.local_devices()[0]
        lam = jax.device_get(jnp.linalg.eigvalsh(jax.device_put(cq, dev)))
        out.append(np.asarray(lam))
        _p(f"gram_spectrum q={q} done")
        del cq
    if jax.process_index() == 0:
        d = os.environ["LXDIAG_DIR"]
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"{name}.npz"), lam=np.array(out), q=np.array(qs), n_logical=n_logical)
        for q, lam in zip(qs, out):
            mx = lam.max()
            _p(f"{name} q={q} n={lam.size} n_logical={n_logical} lam_max={mx:.4e} "
                  f"lam_min={lam.min():.3e} below 1e-8/1e-12/1e-14 of max: "
                  f"{int((lam < 1e-8 * mx).sum())}/{int((lam < 1e-12 * mx).sum())}/{int((lam < 1e-14 * mx).sum())}",
                  )


def perturb_gram(C):
    eps = float(os.environ.get("LXDIAG_CPERT", "0") or 0.0)
    if eps == 0.0:
        return C
    import jax
    import jax.numpy as jnp

    def f(C):
        n = C.shape[-1]
        d = jnp.real(jnp.diagonal(C, axis1=-2, axis2=-1))
        r = jnp.cos(1.7 * jnp.arange(n, dtype=jnp.float64) + 0.3)
        return C + jnp.eye(n, dtype=C.dtype)[None] * (eps * d * r)[:, None, :].astype(C.dtype)

    out = jax.jit(f, out_shardings=C.sharding)(C)
    if jax.process_index() == 0:
        _p(f"charge Gram perturbed: C + {eps:.1e}*diag(diag(C)*r)")
    return out
