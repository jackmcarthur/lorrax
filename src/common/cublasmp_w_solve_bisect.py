"""Bisect correctness in the fused cuBLASMp + cuSOLVERMp W-solve.

For each stop_after_step in {1..10}, call the FFI and compare the
returned intermediate buffer against a numpy reference.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST): return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        init_kwargs = {"local_device_ids": [0]} if cvd and "," not in cvd else {}
        try: jax.distributed.initialize(**init_kwargs)
        except Exception: pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cublasmp import batched_fused_w_solve


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def main():
    Px, Py = 2, 2
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != {Px*Py}"); return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))

    dtype = jnp.complex128
    nq, n = 1, 8
    pref = complex(1.0, 0.0)

    @jax.jit
    def make():
        kv, ki, kc1, kc2 = jax.random.split(jax.random.key(7), 4)
        Vr = jax.random.normal(kv, (nq, n, n), dtype=jnp.float64)
        Vi = jax.random.normal(ki, (nq, n, n), dtype=jnp.float64)
        Vbase = (Vr + 1j * Vi).astype(dtype)
        V = Vbase @ jnp.conj(jnp.swapaxes(Vbase, -1, -2)) + (n * 2.0) * jnp.eye(n, dtype=dtype)[None]
        V = 0.5 * (V + jnp.conj(jnp.swapaxes(V, -1, -2)))
        Cr = jax.random.normal(kc1, (nq, n, n), dtype=jnp.float64)
        Ci = jax.random.normal(kc2, (nq, n, n), dtype=jnp.float64)
        Cbase = (Cr + 1j * Ci).astype(dtype)
        Cpd = Cbase @ jnp.conj(jnp.swapaxes(Cbase, -1, -2))
        Cpd = 0.5 * (Cpd + jnp.conj(jnp.swapaxes(Cpd, -1, -2)))
        chi = -0.01 * Cpd
        return (jax.lax.with_sharding_constraint(V, sh),
                jax.lax.with_sharding_constraint(chi, sh))

    V0, chi0 = make()
    jax.block_until_ready(V0); jax.block_until_ready(chi0)

    V_np0   = np.asarray(multihost_utils.process_allgather(V0))[0]   # (n,n)
    chi_np0 = np.asarray(multihost_utils.process_allgather(chi0))[0]

    # Compute all numpy references.
    X_ref = np.linalg.cholesky(V_np0)                              # lower tri
    T_ref = pref * (X_ref.conj().T @ chi_np0)                      # step 3
    T2_ref = T_ref @ X_ref                                         # step 4
    H_ref = np.eye(n, dtype=np.complex128) - T2_ref                # step 5
    LH_ref = np.linalg.cholesky(H_ref)                             # step 6
    # step 7: copy X -> Z
    Z7_ref = X_ref.copy()
    # step 8: Z = X @ inv(LH)
    Z8_ref = np.linalg.solve(LH_ref.T, X_ref.T).T
    # step 9: Z = Z @ inv(LH^H)  =>  Z9 = Z8 @ inv(LH^H)
    # equivalent to H^{-1} applied on right of X from H=LH LH^H
    Z9_ref = np.linalg.solve(LH_ref.conj(), Z8_ref.T).T
    # alternate form: Z9 = X @ H^{-1}
    H_inv = np.linalg.inv(H_ref)
    Z9_ref_alt = X_ref @ H_inv
    # step 10: W = Z @ X^H
    W_ref = Z9_ref_alt @ X_ref.conj().T

    refs = {
        1:  ("X (lower Chol of V, upper junk ok)", X_ref, "lower"),
        2:  ("X (upper zeroed)",                    X_ref, "full"),
        3:  ("T_tmp = pref X^H chi",                T_ref, "full"),
        4:  ("T2 = T_tmp X",                        T2_ref, "full"),
        5:  ("H = I - T2",                          H_ref, "full"),
        6:  ("L_H (lower Chol of H)",               LH_ref, "lower"),
        7:  ("Z = X (copy)",                        Z7_ref, "full"),
        8:  ("Z = X L_H^{-1}",                      Z8_ref, "full"),
        9:  ("Z = X H^{-1}",                        Z9_ref_alt, "full"),
        10: ("W = X H^{-1} X^H",                    W_ref, "full"),
    }

    _log(f"=== bisect: nq={nq} n={n} mesh={Px}x{Py} pref={pref} ===")
    for step in range(1, 11):
        # remake V/chi every iteration because V is donated.
        V, chi = make()
        jax.block_until_ready(V); jax.block_until_ready(chi)
        out = batched_fused_w_solve(V, chi, pref, mesh=mesh,
                                    stop_after_step=step)
        jax.block_until_ready(out)
        out_np = np.asarray(multihost_utils.process_allgather(out))[0]
        label, ref, mode = refs[step]
        if mode == "lower":
            # Compare only lower triangle (incl diag).
            mask = np.tril(np.ones((n, n), dtype=bool))
            diff = (out_np - ref) * mask
            denom = max(np.max(np.abs(ref * mask)), 1.0)
        else:
            diff = out_np - ref
            denom = max(np.max(np.abs(ref)), 1.0)
        rel = np.max(np.abs(diff)) / denom
        abs_out = np.max(np.abs(out_np))
        abs_ref = np.max(np.abs(ref))
        _log(f"  step {step:2d}  rel_err = {rel:.3e}   |out|max={abs_out:.3e} |ref|max={abs_ref:.3e}   [{label}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
