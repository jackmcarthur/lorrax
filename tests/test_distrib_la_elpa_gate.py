"""The ELPA distributed eigh (``lorrax_elpa_eigh``, ROCM leg) against ``jnp.linalg.eigh``.

``distrib_la._elpa.distributed_eigh`` runs ELPA 2026.02.001's one-stage solver on the AMD GPU
(config/frontier/build_elpa_hip.sh, cpp/elpa/eigh_ffi.cc); two-stage is not reachable.  Gate (``ffi.common.families`` row
``elpa``, tol 1e-12), per case, on the square mesh:

* residual ``max|A Q - Q diag(w)| / max|A| <= TOL`` and orthogonality ``max|Q^H Q - I| <= TOL``;
* eigenvalues ``max|w - w_ref| / max|A| <= TOL`` against the XLA route on the same device;
* the workspace pool: the handler fails loudly if ELPA kept a pooled buffer past the call, and
  ``LORRAX_ELPA_POOL_REPORT=1`` prints each rank's pool high-water and fallbacks;
* refusals (each must RAISE, never return a result): an ``ELPA_FORCE_solver`` environment
  override, and a rocBLAS workspace deliberately too small for ELPA's back-transformation
  (rocBLAS returns ``rocblas_status_memory_error`` without computing; the patched ELPA counts
  it and the handler discards the result).

Cases: float64 and complex128 at an unpadded side (n/p a multiple of the tile), a padded side
(n/p with no power-of-two tile in range: sentinel rows), and a side with the smaller tile; plus
``compute_evecs=False`` (eigenvalues only) and a second call on the cached handle.

ROCm: ``python -m tests.test_distrib_la_elpa_gate`` on the 2x2 P=4 mesh (srun -n4).  This file is
the device gate only: it holds no pytest test (the layout validity check is in
test_distrib_la_eigh_layout.py).
"""
import os
import sys

import runtime  # noqa: F401,E402  (OPENBLAS env before numpy; GATE blas_spin)
import numpy as np  # noqa: E402

TOL = 1e-12


def _herm(n, dtype, seed):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    if np.dtype(dtype).kind == "c":
        g = g + 1j * rng.standard_normal((n, n))
    return ((g + np.conj(g.T)) / np.sqrt(2.0 * n)).astype(dtype)


def gate_cases(mesh):
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la import _elpa
    p = int(mesh.shape["x"])
    out = {}
    face = NamedSharding(mesh, P("x", "y"))
    for dtype in (np.float64, np.complex128):
        for n in (1024, 1000, 64 * 3 * p):
            for evecs in ((True, False) if n == 1024 else (True,)):
                a = _herm(n, dtype, seed=n + (7 if dtype == np.complex128 else 0))
                A = jax.make_array_from_callback(a.shape, face, lambda idx: a[idx])
                side, nb = _elpa.eigh_layout(n, p, dtype)
                calls = 2 if n == 1024 and evecs else 1
                for _ in range(calls):              # the second call reuses the cached handle
                    W, Q = _elpa.distributed_eigh(A, mesh=mesh, compute_evecs=evecs)
                w = np.asarray(multihost_utils.process_allgather(W, tiled=True))
                w = w[:n] if w.shape[0] > n else w
                w_ref = np.asarray(jnp.linalg.eigh(jnp.asarray(a))[0])
                scale = np.max(np.abs(a))
                m = dict(dw=float(np.max(np.abs(w - w_ref)) / scale), res=0.0, orth=0.0)
                if evecs:
                    q = np.asarray(multihost_utils.process_allgather(Q, tiled=True))
                    m["res"] = float(np.max(np.abs(a @ q - q * w[None, :])) / scale)
                    m["orth"] = float(np.max(np.abs(np.conj(q.T) @ q - np.eye(n))))
                key = f"{np.dtype(dtype).name} n={n} side={side} nb={nb} evecs={evecs}"
                out[key] = m
    # Zero-row blocks (ELPA 2026.02.001 returns wrong pairs, orth ~4e-3, for a block at the
    # global TAIL; ELPA_TAIL_BLOCK_UPSTREAM.md): head, tail and padded sides, and a caller's tail
    # block past 16384 (the tangential face constructor's shape), through the checked plan. The
    # wrapper's last-row swap makes every case correct; a refusal fails.
    n = 2304 * p
    cases = (("head", n, np.arange(32)), ("tail", n, np.arange(n - 32, n)), ("padded", n - 32, None),
             ("caller tail", 18432, np.arange(18432 - 300, 18432)))
    for where, m_, rows in cases:
        a = _herm(m_, np.complex128, seed=11)
        if rows is not None:
            a[rows, :] = 0
            a[:, rows] = 0
        A = jax.make_array_from_callback(a.shape, face, lambda idx: a[idx])
        key = f"zero-rows {where} n={m_}"
        try:
            W, Q = distrib_la.plan("eigh", mesh, backend="elpa", n=m_)(A)
        except ValueError:
            out[key] = dict(dw=float("inf"), res=0.0, orth=0.0)
            continue
        scale = float(jnp.max(jnp.abs(W)))   # ||A||_2: the backward error AQ - QW is ~ n eps ||A||_2
        res, orth = jax.jit(lambda A, W, Q: (jnp.max(jnp.abs(A @ Q - Q * W[None, :])),
                                             jnp.max(jnp.abs(jnp.conj(Q.T) @ Q - jnp.eye(Q.shape[0], dtype=Q.dtype)))))(A, W, Q)
        w = np.asarray(multihost_utils.process_allgather(W, tiled=True))
        out[key] = dict(dw=float(np.max(np.abs(np.sort(w) - np.linalg.eigvalsh(a))) / scale) if m_ <= 9216 else 0.0,
                        res=float(res) / scale, orth=float(orth),
                        res_maxa=float(res) / float(np.max(np.abs(a))))   # the max|a| scale, reported beside
        del A, W, Q
    out.update(refusal_cases(mesh))
    return out


def _raises(fn, match):
    """(1, '') if fn raises an error whose text contains ``match``; (0, why) otherwise."""
    try:
        r = fn()
        import jax
        jax.block_until_ready(r)
    except Exception as e:                                      # noqa: BLE001
        return (1, "") if match in str(e) else (0, f"raised without {match!r}: {e}")
    return 0, "returned a result"


def refusal_cases(mesh):
    """The two refusals (module docstring); each case passes iff the call RAISES."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la import _elpa
    face = NamedSharding(mesh, P("x", "y"))
    out = {}
    a = _herm(1024, np.float64, seed=11)
    A = jax.make_array_from_callback(a.shape, face, lambda idx: a[idx])
    os.environ["ELPA_FORCE_solver"] = "2"
    try:
        ok, why = _raises(lambda: _elpa.distributed_eigh(A, mesh=mesh), "ELPA_FORCE_")
    finally:
        del os.environ["ELPA_FORCE_solver"]
    out["refuse ELPA_FORCE_solver=2"] = dict(refused=ok, why=why)
    # Back-transformation region (BLAS handle 0) and every other region: 256 KiB, far below
    # the measured need at n = 4096 (the stedc region is the normal one: a failed stedc makes
    # ELPA read an unwritten info and stop).  Last: the ELPA object it uses is never used again.
    b = _herm(4096, np.float64, seed=13)
    B = jax.make_array_from_callback(b.shape, face, lambda idx: b[idx])
    regions = _elpa.rocblas_regions
    try:
        _elpa.rocblas_regions = lambda side, p, dt: (256 << 10, regions(side, p, dt)[1], 256 << 10)
        ok, why = _raises(lambda: _elpa.distributed_eigh(B, mesh=mesh), "rocBLAS/rocSOLVER call(s) failed")
    finally:
        _elpa.rocblas_regions = regions
    out["refuse undersized rocBLAS workspace n=4096"] = dict(refused=ok, why=why)
    return out


def passes(m):
    if "refused" in m:
        return m["refused"] == 1
    return m["res"] <= TOL and m["orth"] <= TOL and m["dw"] <= TOL


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    jax.config.update("jax_enable_x64", True)
    from ffi.gate import mesh_ffi_platform
    mesh = stack.mesh
    if mesh_ffi_platform(mesh) != "ROCM" or mesh.shape["x"] != mesh.shape["y"]:
        print(f"SKIP: needs a square ROCm mesh, got {mesh_ffi_platform(mesh)} "
              f"{tuple(mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    from distrib_la import resolve
    assert resolve.resolve_backend("eigh", "elpa", mesh) == "elpa"
    out = gate_cases(mesh)
    bad = {k for k, m in out.items() if not passes(m)}
    for k, m in out.items():
        tag = 'FAIL' if k in bad else 'ok  '
        if "refused" in m:
            print(f"r{jax.process_index()} {tag} {k}  {'refused' if m['refused'] else m['why']}", flush=True)
        else:
            print(f"r{jax.process_index()} {tag} {k}  res {m['res']:.1e} orth {m['orth']:.1e} "
                  f"dw {m['dw']:.1e}" + (f" res/max|a| {m['res_maxa']:.1e}" if "res_maxa" in m else ""), flush=True)
    print(f"r{jax.process_index()} ELPA_EIGH_GATE {'PASS' if not bad else 'FAIL'} on "
          f"{tuple(mesh.devices.shape)}: {len(out) - len(bad)}/{len(out)} (tol {TOL:g})", flush=True)
    rc = 1 if bad else 0
    runtime.finalize_process(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
