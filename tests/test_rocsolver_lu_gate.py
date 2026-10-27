"""The ROCM one-device LU solve (``lorrax_rocsolver_lu_solve``, ``distrib_la.local_lu``).

rocBLAS 5 ``{z,d}trsm`` (left side; behind ``jnp.linalg.solve`` and ``rocsolver_getrs``) returns
a wrong solution with status 0 once a 128-block diagonal offset of the triangle crosses 2^31
bytes (complex128 n >= 11776, float64 n >= 16512 with a wide RHS; ROCSOLVER_LU_REPORT.md).
``local_lu.LOCAL_LU_ROUTE`` sends every system whose A or B spans more than 2^31 bytes to the handler
(rocSOLVER getrf + a triangular solve blocked so no rocBLAS operand spans 2^31 bytes); below
that ``jnp.linalg.solve`` runs unchanged.

Gate (``ffi.common.families`` row ``rocsolver``, tol 1e-12): ``||A X - B||_F / (||A||_F ||X||_F)``
per case, the handler forced at small n (against jnp.linalg.solve's residual on the same GCD)
and routed at large n, through the capacity of one GCD (``resolve.ONE_DEVICE_MAX_N``).
ROCm: ``python -m tests.test_rocsolver_lu_gate [--big]`` on one GCD.  Under pytest the CPU
checks of the row run; the large-n regression (``test_large_n_route_is_exact_on_rocm``) runs
only on a ROCm device (skipped elsewhere).
"""
import sys
import time

import runtime  # noqa: F401  (before numpy: runtime.blas_env)
import numpy as np
import pytest

TOL = 1e-12
TOL32 = 1e-5          #: float32 / complex64 (normwise backward error ~ 10-30 eps at n ~ 2e4)




def _tol(dt):
    return TOL32 if np.dtype(dt) in (np.float32, np.complex64) else TOL
#: (n, dtype, nrhs, batch): handler forced (small) -- odd sizes, narrow and wide RHS, a stack.
SMALL = ((312, "complex128", 312, 16), (312, "float64", 7, 16), (1000, "complex128", 3, 0),
         (2048, "complex128", 2048, 0), (4097, "float64", 100, 0), (3000, "complex128", 5000, 0))
#: routed by the span rule: the sizes the defect hits, and (--big) up to one GCD's capacity.
ROUTED = ((11776, "complex128", 11776, 0), (12288, "complex128", 12288, 0),
          (16512, "float64", 16512, 0), (16512, "complex64", 16512, 0), (23296, "float32", 23296, 0))
#: the defect's threshold pairs per dtype, (m - 128)(m + 1) itemsize >= 2^31 (native, reported)
THRESHOLDS = ((11648, "complex128"), (11776, "complex128"), (16384, "float64"), (16512, "float64"),
              (16384, "complex64"), (16512, "complex64"), (23168, "float32"), (23296, "float32"))
#: triangular modes: (n, dtype, nrhs, which) -- forced small, routed large
TRI = ((1000, "complex128", 37, "lu"), (1000, "complex128", 37, "L"), (1000, "complex128", 37, "LH"),
       (1000, "float32", 37, "LT"), (12288, "complex128", 2048, "lu"), (12288, "complex128", 2048, "L"),
       (12288, "complex128", 2048, "LH"), (16512, "complex64", 1024, "L"))
BIG = ((16384, "complex128", 16384, 0), (20480, "float64", 20480, 0),
       (20480, "complex128", 20480, 0), (27427, "complex128", 27427, 0))


def _system(n, dt, nrhs, batch):
    import jax
    import jax.numpy as jnp
    shp = (batch,) if batch else ()

    def mk(k, shape, scale):
        k1, k2 = jax.random.split(k)
        g = jax.random.normal(k1, shape, jnp.float64)
        if np.dtype(dt).kind == "c":
            g = g + 1j * jax.random.normal(k2, shape, jnp.float64)
        return (g * scale).astype(dt)
    A = jax.jit(lambda k: mk(k, shp + (n, n), 1 / np.sqrt(n)))(jax.random.PRNGKey(7))
    B = jax.jit(lambda k: mk(k, shp + (n, nrhs), 1.0))(jax.random.PRNGKey(11))
    return A, B


def _resid(A, X, B):
    import jax
    import jax.numpy as jnp
    fro = lambda x: jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))
    return float(jax.jit(lambda a, x, b: fro(jnp.matmul(a, x) - b) / (fro(a) * fro(x)))(A, X, B))


def run_native(n, dt):
    import jax
    import jax.numpy as jnp
    A, B = _system(n, dt, n, 0)
    X = jax.jit(jnp.linalg.solve)(A, B)
    return dict(n=n, dtype=dt, native_resid=_resid(A, X, B))


def run_tri(n, dt, nrhs, which):
    import jax
    import jax.numpy as jnp
    from distrib_la import local_lu as ll
    row = ll.LOCAL_LU_ROUTE["ROCM"]
    A, B = _system(n, dt, nrhs, 0)
    routed = ll.routes(A.shape, B.shape, A.dtype)
    if which == "lu":
        lu, piv, _ = jax.jit(jax.lax.linalg.lu)(A)
        fn = jax.jit(lambda lu, piv, b: ll._native_tri(row, lu, b, ll._MODE_LU, piv=piv))
        X = fn(lu, piv, B); T = A; del lu
    else:
        L = jax.jit(lambda a: jnp.tril(a) + 4 * jnp.eye(n, dtype=a.dtype))(A)
        mode = {"L": ll._MODE_L, "LT": ll._MODE_LT, "LH": ll._MODE_LT}[which]
        fn = jax.jit(lambda l, b: ll._native_tri(row, l, b, mode, conj=(which == "LH")))
        X = fn(L, B)
        T = {"L": L, "LT": L.T, "LH": jnp.conj(L.T)}[which]
    return dict(n=n, dtype=dt, nrhs=nrhs, which=which, routed=routed, resid=_resid(T, X, B),
                finite=bool(jnp.all(jnp.isfinite(X))))


def run_case(n, dt, nrhs, batch, force):
    import jax
    import jax.numpy as jnp
    from distrib_la import local_lu
    A, B = _system(n, dt, nrhs, batch)
    row = local_lu.local_lu_row(A.shape, B.shape, A.dtype)
    if force:
        fn = jax.jit(lambda a, b: local_lu._native_solve(local_lu.LOCAL_LU_ROUTE["ROCM"], a, b))
    else:
        assert row is not None, (n, dt, nrhs, "the span rule must route this system")
        fn = jax.jit(local_lu.solve_local)
    t = time.perf_counter(); X = fn(A, B); X.block_until_ready(); wall = time.perf_counter() - t
    r = _resid(A, X, B)
    ref = _resid(A, jax.jit(jnp.linalg.solve)(A, B), B) if force else float("nan")
    return dict(n=n, dtype=dt, nrhs=nrhs, batch=batch, routed=row is not None, wall_s=round(wall, 3),
                resid=r, native_resid=ref, finite=bool(jnp.all(jnp.isfinite(X))))


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    if leg is None or leg.key != "ROCM":
        print("[lu gate] not a ROCM device: nothing to gate", flush=True)
        return 0
    cases = [(c, True) for c in SMALL] + [(c, False) for c in ROUTED]
    if "--big" in sys.argv:
        cases += [(c, False) for c in BIG]
    bad = 0
    for (n, dt, nrhs, batch), force in cases:
        out = run_case(n, dt, nrhs, batch, force)
        ok = out["finite"] and out["resid"] <= _tol(dt)
        bad += not ok
        print(f"[lu gate] {'PASS' if ok else 'FAIL'} {out}", flush=True)
    for n, dt, nrhs, which in TRI:
        out = run_tri(n, dt, nrhs, which)
        ok = out["finite"] and out["resid"] <= _tol(dt)
        bad += not ok
        print(f"[lu gate] {'PASS' if ok else 'FAIL'} tri {out}", flush=True)
    if "--thresholds" in sys.argv:          # the defect itself, on the native path (reported)
        for n, dt in THRESHOLDS:
            print(f"[lu gate] native {run_native(n, dt)}", flush=True)
    print(f"[lu gate] {'PASS' if not bad else f'FAIL ({bad})'}", flush=True)
    return 1 if bad else 0


# ---- pytest: CPU checks of the row (the ROCm targets table check is in test_active_local_native_gate)

def _on_rocm():
    try:
        import jax
        from lxkit import device_vendor, leg_for_vendor
        leg = leg_for_vendor(device_vendor(jax.devices()[0]))
        return leg is not None and leg.key == "ROCM"
    except Exception:                                                       # noqa: BLE001
        return False


def test_workspace_query_modes_and_named_refusals(monkeypatch):
    """All modes use the native size owner and never hide a failed query."""
    import ctypes
    from types import SimpleNamespace
    from distrib_la import loader, local_lu as ll

    calls = []

    def query(mode, kind, n, nrhs, out, error, capacity):
        calls.append((mode, kind, n, nrhs))
        if n == 19:
            ctypes.memmove(error, b"query rejected\0", 15)
            return 1
        out._obj.value = 768 + (4 * n if mode == 0 else 0)
        return 0

    monkeypatch.setattr(loader, "get_lib", lambda leg: SimpleNamespace(
        lrx_rocsolver_lu_workspace_bytes=query))
    dtypes = ("float32", "float64", "complex64", "complex128")
    for mode in range(4):
        expected = 768 + (4 * 17 if mode == 0 else 0)
        for dtype in dtypes:
            assert ll._workspace_bytes(mode, 17, 5, dtype) == expected
    assert calls == [(mode, kind, 17, 5) for mode in range(4) for kind in range(4)]
    with pytest.raises(RuntimeError, match="GATE local_lu_workspace.*query rejected"):
        ll._workspace_bytes(2, 19, 5, "complex128")
    monkeypatch.setattr(loader, "get_lib", lambda leg: SimpleNamespace())
    with pytest.raises(RuntimeError, match="must export lrx_rocsolver_lu_workspace_bytes"):
        ll._workspace_bytes(2, 21, 5, "float64")


def test_native_call_declares_queried_workspace_for_matrix_and_vector_rhs(monkeypatch):
    """Trace the real bridge: even discarded scratch remains an FFI result.

    This checks the Python ABI without executing or emulating a native solve.
    Numerical and exact workspace-query checks require the ROCm gate.
    """
    import jax
    import jax.numpy as jnp
    from distrib_la import local_lu as ll

    queries = []

    def query(mode, n, nrhs, dtype):
        queries.append((mode, n, nrhs, dtype))
        return 1024 + mode

    monkeypatch.setattr(ll, "_require_target", lambda target: None)
    monkeypatch.setattr(ll, "_workspace_bytes", query)
    row = ll.LOCAL_LU_ROUTE["ROCM"]
    for vector in (False, True):
        a = jax.ShapeDtypeStruct((2, 17, 17), jnp.complex64)
        b = jax.ShapeDtypeStruct((2, 17) if vector else (2, 17, 5), jnp.complex64)
        for mode in range(4):
            fn = (lambda a, b: ll._native_solve(row, a, b)) if mode == 0 else (
                lambda a, b: ll._native_tri(row, a, b, mode))
            traced = jax.make_jaxpr(fn)(a, b)
            calls = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == "ffi_call"]
            assert len(calls) == 1
            scratch = calls[0].outvars[-1].aval
            assert scratch.shape == (1024 + mode,) and scratch.dtype == jnp.uint8
            assert traced.out_avals[0].shape == b.shape
            assert queries[-1] == (mode, 17, 1 if vector else 5, "complex64")


@pytest.mark.skipif(not _on_rocm(), reason="GPU (ROCm) regression: needs an MI250X GCD")
@pytest.mark.parametrize("n,dt", [(11776, "complex128"), (16512, "float64"), (16512, "complex64"),
                                  (23296, "float32")])
def test_large_n_route_is_exact_on_rocm(n, dt):
    """The regression: at the smallest sizes the rocBLAS trsm defect hits, the routed solve is
    exact (jnp.linalg.solve there: relative residual 5e-3 / 2e-3)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    out = run_case(n, dt, n, 0, force=False)
    assert out["routed"] and out["finite"] and out["resid"] <= _tol(dt), out


if __name__ == "__main__":
    rc = main()
    runtime.finalize_process(rc)
    sys.exit(rc)
