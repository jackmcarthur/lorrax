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
import os
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


# ---- pytest: the row's logic (CPU) ------------------------------------------------------

def test_the_row_is_rocm_only_and_the_span_rule():
    from distrib_la.local_lu import LOCAL_LU_ROUTE, local_lu_row
    assert set(LOCAL_LU_ROUTE) == {"ROCM"}
    row = LOCAL_LU_ROUTE["ROCM"]
    assert row.target == "lorrax_rocsolver_lu_solve" and row.min_span_bytes == 1 << 31
    assert row.tri_target == "lorrax_rocsolver_tri_solve"
    assert set(row.dtypes) == {"float32", "float64", "complex64", "complex128"}
    # The largest exact complex128 / float64 square systems of the native path stay native...
    assert local_lu_row((11585, 11585), (11585, 11585), np.complex128, "ROCM") is None
    assert local_lu_row((16384, 16384), (16384, 16384), np.float64, "ROCM") is None
    assert local_lu_row((16, 312, 312), (16, 312, 312), np.complex128, "ROCM") is None   # Fe
    # ...and the sizes the rocBLAS trsm defect hits are routed (A's span or B's).
    for n, dt in ((11776, np.complex128), (12288, np.complex128), (16512, np.float64)):
        assert local_lu_row((n, n), (n, n), dt, "ROCM") is row
        assert local_lu_row((n, n), (n, 1), dt, "ROCM") is row                 # A alone spans it
    assert local_lu_row((4096, 4096), (4096, 32769), np.complex128, "ROCM") is row  # B spans it
    assert local_lu_row((4096, 4096), (4096, 32768), np.complex128, "ROCM") is None  # exactly 2^31
    # complex64 / float32 at theirs (the span rule contains the defect's rule for every n)
    assert local_lu_row((16384, 16384), (16384, 16384), np.complex64, "ROCM") is None
    assert local_lu_row((16512, 16512), (16512, 16512), np.complex64, "ROCM") is row
    assert local_lu_row((23168, 23168), (23168, 23168), np.float32, "ROCM") is None
    assert local_lu_row((23296, 23296), (23296, 23296), np.float32, "ROCM") is row
    assert local_lu_row((12288, 12288), (12288,), np.complex128, "ROCM") is row     # a vector b
    for m in range(256, 40000, 128):
        for item in (4, 8, 16):
            if (m - 128) * (m + 1) * item >= 1 << 31:        # the defect's condition
                assert m * m * item > 1 << 31                # ... is always routed
    for leg in ("CUDA", "cpu"):
        assert local_lu_row((12288, 12288), (12288, 12288), np.complex128, leg) is None


def test_below_the_span_solve_local_is_jnp_linalg_solve_bit_for_bit():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from distrib_la import local_lu
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 40, 40)) + 1j * rng.normal(size=(4, 40, 40)) + 4 * np.eye(40)
    b = rng.normal(size=(4, 40, 5)) + 0j
    assert np.array_equal(np.asarray(local_lu.solve_local(jnp.asarray(a), jnp.asarray(b))),
                          np.asarray(jnp.linalg.solve(jnp.asarray(a), jnp.asarray(b))))


def test_the_target_is_in_both_rocm_tables_and_the_family_row():
    pytest.importorskip("jax")
    from ffi.common import ffi_loader, families
    from distrib_la import loader
    for t, sym in (("lorrax_rocsolver_lu_solve", "RocsolverLuSolveFfi"),
                   ("lorrax_rocsolver_tri_solve", "RocsolverTriSolveFfi")):
        assert loader._PLATFORMS["ROCM"]["targets"][t] == ffi_loader._PLATFORMS["ROCM"]["targets"][t] == sym
        assert families.family_of(t) == "rocsolver" and families.state("rocsolver", "ROCM")[0] == "built"
        for leg in ("CUDA", "cpu"):
            assert t not in loader._PLATFORMS[leg]["targets"] and t not in ffi_loader._PLATFORMS[leg]["targets"]


def test_the_handler_keeps_every_rocblas_operand_below_two_gib():
    """The span rule is in the handler source: the limit, the blocking and the reverse laswp."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "ffi", "cpp", "rocsolver",
                            "lu_solve_ffi.cc")).read()
    for needle in ("INT32_MAX) / static_cast<int64_t>(esize)", "b.limit / n", "(b.limit - n) / n + 1",
                   "(limit - k) / lda + 1", "pv, -1)", "pv, 1)"):
        assert needle in src, needle


def _on_rocm():
    try:
        import jax
        from lxkit import device_vendor, leg_for_vendor
        leg = leg_for_vendor(device_vendor(jax.devices()[0]))
        return leg is not None and leg.key == "ROCM"
    except Exception:                                                       # noqa: BLE001
        return False


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


@pytest.mark.skipif(not _on_rocm(), reason="GPU (ROCm) regression: needs an MI250X GCD")
@pytest.mark.parametrize("which", ["lu", "L", "LH"])
def test_large_n_triangular_route_is_exact_on_rocm(which):
    import jax
    jax.config.update("jax_enable_x64", True)
    out = run_tri(12288, "complex128", 1024, which)
    assert out["routed"] and out["finite"] and out["resid"] <= TOL, out


if __name__ == "__main__":
    rc = main()
    runtime.finalize_process(rc)
    sys.exit(rc)
