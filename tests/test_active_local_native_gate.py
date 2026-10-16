"""The native active-range local GEMM (``lorrax_cublas_local_*``) against the XLA routes on the
same device.

Two native routes.  ``"device"`` (``ACTIVE_LOCAL_ROUTE``'s ROCM row; ``_active_local_device``,
``src/ffi/cpp/active_gemm/``): one run-time-compiled kernel that reads the bounds on the device.
``"native"`` (the CUDA row; built and gated on ROCM too) is one leg-neutral source (``src/ffi/cpp/cublas/local_active_gemm_ffi.cc`` over ``common/gpu_blas.h``:
cuBLAS / rocBLAS strided-batched GEMMs on pointer views of the original buffers).  Gate
(``ffi.common.families`` row ``cublas_local``, tol 1e-13 relative to max|reference|):

* raw targets on each process's GCD -- the aliased ``_native`` (beta 0 and complex beta), the
  ``_out`` target (beta 0, no C), the ``_prepared`` target (bounds as attributes) -- and the
  guarded Python arm ``active_local_cuda`` against ``masked_local_matmul``,
  ``active_local_matmul`` (interval) and a host reference, for mixed / equal / full / empty
  intervals with NaN in every inactive column of A, row of B and entry of w, complex128 and
  float64, and the two Fe Sigma tau shapes;
* the plan: ``local_gemm_plan(enable_active_range=True)`` on the run's mesh (P=1 or the 2x2
  P=4 mesh), ``active_range`` and ``prepare_active_range``, native vs the plans rebuilt with the
  row set to ``masked`` and ``interval``, compared per addressable shard;
* invalid bounds: the guarded arm gives all-NaN (no native call); the raw target refuses
  with the CUDA contract's message (``0 <= lo <= hi <= storage K``).

Plus a timing section (one GCD): N chained calls in ONE executable (an optimization barrier
per step, so no call is CSE'd), native (per-call bounds read + stream sync) vs prepared (no
read) vs masked, so the bounds read's cost is the native - prepared difference.

Device: ``python -m tests.test_active_local_native_gate [--time]`` on any mesh.  Under pytest
the device gate skips; the rows' CPU checks run.
"""
import sys

# runtime before numpy (GATE blas_spin).
import runtime  # noqa: F401,E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TOL = 1e-13
NQ, M, K, N = 5, 7, 37, 6
BOUNDS = {
    "mixed": np.array([[0, 37], [3, 20], [11, 12], [5, 5], [16, 37]], np.int32),
    "equal": np.array([[4, 29]] * NQ, np.int32),
    "full": np.array([[0, K]] * NQ, np.int32),
    "empty": np.array([[9, 9]] * NQ, np.int32),
}
# The Fe Sigma tau shapes (sigma_tau_kernel_bins.csv): W(tau) synthesis c128[64,156,156] over
# the pole axis (K 936 at the deck's cap) and the Green build c128[64,312,312] over bands.
FE_SHAPES = {"fe_w": (64, 156, 936, 156), "fe_g": (64, 312, 36, 312)}   # Fe window module


def _draw(rng, shape, dtype):
    x = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def _operands(rng, nq, m, k, n, dtype):
    return (_draw(rng, (nq, m, k), dtype), _draw(rng, (nq, k, n), dtype),
            _draw(rng, (nq, k), dtype), _draw(rng, (nq, m, n), dtype))


def _poison(a, b, w, bounds):
    """NaN in every inactive column of a, row of b, entry of w."""
    a, b, w = a.copy(), b.copy(), w.copy()
    k = a.shape[-1]
    for q, (lo, hi) in enumerate(bounds):
        dead = np.r_[0:lo, hi:k]
        a[q][:, dead] = np.nan
        b[q][dead] = np.nan
        w[q][dead] = np.nan
    return a, b, w


def _reference(a, b, w, bounds, c, alpha, beta):
    out = np.zeros((a.shape[0], a.shape[1], b.shape[2]), a.dtype)
    for q, (lo, hi) in enumerate(bounds):
        out[q] = (a[q][:, lo:hi] * w[q][lo:hi]) @ b[q][lo:hi]
    out = alpha * out
    return out + beta * c if beta != 0 else out


def _fe_bounds(rng, nq, k):
    """Mixed per-parent windows as the tau loop sees them: runs of equal rows, some empty."""
    lo = np.sort(rng.integers(0, k // 2, nq // 8)).repeat(8)
    hi = np.minimum(k, lo + rng.integers(0, k // 2, nq // 8).repeat(8))
    b = np.stack([lo, hi], 1).astype(np.int32)
    b[5] = (b[5, 0], b[5, 0])                                     # one empty row
    return b


def _rel(x, ref):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        return float("inf")
    return float(np.abs(x - ref).max() / max(np.abs(ref).max(), 1.0))


def raw_cases(platform):
    """Every raw target and the guarded arm on this process's first GCD."""
    import jax
    import jax.numpy as jnp
    from distrib_la import _active_local_cuda as nat
    from distrib_la import _active_local_device as dvc
    from distrib_la._active_local import active_local_matmul, masked_local_matmul

    dvc.require_active_local_device(platform)
    nat.require_active_local_cuda(platform)
    nat.require_prepared_active_local_cuda(platform)
    dev = jax.local_devices()[0]
    put = lambda x: jax.device_put(jnp.asarray(x), dev)
    out, loud, conj_bitwise = {}, {}, {}
    cases = [(name, (NQ, M, K, N), dt, bd) for name, bd in BOUNDS.items()
             for dt in (np.complex128, np.float64)]
    rng0 = np.random.default_rng(11)
    for name, shp in FE_SHAPES.items():
        cases.append((name, shp, np.complex128, _fe_bounds(rng0, shp[0], shp[2])))
    for name, (nq, m, k, n), dt, bounds in cases:
        rng = np.random.default_rng(20261016)
        a0, b0, w0, c = _operands(rng, nq, m, k, n, dt)
        a, b, w = _poison(a0, b0, w0, bounds)
        cplx = np.dtype(dt).kind == "c"
        alpha = np.complex128(1.25 + 0.5j) if cplx else np.complex128(1.25)
        for beta in (np.complex128(0.0), np.complex128(0.5 - 0.25j) if cplx else np.complex128(0.5)):
            tag = f"{name}/{np.dtype(dt).name}/beta{'0' if beta == 0 else 'c'}"
            ref = _reference(a0, b0, w0, bounds, c, alpha, beta)
            A, B, W, C, BD = (put(x) for x in (a, b, w, c, bounds))
            wa = jax.jit(lambda A, W: A * W[:, None, :])(A, W)       # the arm's weighted A
            got = {}
            for guard in ("cond", "select"):
                got[f"arm_{guard}"] = jax.jit(lambda A, B, BD, W, C: nat.active_local_cuda(
                    A, B, BD, W, C if beta != 0 else None, alpha=alpha, beta=beta,
                    guard=guard))(A, B, BD, W, C)
            # "full": exact on finite inactive storage; on NaN storage, loud (NaN), never wrong
            A0, W0 = put(a0), put(w0)
            got["arm_full"] = jax.jit(lambda A, B, BD, W, C: nat.active_local_cuda(
                A, B, BD, W, C if beta != 0 else None, alpha=alpha, beta=beta,
                guard="full"))(A0, put(b0), BD, W0, C)
            pf = np.asarray(jax.jit(lambda A, B, BD, W, C: nat.active_local_cuda(
                A, B, BD, W, C if beta != 0 else None, alpha=alpha, beta=beta,
                guard="full"))(A, B, BD, W, C))
            ok = np.isnan(pf) | (np.abs(pf - ref) <= TOL * max(np.abs(ref).max(), 1.0))
            loud[tag] = bool(ok.all())
            got["device"] = jax.jit(lambda A, B, BD, W, C: dvc.active_local_device(
                A, B, BD, W, C if beta != 0 else None, alpha=alpha, beta=beta))(A, B, BD, W, C)
            # conj_b: the kernel reads conj(Bc) on load; on Bc = conj(b) it must reproduce the
            # plain call bit for bit (the same products in the same order).
            Bc = put(np.conj(b))
            got["device_conj_b"] = jax.jit(lambda A, B, BD, W, C: dvc.active_local_device(
                A, B, BD, W, C if beta != 0 else None, alpha=alpha, beta=beta,
                conj_b=True))(A, Bc, BD, W, C)
            conj_bitwise[tag] = bool(np.array_equal(np.asarray(got["device_conj_b"]),
                                                    np.asarray(got["device"]), equal_nan=True))
            got["device_raw_unweighted"] = jax.jit(lambda X, B, BD, C: dvc.device_call(
                X, B, BD, None, C if beta != 0 else None, alpha=alpha, beta=beta))(wa, B, BD, C)
            if np.all(bounds == bounds[0]):
                got["device_1pair"] = jax.jit(lambda A, B, BD, W, C: dvc.device_call(
                    A, B, BD[:1], W, C if beta != 0 else None, alpha=alpha, beta=beta))(A, B, BD, W, C)
            got["masked"] = jax.jit(lambda A, B, BD, W, C: masked_local_matmul(
                A, B, BD, W, C, alpha=alpha, beta=beta))(A, B, BD, W, C)
            got["interval"] = jax.jit(lambda A, B, BD, W, C: active_local_matmul(
                A, B, BD, W, C, alpha=alpha, beta=beta))(A, B, BD, W, C)
            got["native"] = jax.jit(lambda X, B, BD, C: nat._native(
                X, B, BD, C, alpha=alpha, beta=beta))(wa, B, BD, C)
            pairs = np.asarray(bounds, np.int64).reshape(-1)
            got["prepared"] = jax.jit(lambda X, B, C: nat._prepared_native(
                X, B, C, active_bounds=pairs, alpha=alpha, beta=beta))(wa, B, C)
            if beta == 0:
                got["native_out"] = jax.jit(lambda X, B, BD: nat._native_out(
                    X, B, BD, alpha=alpha))(wa, B, BD)
                # one shared bound pair (n_bounds == 1), the handler's other bounds shape
                if np.all(bounds == bounds[0]):
                    got["native_out_1pair"] = jax.jit(lambda X, B, BD: nat._native_out(
                        X, B, BD[:1], alpha=alpha))(wa, B, BD)
            ref_m = np.asarray(got["masked"])
            out[tag] = {k2: (_rel(v, ref), _rel(v, ref_m)) for k2, v in got.items()}
    # invalid bounds: the guarded arm gives NaN without a native call; the raw target refuses
    rng = np.random.default_rng(1)
    a, b, w, c = _operands(rng, NQ, M, K, N, np.complex128)
    A, B, W = (put(x) for x in (a, b, w))
    inval = {}
    for bad in ([-1, 4], [6, 3], [0, K + 1]):
        BD = np.array([bad] * NQ, np.int32)
        BD[1] = (2, 9)                                            # one valid row: still all-NaN
        BD = put(BD)
        for guard in ("cond", "select"):
            g = np.asarray(jax.jit(lambda A, B, BD, W: nat.active_local_cuda(
                A, B, BD, W, alpha=np.complex128(1), beta=np.complex128(0),
                guard=guard))(A, B, BD, W))
            inval[f"arm_{guard}{bad}"] = bool(np.all(np.isnan(g)))
        g = np.asarray(jax.jit(lambda A, B, BD, W: nat.active_local_cuda(
            A, B, BD, W, alpha=np.complex128(1), beta=np.complex128(0),
            guard="full"))(A, B, BD, W))
        inval[f"arm_full{bad}"] = bool(np.all(np.isnan(g)))
        g = np.asarray(jax.jit(lambda A, B, BD, W: dvc.active_local_device(
            A, B, BD, W, alpha=np.complex128(1), beta=np.complex128(0)))(A, B, BD, W))
        inval[f"device{bad}"] = bool(np.all(np.isnan(g)))
        g = np.asarray(jax.jit(lambda A, B, BD, W, C: dvc.active_local_device(
            A, B, BD, W, C, alpha=np.complex128(1), beta=np.complex128(0.5)))(A, B, BD, W, put(c)))
        inval[f"device_withC{bad}"] = bool(np.all(np.isnan(g)))
    for tag, v in loud.items():
        inval[f"full_poisoned_is_nan_or_exact {tag}"] = v
    for tag, v in conj_bitwise.items():
        inval[f"device_conj_b_bitwise {tag}"] = v
    try:
        BD = put(np.array([[6, 3]] * NQ, np.int32))
        np.asarray(jax.jit(lambda A, B, BD: nat._native_out(A, B, BD, alpha=np.complex128(1)))(A, B, BD))
        inval["raw_refuses"] = False
    except Exception as exc:                                      # noqa: BLE001
        inval["raw_refuses"] = "0 <= lo <= hi <= storage K" in str(exc)
    return out, inval


def plan_cases(mesh, platform):
    """local_gemm_plan active_range / prepare_active_range: native vs the XLA rows."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la import _active_local as al
    from distrib_la.matmul_plan import local_gemm_plan

    px, py = (mesh.shape[a] for a in ("x", "y"))
    nq, m, k, n = 16, 8 * px, 64, 8 * py
    rng = np.random.default_rng(5)
    a0, b0, w0, c = _operands(rng, nq, m, k, n, np.complex128)
    lo = rng.integers(0, 30, nq).astype(np.int32); lo[3] = 0
    hi = np.minimum(k, lo + rng.integers(0, 40, nq)).astype(np.int32); hi[3] = k; hi[7] = lo[7]
    bounds = np.stack([lo, hi], 1)
    a, b, w = _poison(a0, b0, w0, bounds)
    alpha, beta = np.complex128(0.75 - 0.5j), np.complex128(0.5 + 0.25j)
    ref = _reference(a0, b0, w0, bounds, c, alpha, beta)
    sa, sb, so = (NamedSharding(mesh, s) for s in (P(None, "x", None), P(None, None, "y"),
                                                   P(None, "x", "y")))
    glob = lambda x, s: jax.make_array_from_callback(x.shape, s, lambda idx: x[idx])
    row0 = al.ACTIVE_LOCAL_ROUTE[platform]
    res = {}
    try:
        for route in ("device", "native", "masked", "interval"):
            al.ACTIVE_LOCAL_ROUTE[platform] = route
            plan = local_gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=jnp.complex128,
                                   alpha=alpha, beta=beta, enable_active_range=True)
            A, B = glob(a, sa), glob(b, sb)
            got = plan.active_range(A, B, jnp.asarray(lo), jnp.asarray(hi), glob(c, so),
                                    weights=jnp.asarray(w))
            prep = plan.prepare_active_range(lo.tolist(), hi.tolist())
            got_p = prep(A, B, glob(c, so), weights=jnp.asarray(w))
            # conj_b on conj(b): every route must reproduce the plain contraction
            got_cj = plan.active_range(A, glob(np.conj(b), sb), jnp.asarray(lo), jnp.asarray(hi),
                                       glob(c, so), weights=jnp.asarray(w), conj_b=True)
            for label, arr in ((route, got), (route + "_prepared", got_p),
                               (route + "_conj_b", got_cj)):
                res[label] = max(_rel(s.data, ref[s.index]) for s in arr.addressable_shards)
    finally:
        al.ACTIVE_LOCAL_ROUTE[platform] = row0
    return res


def timing(platform, reps=64, rounds=5):
    """Seconds per call of N chained calls inside ONE executable (one GCD)."""
    import time
    import jax
    import jax.numpy as jnp
    from distrib_la import _active_local_cuda as nat
    from distrib_la import _active_local_device as dvc
    from distrib_la._active_local import masked_local_matmul

    dev = jax.local_devices()[0]
    out = {}
    one = np.complex128(1.0)
    shapes = dict(FE_SHAPES)
    shapes.update({f"{k2}_distinct": v for k2, v in FE_SHAPES.items()})
    shapes.update({f"{k2}_fullK": v for k2, v in FE_SHAPES.items()})   # live 1.0: kernel efficiency
    for name, (nq, m, k, n) in shapes.items():
        rng = np.random.default_rng(3)
        a, b, w, c = _operands(rng, nq, m, k, n, np.complex128)
        bounds = _fe_bounds(rng, nq, k)
        if name.endswith("_distinct"):          # every parent its own interval: nq BLAS calls
            lo = rng.integers(0, k // 2, nq)
            bounds = np.stack([lo, np.minimum(k, lo + rng.integers(1, k // 2, nq))], 1).astype(np.int32)
        if name.endswith("_fullK"):
            bounds = np.array([[0, k]] * nq, np.int32)
        pairs = bounds.astype(np.int64).reshape(-1)
        A, B, W, C, BD = (jax.device_put(jnp.asarray(x), dev) for x in (a, b, w, c, bounds))
        small = np.complex128(1e-3)

        def chain(step, ew=0):
            def f(A, B, BD, W, C):
                for _ in range(reps):                      # unrolled: no device loop trips
                    # a barrier per step: fresh operand values, so XLA cannot CSE the dots
                    A, B, BD, W, C = jax.lax.optimization_barrier((A, B, BD, W, C))
                    C = step(A, B, BD, W, C)
                    for _ in range(ew):                    # small unfusable kernels between calls
                        C = jax.lax.optimization_barrier(C * (1 + 1e-12) + 1e-30)
                return C
            return jax.jit(f)
        nat_step = lambda A, B, BD, W, C: nat._native(A, B, BD, C, alpha=small, beta=one)
        prep_step = lambda A, B, BD, W, C: nat._prepared_native(
            A, B, C, active_bounds=pairs, alpha=small, beta=one)
        mask_step = lambda A, B, BD, W, C: masked_local_matmul(A, B, BD, W, C, alpha=small, beta=one)
        arm_step = lambda A, B, BD, W, C: nat.active_local_cuda(A, B, BD, W, C, alpha=small, beta=one)
        dev_step = lambda A, B, BD, W, C: dvc.active_local_device(A, B, BD, W, C, alpha=small, beta=one)
        fns = {"native": chain(nat_step), "prepared": chain(prep_step), "masked": chain(mask_step),
               "arm": chain(arm_step),
               "arm_full": chain(lambda A, B, BD, W, C: nat.active_local_cuda(
                   A, B, BD, W, C, alpha=small, beta=one, guard="full")),
               "device": chain(dev_step),
               "device_unweighted": chain(lambda A, B, BD, W, C: dvc.device_call(
                   A, B, BD, None, C, alpha=small, beta=one)),
               "native+8ew": chain(nat_step, 8), "device+8ew": chain(dev_step, 8),
               "masked+8ew": chain(mask_step, 8)}
        for label, f in fns.items():
            f(A, B, BD, W, C).block_until_ready()
            best = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                f(A, B, BD, W, C).block_until_ready()
                best.append((time.perf_counter() - t0) / reps)
            out[f"{name}/{label}"] = (min(best), float(np.median(best)))
        live = float(np.mean(bounds[:, 1] - bounds[:, 0])) / k
        out[f"{name}/live_fraction"] = (live, live)
    return out


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    jax.config.update("jax_enable_x64", True)
    from ffi.gate import mesh_ffi_platform
    mesh = stack.mesh
    platform = mesh_ffi_platform(mesh)
    r = jax.process_index()
    if platform not in ("ROCM", "CUDA"):
        print(f"SKIP: needs a device mesh, got {platform}", flush=True)
        runtime.finalize_process(0)
        return 0
    bad = 0
    raw, inval = raw_cases(platform)
    for tag, d in raw.items():
        worst = max(v[0] for v in d.values())
        ok = worst <= TOL and max(v[1] for v in d.values()) <= TOL
        bad += not ok
        print(f"r{r} {'ok  ' if ok else 'FAIL'} raw {tag}  " +
              "  ".join(f"{k2} {v[0]:.1e}/{v[1]:.1e}" for k2, v in d.items()), flush=True)
    for k2, v in inval.items():
        bad += not v
        print(f"r{r} {'ok  ' if v else 'FAIL'} invalid {k2}", flush=True)
    pl = plan_cases(mesh, platform)
    for k2, v in pl.items():
        ok = v <= TOL
        bad += not ok
        print(f"r{r} {'ok  ' if ok else 'FAIL'} plan {k2} {v:.1e}", flush=True)
    if "--time" in sys.argv:          # every rank (GATE cross_rank_compile_agreement)
        for k2, (t, med) in timing(platform).items():
            print(f"r{r} time {k2} min {t * 1e6:.1f} us  median {med * 1e6:.1f} us"
                  if not k2.endswith("live_fraction") else f"r{r} time {k2} {t:.3f}", flush=True)
    n = len(raw) + len(inval) + len(pl)
    print(f"r{r} ACTIVE_LOCAL_NATIVE_GATE {'PASS' if not bad else 'FAIL'} on {platform} "
          f"{tuple(mesh.devices.shape)}: {n - bad}/{n} (tol {TOL:g})", flush=True)
    rc = 1 if bad else 0
    runtime.finalize_process(rc)
    return rc


# ---- pytest: the rows (CPU) -------------------------------------------------------------

def test_rows():
    pytest.importorskip("jax")
    from distrib_la._active_local import (ACTIVE_LOCAL_GUARD, ACTIVE_LOCAL_ROUTE,
                                          active_local_guard, active_local_route)
    assert ACTIVE_LOCAL_ROUTE["CUDA"] == "native" and ACTIVE_LOCAL_ROUTE["ROCM"] == "device"
    assert active_local_route("cpu") == "interval"
    assert ACTIVE_LOCAL_GUARD == {"CUDA": "cond", "ROCM": "select"}
    assert active_local_guard("tpu") == "cond"


def test_select_guard_lowers_with_no_device_control_flow():
    """The ROCM guard's point: the arm lowers with no cond (CUDA's keeps its two)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from distrib_la import _active_local_cuda as nat
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2)
    a, b, w, _ = _operands(rng, NQ, M, K, N, np.complex128)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(BOUNDS["mixed"]), jnp.asarray(w))
    one, zero = np.complex128(1), np.complex128(0)
    def text(guard):
        return jax.jit(lambda *x: nat.active_local_cuda(*x, alpha=one, beta=zero, guard=guard)
                       ).lower(*args).as_text()
    assert "stablehlo.case" not in text("select") and "stablehlo.while" not in text("select")
    assert text("cond").count("stablehlo.case") == 2


def test_device_arm_lowers_to_one_call_and_no_control_flow():
    """Owner rules: no host sync, no device control flow, no operand copy -- one custom call."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from distrib_la import _active_local_device as dvc
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2)
    a, b, w, c = _operands(rng, NQ, M, K, N, np.complex128)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(BOUNDS["mixed"]), jnp.asarray(w),
            jnp.asarray(c))
    text = jax.jit(lambda *x: dvc.active_local_device(*x, alpha=np.complex128(1),
                                                      beta=np.complex128(0.5))).lower(*args).as_text()
    assert text.count("stablehlo.custom_call") == 1 and "lorrax_active_gemm_device" in text
    for op in ("stablehlo.case", "stablehlo.while", "stablehlo.select", "stablehlo.multiply"):
        assert op not in text, op


def test_targets_in_both_rocm_tables_and_the_family_row():
    pytest.importorskip("jax")
    from ffi.common import ffi_loader, families
    from distrib_la import loader
    for t in ("lorrax_active_gemm_device", "lorrax_active_gemm_device_out"):
        sym = loader._PLATFORMS["CUDA"]["targets"][t]          # built on both device legs
        assert ffi_loader._PLATFORMS["CUDA"]["targets"][t] == sym
        assert loader._PLATFORMS["ROCM"]["targets"][t] == sym == ffi_loader._PLATFORMS["ROCM"]["targets"][t]
        assert families.family_of(t) == "active_gemm"
    assert families.state("active_gemm", "ROCM")[0] == "built"
    assert families.state("active_gemm", "CUDA")[0] == "built"
    assert families.FAMILIES["active_gemm"].tol == TOL
    for t in ("lorrax_cublas_local_active_range_gemm", "lorrax_cublas_local_active_range_gemm_out",
              "lorrax_cublas_local_prepared_active_range_gemm"):
        sym = loader._PLATFORMS["CUDA"]["targets"][t]
        assert loader._PLATFORMS["ROCM"]["targets"][t] == sym
        assert ffi_loader._PLATFORMS["ROCM"]["targets"][t] == sym
        assert families.family_of(t) == "cublas_local"
    assert families.state("cublas_local", "ROCM")[0] == "built"
    assert families.FAMILIES["cublas_local"].tol == TOL


def test_poison_marks_every_inactive_entry():
    rng = np.random.default_rng(0)
    a, b, w, _ = _operands(rng, NQ, M, K, N, np.complex128)
    pa, pb, pw = _poison(a, b, w, BOUNDS["mixed"])
    for q, (lo, hi) in enumerate(BOUNDS["mixed"]):
        assert np.all(np.isfinite(pa[q][:, lo:hi])) and np.isnan(pw[q][:lo]).all()
        assert np.isnan(pb[q][hi:]).all()


def test_device_gate_needs_a_device():
    pytest.skip("run as a script on a device mesh: python -m tests.test_active_local_native_gate")


if __name__ == "__main__":
    sys.exit(main())
