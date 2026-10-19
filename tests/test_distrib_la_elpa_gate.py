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

ROCm: ``python -m tests.test_distrib_la_elpa_gate`` on the 2x2 P=4 mesh (srun -n4).  Under
pytest the device gate skips; the layout, pool-envelope and table checks run on the CPU.
"""
import os
import sys

import runtime  # noqa: F401,E402  (OPENBLAS env before numpy; GATE blas_spin)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TOL = 1e-12

#: ELPA's live device high-water through the allocator hook (GB per rank, max over ranks),
#: MI250X, ELPA 2026.02.001, one-stage (ELPA_ROCM_REPORT.md table "pool"): dtype -> [(n, p, GB)].
POOL_MEASURED = {
    "complex128": [(16384, 2, 1.638), (20480, 2, 2.541), (30720, 2, 5.699), (30720, 4, 1.436),
                   (51200, 4, 3.967)],
    "float64": [(16384, 2, 1.101), (20480, 2, 1.702), (30720, 2, 3.811), (40960, 2, 6.759),
                (30720, 4, 0.964), (51200, 4, 2.656)],
}

#: rocBLAS's managed high-water per ELPA handle (GB), 2x2, job 5700309 (ELPA_ROCM_REPORT.md
#: addendum): (n, dtype) -> (BLAS handle 0, solver handle 0); every other handle stayed at
#: rocBLAS's 32 MiB default.
RB_MEASURED = {
    (20480, "complex128"): (0.203, 0.454), (51200, "complex128"): (0.454, 2.657),
    (20480, "float64"): (0.118, 0.454), (51200, "float64"): (0.244, 2.657),
}


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
                    # A padded side goes through the plan (the one padding mechanism:
                    # zero rows, deflated, dropped by count); the wrapper refuses it.
                    W, Q = (_elpa.distributed_eigh(A, mesh=mesh, compute_evecs=evecs) if side == n else
                            distrib_la.plan("eigh", mesh, backend="elpa", n=n)(A))
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
    # block past 16384 (the shared-pole face constructor's shape), through the checked plan. The
    # wrapper's last-row swap and head padding make every case correct; a refusal fails.
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


# ---- pytest: layout, pool envelope and tables (CPU) ---------------------------------------

@pytest.mark.parametrize("n,p,dtype,want", [
    (20480, 2, "complex128", (20480, 64)), (20480, 2, "float64", (20480, 64)),
    (51200, 4, "complex128", (51200, 64)), (30720, 2, "float64", (30720, 64)),
    (2 * 10000, 2, "float64", (2 * 10048, 64)),    # 10000 = 2^4 * 625: padded
    (2 * 10000, 2, "complex128", (2 * 10048, 64)),
    (2 * 50, 2, "complex128", (2 * 64, 32)),        # local edge below the tile
])
def test_layout(n, p, dtype, want):
    pytest.importorskip("jax")
    from distrib_la._elpa import eigh_layout
    assert eigh_layout(n, p, dtype) == want


@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_every_layout_is_a_power_of_two_tile_with_less_than_one_tile_of_padding(p, dtype):
    pytest.importorskip("jax")
    from distrib_la._elpa import eigh_layout
    for local in list(range(1, 300)) + list(range(300, 30000, 37)):
        side, nb = eigh_layout(p * local, p, dtype)
        assert nb & (nb - 1) == 0 and side % p == 0 and (side // p) % nb == 0
        assert p * local <= side < p * (local + nb)


def test_the_pool_is_a_lower_envelope_of_the_measured_high_water():
    pytest.importorskip("jax")
    from distrib_la import _elpa
    for dtype, rows in POOL_MEASURED.items():
        for n, p, gb in rows:
            b = _elpa.pool_bytes(n, p, dtype)
            assert b <= (gb + 0.0005) * 1e9, (dtype, n, p, b, gb)   # never above ELPA's use
            assert b >= 0.97 * gb * 1e9, (dtype, n, p, b, gb)       # and within 3 % of it


def test_the_rocblas_regions_cover_the_measured_need_with_the_margin():
    pytest.importorskip("jax")
    from distrib_la import _elpa
    for (n, dtype), (back_gb, stedc_gb) in RB_MEASURED.items():
        back, stedc, other = _elpa.rocblas_regions(n, 2, dtype)
        for got, gb in ((back, back_gb), (stedc, stedc_gb)):
            assert got >= 1.24 * (gb - 0.0005) * 1e9, (n, dtype, got, gb)     # the 25 % margin
            assert got <= 1.26 * (gb + 0.0005) * 1e9, (n, dtype, got, gb)     # and no more
        assert other == 32 << 20
        assert _elpa.rocblas_bytes(n, 2, dtype, threads=7) == back + stedc + 12 * other   # 14 handles
        assert _elpa.workspace_bytes(n, 2, dtype, threads=7) == \
            _elpa.pool_bytes(n, 2, dtype) + back + stedc + 12 * other


def test_two_stage_cannot_be_selected():
    """ELPA's two-stage path is unreachable: no Python knob, no FFI attribute, the handle
    pinned to ELPA_SOLVER_1STAGE (read back) and ELPA_FORCE_* overrides refused."""
    pytest.importorskip("jax")
    import inspect
    import pathlib
    from distrib_la import _elpa, resolve
    sig = inspect.signature(_elpa.distributed_eigh)
    assert "solver" not in sig.parameters
    with pytest.raises(TypeError):
        sig.bind(None, mesh=None, solver=2)
    assert not any("two_stage" in k.lower() or k == "SOLVER" for k in vars(_elpa))
    assert not any("2" in b or "stage" in b for b in resolve.EIGH_BACKENDS)
    cc = (pathlib.Path(__file__).resolve().parents[1] / "src/ffi/cpp/elpa/eigh_ffi.cc").read_text()
    assert '{"solver", ELPA_SOLVER_1STAGE}' in cc and "2STAGE" not in cc
    assert 'Attr<int64_t>("solver")' not in cc
    assert '"ELPA_FORCE_"' in cc and 'elpa_get(h, "solver"' in cc


def test_the_backend_is_in_every_table():
    pytest.importorskip("jax")
    from ffi.common import ffi_loader, families
    import importlib
    from distrib_la import loader, resolve
    plan = importlib.import_module("distrib_la.plan")      # the module (distrib_la.plan is the function)
    t = "lorrax_elpa_eigh"
    assert loader._PLATFORMS["ROCM"]["targets"][t] == ffi_loader._PLATFORMS["ROCM"]["targets"][t] \
        == "ElpaEighFfi"
    for leg in ("CUDA", "cpu"):
        assert t not in loader._PLATFORMS[leg]["targets"] and t not in ffi_loader._PLATFORMS[leg]["targets"]
    assert families.family_of(t) == "elpa" and families.state("elpa", "ROCM")[0] == "built"
    assert families.FAMILIES["elpa"].tol == TOL
    assert "elpa" in resolve.EIGH_BACKENDS and resolve._SPEC[("eigh", "elpa")] == (t, ("ROCM",))
    assert plan._IMPL[("eigh", "elpa")]["one"] == "distributed_eigh"
    from distrib_la import _elpa
    assert resolve.backend_module("elpa") is _elpa
    with pytest.raises(ValueError, match="SQUARE"):
        resolve._check_geometry("eigh", "elpa", 2, 4)


def test_the_closure_classifies_elpa_as_engine_private():
    from lxkit import native_provider as npv
    for name in ("libelpa_openmp.so.19", "libelpa.so.19"):
        assert npv.is_private_redistributable(name) and npv._MUST_PIN.search(name)
    assert not npv.is_private_redistributable("libscalapack.so.2.2")   # hidden inside libelpa


if __name__ == "__main__":
    sys.exit(main())
