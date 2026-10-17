"""The ELPA distributed eigh (``lorrax_elpa_eigh``, ROCM leg) against ``jnp.linalg.eigh``.

``distrib_la._elpa.distributed_eigh`` runs ELPA 2026.02.001's two-stage solver on the AMD GPU
(config/frontier/build_elpa_hip.sh, cpp/elpa/eigh_ffi.cc).  Gate (``ffi.common.families`` row
``elpa``, tol 1e-12), per case, on the square mesh:

* residual ``max|A Q - Q diag(w)| / max|A| <= TOL`` and orthogonality ``max|Q^H Q - I| <= TOL``;
* eigenvalues ``max|w - w_ref| / max|A| <= TOL`` against the XLA route on the same device;
* the workspace pool: the handler fails loudly if ELPA kept a pooled buffer past the call, and
  ``LORRAX_ELPA_POOL_REPORT=1`` prints each rank's pool high-water and fallbacks.

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
#: MI250X, ELPA 2026.02.001 (ELPA_ROCM_REPORT.md table "pool"): (solver, dtype) -> [(n, p, GB)].
#: Two-stage is dtype-independent at every point measured both ways.
_P2 = [(4096, 2, 0.105), (16384, 2, 1.630), (20480, 2, 2.541), (30720, 2, 5.698),
       (40960, 2, 10.114), (51200, 2, 15.789), (30720, 4, 1.436), (51200, 4, 3.967)]
POOL_MEASURED = {
    (2, "complex128"): _P2, (2, "float64"): _P2,
    (1, "complex128"): [(16384, 2, 1.638), (20480, 2, 2.541), (30720, 2, 5.699), (30720, 4, 1.436),
                        (51200, 4, 3.967)],
    (1, "float64"): [(16384, 2, 1.101), (20480, 2, 1.702), (30720, 2, 3.811), (40960, 2, 6.759),
                     (30720, 4, 0.964), (51200, 4, 2.656)],
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
    from distrib_la import _elpa
    p = int(mesh.shape["x"])
    out = {}
    face = NamedSharding(mesh, P("x", "y"))
    for dtype in (np.float64, np.complex128):
        for n in (1024, 1000, 64 * 3 * p):
            for evecs in ((True, False) if n == 1024 else (True,)):
                a = _herm(n, dtype, seed=n + (7 if dtype == np.complex128 else 0))
                A = jax.make_array_from_callback(a.shape, face, lambda idx: a[idx])
                side, nb = _elpa.eigh_layout(n, p, dtype)          # the default (1-stage) layout
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
    return out


def passes(m):
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
        print(f"r{jax.process_index()} {'FAIL' if k in bad else 'ok  '} {k}  res {m['res']:.1e} "
              f"orth {m['orth']:.1e} dw {m['dw']:.1e}", flush=True)
    print(f"r{jax.process_index()} ELPA_EIGH_GATE {'PASS' if not bad else 'FAIL'} on "
          f"{tuple(mesh.devices.shape)}: {len(out) - len(bad)}/{len(out)} (tol {TOL:g})", flush=True)
    rc = 1 if bad else 0
    runtime.finalize_process(rc)
    return rc


# ---- pytest: layout, pool envelope and tables (CPU) ---------------------------------------

@pytest.mark.parametrize("n,p,dtype,solver,want", [
    (20480, 2, "complex128", 1, (20480, 64)), (20480, 2, "float64", 1, (20480, 64)),
    (51200, 4, "complex128", 1, (51200, 64)), (30720, 2, "float64", 1, (30720, 64)),
    (2 * 10000, 2, "float64", 1, (2 * 10048, 64)),    # 10000 = 2^4 * 625: padded
    (2 * 10000, 2, "complex128", 1, (2 * 10048, 64)),
    (2 * 50, 2, "complex128", 1, (2 * 64, 32)),        # local edge below the tile
    (20480, 2, "float64", 2, (20480, 128)), (20480, 2, "complex128", 2, (20480, 64)),
    (2 * 10048, 2, "float64", 2, (2 * 10048, 64)),    # 10048 = 2^6 * 157: the smaller tile
    (2 * 10000, 2, "float64", 2, (2 * 10112, 128)),
])
def test_layout(n, p, dtype, solver, want):
    pytest.importorskip("jax")
    from distrib_la._elpa import eigh_layout
    assert eigh_layout(n, p, dtype, solver) == want


@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("solver", [1, 2])
def test_every_layout_is_a_power_of_two_tile_with_less_than_one_tile_of_padding(p, dtype, solver):
    pytest.importorskip("jax")
    from distrib_la._elpa import eigh_layout
    for local in list(range(1, 300)) + list(range(300, 30000, 37)):
        side, nb = eigh_layout(p * local, p, dtype, solver)
        assert nb & (nb - 1) == 0 and side % p == 0 and (side // p) % nb == 0
        assert p * local <= side < p * (local + nb)


def test_the_pool_is_a_lower_envelope_of_the_measured_high_water():
    pytest.importorskip("jax")
    from distrib_la import _elpa
    assert _elpa.SOLVER == 1
    for (solver, dtype), rows in POOL_MEASURED.items():
        for n, p, gb in rows:
            b = _elpa.pool_bytes(n, p, dtype, solver)
            assert b <= (gb + 0.0005) * 1e9, (solver, dtype, n, p, b, gb)   # never above ELPA's use
            assert b >= 0.97 * gb * 1e9, (solver, dtype, n, p, b, gb)       # and within 3 % of it


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
