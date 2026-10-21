"""Tiny general-eigenpair checks; CPU is an oracle, CUDA lowering is explicit."""
from functools import partial
import ctypes
import io
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from distrib_la import (eig_local, eig_local_diagnostics, eig_local_workspace, local_batch,
                        resolve_eig_local_backend, solve_local)
from distrib_la import loader, local_eig


@pytest.fixture(autouse=True)
def _x64():
    resolve_eig_local_backend.cache_clear()
    loader._jax_geev_query_context.cache_clear()
    loader.local_geev_workspace.cache_clear()
    with jax.enable_x64(True):
        yield
    resolve_eig_local_backend.cache_clear()
    loader._jax_geev_query_context.cache_clear()
    loader.local_geev_workspace.cache_clear()


def _nonnormal(n=8):
    rng = np.random.default_rng(51)
    v = np.eye(n) + 0.08 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    w = np.linspace(-1.3, 2.1, n) + 0.3j * np.arange(n)
    return (v * w) @ np.linalg.inv(v), w


def test_nonnormal_modal_rows_reconstruct_resolvent():
    a, expected = _nonnormal()
    backend = resolve_eig_local_backend()

    @jax.jit
    def run(a):
        w, v = eig_local(a, backend=backend)
        dual = solve_local(v, jnp.eye(a.shape[-1], dtype=a.dtype))
        return w, v, dual, eig_local_diagnostics(a, w, v, dual)

    w, v, dual, diagnostics = jax.device_get(run(jnp.asarray(a)))
    np.testing.assert_allclose(np.sort_complex(w), np.sort_complex(expected), atol=2e-13)
    status, eig_error, inverse_error, condition = diagnostics
    assert status == 0
    assert eig_error < 1e-14 and inverse_error < 1e-14
    assert condition >= a.shape[-1] - 1e-12
    z = 3.2 + 0.7j
    modal = (v / (z - w)) @ dual
    np.testing.assert_allclose(modal, np.linalg.solve(z * np.eye(len(w)) - a, np.eye(len(w))),
                               atol=2e-13, rtol=2e-13)
    # This matrix is not normal: the dual rows cannot be replaced by V^H.
    assert np.linalg.norm(dual - v.conj().T) > 0.1


def test_batched_scalar_degenerate_and_nonnormal_matrices():
    a, _ = _nonnormal(6)
    stack = jnp.asarray(np.stack((a, np.zeros_like(a), 2j * np.eye(6))))
    backend = resolve_eig_local_backend()
    w, v = jax.jit(partial(eig_local, backend=backend))(stack)
    dual = solve_local(v, jnp.broadcast_to(jnp.eye(6, dtype=v.dtype), v.shape))
    status, errors, inverse_errors, _ = jax.jit(eig_local_diagnostics)(stack, w, v, dual)
    np.testing.assert_array_equal(np.asarray(status), np.zeros(3, dtype=np.int32))
    assert np.max(np.asarray(errors)) < 1e-14
    assert np.max(np.asarray(inverse_errors)) < 1e-14


def test_diagnostics_report_corrupted_and_nonfinite_results():
    a = jnp.diag(jnp.asarray([1 + 0j, 2 + 0j]))
    v = jnp.eye(2, dtype=jnp.complex128)
    good = eig_local_diagnostics(a, jnp.diag(a), v, v)
    bad_eigenvalue = eig_local_diagnostics(a, jnp.diag(a) + 1, v, v)
    bad_inverse = eig_local_diagnostics(a, jnp.diag(a), v, 2 * v)
    nonfinite = eig_local_diagnostics(a, jnp.diag(a), v, v.at[0, 0].set(jnp.nan))
    assert int(good[0]) == 0
    assert int(bad_eigenvalue[0]) == 2
    assert int(bad_inverse[0]) == 4
    assert int(nonfinite[0]) == 1


def test_large_condition_is_informative_not_a_failure():
    v = jnp.diag(jnp.asarray([1 + 0j, 1e-12 + 0j]))
    dual = jnp.diag(jnp.asarray([1 + 0j, 1e12 + 0j]))
    a = jnp.diag(jnp.asarray([1 + 0j, 2 + 0j]))
    status, _, _, condition = eig_local_diagnostics(a, jnp.diag(a), v, dual)
    assert int(status) == 0
    assert float(condition) > 1e11


def test_provider_refuses_gpu_host_fallback(monkeypatch):
    monkeypatch.setattr(local_eig, "_process_backend", lambda: "cusolver")
    with pytest.raises(RuntimeError, match="GPU host fallback is forbidden"):
        eig_local(jnp.eye(2, dtype=jnp.complex128), backend="lapack")


def test_unsupported_accelerator_refuses(monkeypatch):
    device = SimpleNamespace(platform="gpu", device_kind="AMD Instinct MI250X",
                             client=SimpleNamespace(platform="rocm", platform_version="rocm 7"))
    monkeypatch.setattr(local_eig.jax, "local_devices", lambda: [device])
    with pytest.raises(RuntimeError, match="no device-native general eigensolver on rocm"):
        resolve_eig_local_backend()


def test_compile_probe_refuses_without_fallback(monkeypatch):
    class FailedProbe:
        def lower(self, _):
            raise RuntimeError("cuSOLVER version too old")

    monkeypatch.setattr(local_eig, "_process_backend", lambda: "cusolver")
    monkeypatch.setattr(local_eig.jax, "jit", lambda *args, **kwargs: FailedProbe())
    with pytest.raises(RuntimeError, match="cusolver complex128 compile probe failed"):
        resolve_eig_local_backend()


def test_wrong_dtype_shape_and_provider_refuse():
    backend = resolve_eig_local_backend()
    with pytest.raises(TypeError, match="must be complex128"):
        eig_local(jnp.eye(2), backend=backend)
    with pytest.raises(ValueError, match="must be nonempty"):
        eig_local(jnp.zeros((2, 3), dtype=jnp.complex128), backend=backend)
    with pytest.raises(ValueError, match="unknown backend"):
        eig_local(jnp.eye(2, dtype=jnp.complex128), backend="auto")


def test_local_batch_composition_reconstructs_on_one_device():
    backend = resolve_eig_local_backend()
    mesh = Mesh(np.asarray(jax.local_devices()[:1], dtype=object).reshape(1, 1), ("x", "y"))

    def reconstruct(a):
        w, v = eig_local(a, backend=backend)
        return (v * w) @ solve_local(v, jnp.eye(a.shape[-1], dtype=a.dtype))

    a, _ = _nonnormal(4)
    stack = jnp.asarray(np.stack((a, 2 * a, 3 * a)))
    run = local_batch(reconstruct, mesh)
    np.testing.assert_allclose(np.asarray(run(stack)), np.asarray(stack), atol=2e-13, rtol=2e-13)


def test_actual_provider_lowering_names_device_geev():
    backend = resolve_eig_local_backend()
    text = jax.jit(partial(eig_local, backend=backend)).lower(
        jax.ShapeDtypeStruct((2, 2), jnp.complex128)).as_text()
    if backend == "cusolver":
        assert "cusolver_geev_ffi" in text
        assert "hybrid_eig" not in text and "lapack" not in text
    else:
        assert "lapack_zgeev_ffi" in text
    assert "host_callback" not in text and "xla_python_cpu_callback" not in text


def test_workspace_receipt_distinguishes_cpu_and_cuda(monkeypatch):
    if resolve_eig_local_backend() == "lapack":
        receipt = eig_local_workspace(8, backend="lapack")
        assert receipt["device_bytes"] == 0 and receipt["host_bytes"] is None
    monkeypatch.setattr(local_eig, "resolve_eig_local_backend", lambda: "cusolver")
    monkeypatch.setattr(loader, "local_geev_workspace", lambda n: (8192, 4096, "/jax/libcusolver.so.12"))
    receipt = eig_local_workspace(8, backend="cusolver")
    assert receipt == dict(device_bytes=8192, host_bytes=4096, provider="cusolverDnXgeev",
                           library_path="/jax/libcusolver.so.12", batch_workspace_reuse="serial")
    with pytest.raises(RuntimeError, match="actual provider is cusolver"):
        eig_local_workspace(8, backend="lapack")


def test_geev_query_matches_jax_abi_and_reuses_only_mapped_provider(monkeypatch):
    calls = []

    def create(handle):
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        calls.append("create")
        return 0

    def params(handle):
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 456
        return 0

    def query(*args):
        calls.append(("query", args))
        ctypes.cast(args[-2], ctypes.POINTER(ctypes.c_size_t))[0] = 8192
        ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_size_t))[0] = 4096
        return 0

    def cleanup(params):
        calls.append(("cleanup", params.value))
        return 0

    lib = SimpleNamespace(cusolverDnCreate=create, cusolverDnCreateParams=params,
                          cusolverDnDestroyParams=cleanup, cusolverDnXgeev_bufferSize=query)
    mapped = ("a-b r-xp 00000000 01:01 123 /jax/libcusolver.so.12\n"
              "b-c r--p 00000000 01:01 123 /jax/libcusolver.so.12\n")
    monkeypatch.setattr(loader, "open", lambda _: io.StringIO(mapped), raising=False)

    def load(path, *, mode):
        assert path == "/jax/libcusolver.so.12"
        assert mode & os.RTLD_NOLOAD
        return lib

    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    assert loader.local_geev_workspace(8) == (8192, 4096, "/jax/libcusolver.so.12")
    assert loader.local_geev_workspace(8) == (8192, 4096, "/jax/libcusolver.so.12")
    assert calls[0] == "create" and len(calls) == 3
    args = calls[1][1]
    assert args[:2][0].value == 123 and args[:2][1].value == 456
    assert args[2:-2] == (0, 1, 8, 5, None, 8, 5, None, 5, None, 8, 5, None, 8, 5)
    assert len(query.argtypes) == len(args) == 19
    assert calls[2] == ("cleanup", 456)


def test_geev_query_refuses_provider_error_and_releases_params(monkeypatch):
    cleaned = []

    def params(handle):
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 456
        return 0

    lib = SimpleNamespace(cusolverDnCreateParams=params,
                          cusolverDnXgeev_bufferSize=lambda *args: 3,
                          cusolverDnDestroyParams=lambda p: cleaned.append(p.value) or 0)
    monkeypatch.setattr(loader, "_jax_geev_query_context", lambda: (lib, ctypes.c_void_p(123), "/jax/libcusolver.so.12"))
    with pytest.raises(RuntimeError, match="Xgeev size query failed: status=3"):
        loader.local_geev_workspace(8)
    assert cleaned == [456]
