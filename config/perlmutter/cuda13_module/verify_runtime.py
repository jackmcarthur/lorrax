"""One combined P=4 acceptance leg for the Perlmutter CUDA-13 module lane."""

# ruff: noqa: E402 -- distributed initialization must precede JAX imports.

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "distrib_la" / "tests"))

# LORRAX owns distributed initialization and must do it before any call that
# initializes the JAX backend.  This also exercises the real startup gates.
import runtime

runtime.initialize_communicator_stack()

import jax
import jax.numpy as jnp
import test_distrib_la_contract as contract
from jax.sharding import PartitionSpec as P

from ffi.fft import make_flat_k_fft_ffi


def announce(message: str) -> None:
    if jax.process_index() == 0:
        print(message, flush=True)


def check_stack_identity() -> None:
    expected = {
        "jax": "0.9.1",
        "jaxlib": "0.9.1",
        "jax-cuda13-plugin": "0.9.1",
        "jax-cuda13-pjrt": "0.9.1",
        "nvidia-cudnn-cu13": "9.12.0.46",
        "nvidia-cusolvermp-cu13": "0.9.1.9318.post1",
        "nvidia-cublasmp-cu13": "0.10.0.3695",
    }
    got = {name: importlib.metadata.version(name) for name in expected}
    assert got == expected, f"version drift: got {got}, expected {expected}"
    assert jax.process_count() == 4, jax.process_count()
    assert jax.device_count() == 4, jax.devices()
    assert jax.local_device_count() == 1, jax.local_devices()

    so = Path(os.environ["LORRAX_FFI_SO"])
    needed = subprocess.check_output(
        ["readelf", "-d", str(so)], text=True
    )
    assert "libcudart.so.13" in needed, needed
    assert "libcudart.so.12" not in needed, needed
    assert "libmpi" not in needed, needed
    announce(f"PASS stack identity: {got}; four ranks/four GPUs; CUDA-13 ELF")


def run_distributed_linalg(mesh) -> None:
    checks = (
        ("cusolvermp eigh complex128", contract.check_cusolvermp_eigh,
         (mesh, "complex128")),
        ("cusolvermp eigh float64", contract.check_cusolvermp_eigh,
         (mesh, "float64")),
        ("cusolvermp potrf/potrs", contract.check_cusolvermp_chol,
         (mesh, "complex128")),
        ("cusolvermp getrf/getrs", contract.check_cusolvermp_lu,
         (mesh, "complex128")),
        ("cublasmp gemm", contract.check_cublasmp_gemm,
         (mesh, "complex128")),
        ("cublasmp fused W solve", contract.check_cublasmp_wsolve,
         (mesh, "complex128")),
    )
    for label, fn, args in checks:
        fn(*args)
        announce(f"PASS {label}")

    # Production consumer: eigenvalues, inverted energies, and the
    # phase-invariant eigenvector density matrix must match native JAX.
    contract.check_compute_wfns_fi_backend(mesh, "cusolvermp")
    announce("PASS BSE setup consumer: cuSOLVERMp eigh matches native")


def check_cufft(mesh) -> None:
    kgrid = (2, 3, 4)
    rng = np.random.default_rng(23)
    x = (rng.standard_normal((24, 7))
         + 1j * rng.standard_normal((24, 7))).astype(np.complex128)
    fn = make_flat_k_fft_ffi(
        mesh,
        kgrid,
        P(None, None, None, None),
        kind="fftn",
        norm="backward",
        out_spec=None,
    )
    global_result = jax.jit(fn)(jnp.asarray(x))
    # The k/trailing axes are replicated for this handler.  Each process can
    # compare its one local replica without trying to fetch remote devices.
    got = np.asarray(global_result.addressable_data(0))
    want = np.fft.fftn(
        x.reshape(*kgrid, 7), axes=(0, 1, 2)
    ).reshape(x.shape)
    err = float(np.max(np.abs(got - want)))
    assert err < 1e-11, err
    announce(f"PASS cuFFT nonuniform batched flat-k: max error {err:.3e}")


def main() -> None:
    mesh = contract._mesh_from_arg("2x2")
    announce(f"source={ROOT} revision=" + subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip())
    check_stack_identity()
    run_distributed_linalg(mesh)
    check_cufft(mesh)
    announce("CUDA13 MODULE ACCEPTANCE: ALL PASSED")


if __name__ == "__main__":
    main()
