"""Local complex128 spin rotation; spatial symmetry stays in maps.py."""
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from lxkit import native_provider

from ._shard_map import shard_map

_TARGET = "lorrax_symmetry_spin_rotate_centroid"
_ABI = 6  # src/ffi/cpp/common/lorrax_ffi_abi.h (tests/test_ffi_abi_stamp.py)
_SPECS = {
    "CUDA": dict(env="LORRAX_FFI_SO", so_name="liblorrax_ffi.so",
                 build_hint="build the LORRAX CUDA provider with SpinRotateCentroidCudaFfi"),
    "cpu": dict(env="LORRAX_FFI_HOST_SO", so_name="liblorrax_ffi_host.so",
                build_hint="build the LORRAX host provider"),
}


def _checkout_candidates():
    """The checkout's build tree, the default the other two loaders search
    (``src/ffi/cpp/build{,_host}/``); a pinned ``LORRAX_FFI_SO`` still wins."""
    found = {}
    for name, sub in (("CUDA", "build"), ("cpu", "build_host")):
        for parent in Path(__file__).resolve().parents:
            cand = parent / "src" / "ffi" / "cpp" / sub / _SPECS[name]["so_name"]
            if cand.is_file():
                found[name] = [cand]
                break
    return found


@lru_cache(maxsize=1)
def _register():
    path = native_provider.locate_library(
        "CUDA", specs=_SPECS, candidates=_checkout_candidates(),
        expected_abi=_ABI)
    lib, _ = native_provider.open_and_attest(
        path, platform="CUDA", expected_abi=_ABI,
        abi_symbols={"CUDA": "lorrax_ffi_cuda_abi_version"},
        build_hint=_SPECS["CUDA"]["build_hint"])
    if not hasattr(lib, "SpinRotateCentroidCudaFfi"):
        raise RuntimeError(f"{path} lacks SpinRotateCentroidCudaFfi; rebuild the CUDA provider")
    jax.ffi.register_ffi_target(
        _TARGET, jax.ffi.pycapsule(lib.SpinRotateCentroidCudaFfi), platform="CUDA")
    return lib  # Retain the provider for the registered handler's lifetime.


@lru_cache(maxsize=16)
def _kernel(mesh):
    _register()

    @partial(shard_map, mesh=mesh, in_specs=(P(None, 'x', None, 'y', None), P()),
               out_specs=P(None, 'x', None, 'y', None), check_vma=False)
    def rotate(g, u):
        return jax.ffi.ffi_call(
            _TARGET, jax.ShapeDtypeStruct(g.shape, g.dtype),
            input_output_aliases={0: 0})(g, u)

    return jax.jit(rotate)


def rotate_spin(spatial, spin, mesh):
    return _kernel(mesh)(spatial, jnp.asarray(spin, dtype=spatial.dtype))
