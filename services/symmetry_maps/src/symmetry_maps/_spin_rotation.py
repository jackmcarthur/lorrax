"""Local complex128 spin rotation; spatial symmetry stays in maps.py."""
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from lxkit import native_provider

from ._shard_map import shard_map

_TARGET = "lorrax_symmetry_spin_rotate"
_ABI = 3  # src/ffi/cpp/common/lorrax_ffi_abi.h
_SPECS = {
    "CUDA": dict(env="LORRAX_FFI_SO", so_name="liblorrax_ffi.so",
                 build_hint="build the LORRAX CUDA provider with SpinRotateCudaFfi"),
    "cpu": dict(env="LORRAX_FFI_HOST_SO", so_name="liblorrax_ffi_host.so",
                build_hint="build the LORRAX host provider"),
}


@lru_cache(maxsize=1)
def _register():
    path = native_provider.locate_library(
        "CUDA", specs=_SPECS, candidates={}, expected_abi=_ABI)
    lib, _ = native_provider.open_and_attest(
        path, platform="CUDA", expected_abi=_ABI,
        abi_symbols={"CUDA": "lorrax_ffi_cuda_abi_version"},
        build_hint=_SPECS["CUDA"]["build_hint"])
    if not hasattr(lib, "SpinRotateCudaFfi"):
        raise RuntimeError(f"{path} lacks SpinRotateCudaFfi; rebuild the CUDA provider")
    jax.ffi.register_ffi_target(
        _TARGET, jax.ffi.pycapsule(lib.SpinRotateCudaFfi), platform="CUDA")
    return lib  # Retain the provider for the registered handler's lifetime.


@lru_cache(maxsize=16)
def _kernel(mesh):
    _register()

    @shard_map(mesh=mesh, in_specs=(P(None, None, 'x', None, 'y'), P()),
               out_specs=P(None, None, 'x', None, 'y'), check_rep=False)
    def rotate(g, u):
        return jax.ffi.ffi_call(
            _TARGET, jax.ShapeDtypeStruct(g.shape, g.dtype),
            input_layouts=[(0, 1, 2, 3, 4), (0, 1, 2)],
            output_layouts=[(0, 1, 2, 3, 4)],
            input_output_aliases={0: 0})(g, u)

    return jax.jit(rotate)


def rotate_spin(spatial, spin, mesh):
    return _kernel(mesh)(spatial, jnp.asarray(spin, dtype=spatial.dtype))
