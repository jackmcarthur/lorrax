"""The platform legs of the LORRAX native-provider ABI: one row per XLA platform.

A *leg* is one native library that registers LORRAX's FFI target names
under one XLA platform: the jaxlib split (cpu LAPACK, CUDA cuSOLVER) that
keeps every ``jax.ffi.ffi_call`` site platform-blind.  Every fact that
differs between legs and that more than one service needs lives in
:data:`LEGS`, once:

* ``key``: the XLA FFI platform name the leg registers under (``"cpu"``,
  ``"CUDA"``, ``"ROCM"``), which is also the FFI key every gate,
  router and preference row is written in;
* ``vendor``: what :func:`lxkit.device_vendor` reads from a device of the leg;
* the library file, its absolute-path pin and its in-tree build directory;
* the C entry-point suffix (``cpp/common/c_abi.h``) and the ABI-stamp symbol;
* how a process can tell, before any JAX backend exists, that it can host
  the leg: the ``JAX_PLATFORMS`` names, the visibility variables whose empty
  value masks the vendor, and the device nodes;
* the runtime policy that differs by vendor (owner ``runtime``; evidence in
  ``docs/architecture/compilation.md`` section 3 and ``memory-model.md``): the
  ``jax_plugins`` directory prefix of the leg's PJRT plugin, the XLA flags the
  runtime merges into ``XLA_FLAGS`` (a caller's value wins, flag by flag), the
  allocator the GPU pool policy sets, and the API the runtime reads the card's
  total memory through.  Each is a *declaration*; ``runtime`` implements it;
* ``runtime_kernel_caches``: the leg's vendor libraries that compile device
  kernels at run time and keep them across processes only where a variable
  points (rocFFT's hipRTC kernels, ``ROCFFT_RTC_CACHE_PATH``; AMD comgr's
  code-object cache, ``AMD_COMGR_CACHE_DIR``, default ``~/.cache/comgr``).
  :mod:`common.jax_compile_cache` points each beside the persistent compile
  cache (Precursors lane, 2026-10-14);
* the vendor facts the memory model, the network policy and the profiler
  guard implement (owner ``runtime``; Runtime-LEGS lane, 2026-10-15):
  ``fft_plan_scratch`` (whose plan-workspace query prices the FFT scratch
  XLA's buffer assignment cannot see, :mod:`runtime.aot_memory`),
  ``outside_pool_api`` and ``memory_telemetry`` (the readers of bytes outside
  the XLA pool and of whole-device used memory, :mod:`runtime.xla_memory`),
  ``collective_library`` and ``collective_net_plugin`` (the collective
  library of the leg's XLA collectives, which names the site profile's
  section, and the net plugin it dlopens by name,
  :mod:`runtime.network_env`), and ``profiler`` (the vendor tracer and the
  largest step it does not deadlock, ``runtime.check_profiler_ranks``).

What stays OUT is per-service: target-symbol tables, ctypes declarations,
build hints and backend preference rows (the "policy, never tables" rule
in :mod:`lxkit`).  A leg in :data:`LEGS` need not have a library on any
machine: ``ROCM`` is declared before its library exists, so a ROCm mesh
resolves to a named leg whose missing ``.so`` is refused by name like any
other, instead of an unmapped vendor string.

Stdlib-only, like the rest of :mod:`lxkit`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional

__all__ = [
    "PlatformLeg", "LEGS", "HOST", "accelerator_legs", "leg_for_vendor", "leg_for_jax_platform",
    "leg_device_present", "process_accelerator_leg", "accelerator_leg_to_open_first",
]


@dataclass(frozen=True)
class PlatformLeg:
    """One native-provider leg (see the module docstring for each field's owner)."""

    key: str                        #: XLA FFI platform name: "cpu" | "CUDA" | "ROCM"
    vendor: str                     #: lxkit.device_vendor() of the leg's devices
    so_name: str                    #: library file name
    so_env: str                     #: absolute-path pin of that file
    build_subdir: str               #: in-tree build dir, relative to src/ffi/cpp/
    c_abi_suffix: str               #: LRX_C_ENTRY suffix on this leg
    abi_symbol: str                 #: C entry point that reports the handler ABI
    accelerator: bool               #: a device leg (opened before the host leg)
    jax_platforms: tuple = ()       #: JAX_PLATFORMS first entries that select it
    visibility_envs: tuple = ()     #: empty value of any of these masks the vendor
    device_globs: tuple = ()        #: device nodes (or driver shims) that mean present
    jax_plugin_prefix: str = ""     #: ``jax_plugins/<prefix>*`` of the leg's PJRT plugin
    xla_flag_defaults: tuple = ()   #: ``(flag, value)`` pairs runtime merges into XLA_FLAGS
    xla_allocator: str = ""         #: XLA_PYTHON_CLIENT_ALLOCATOR of the pool policy ("" = jaxlib's)
    xla_allocators: tuple = ()      #: allocators the leg's PJRT plugin can build (() = unchecked)
    device_memory_api: str = ""     #: card-total reader in runtime.xla_memory ("" = none)
    #: ``(env, name, scope)``: run-time-compiled vendor kernels; scope
    #: ``"process-file"`` (one file per process) or ``"shared-dir"``.
    runtime_kernel_caches: tuple = ()
    #: plan-workspace query in runtime.aot_memory (``"cufft"`` | ``"hipfft"``);
    #: "" = the leg's FFTs take no plan scratch outside buffer assignment.
    fft_plan_scratch: str = ""
    #: bytes-outside-the-pool reader in runtime.xla_memory ("" = none).
    outside_pool_api: str = ""
    #: whole-device used-memory sampler in runtime.xla_memory ("" = none).
    memory_telemetry: str = ""
    #: collective library of the leg's XLA collectives; the section of
    #: ``config/<site>/site.toml`` that runtime.network_env applies ("" = none).
    collective_library: str = ""
    #: net plugin file the collective library dlopens by name when
    #: ``NCCL_NET_PLUGIN`` is unset.
    collective_net_plugin: str = ""
    #: ``(tool, mapped-library marker, largest process count)``: the vendor
    #: tracer, how a process knows it runs under it, and the largest step it
    #: does not deadlock (0 = no limit).
    profiler: tuple = ()


HOST = PlatformLeg(
    key="cpu", vendor="cpu",
    so_name="liblorrax_ffi_host.so", so_env="LORRAX_FFI_HOST_SO",
    build_subdir="build_host", c_abi_suffix="_host",
    abi_symbol="lorrax_ffi_host_abi_version", accelerator=False,
    jax_platforms=("cpu",))

#: Every leg, by XLA platform key.  The CUDA library keeps its historical
#: file name and pin (``liblorrax_ffi.so``, ``LORRAX_FFI_SO``) so sealed
#: bundles and site modules stay valid; the symmetry is in this table.
LEGS: Mapping[str, PlatformLeg] = MappingProxyType({
    "cpu": HOST,
    "CUDA": PlatformLeg(
        key="CUDA", vendor="cuda",
        so_name="liblorrax_ffi.so", so_env="LORRAX_FFI_SO",
        build_subdir="build", c_abi_suffix="",
        abi_symbol="lorrax_ffi_cuda_abi_version", accelerator=True,
        jax_platforms=("cuda", "gpu"),
        visibility_envs=("CUDA_VISIBLE_DEVICES",),
        # WSL2 has no /dev/nvidia*; its driver is the mapped libcuda.
        device_globs=("/dev/nvidia[0-9]*", "/dev/nvidiactl",
                      "/usr/lib/wsl/lib/libcuda.so.1"),
        jax_plugin_prefix="xla_cuda",
        # Claims 683 (autotune 0) and 4179 (LLVM module parallelism).
        xla_flag_defaults=(("--xla_gpu_autotune_level", "0"),
                           ("--xla_gpu_enable_llvm_module_compilation_parallelism", "true")),
        xla_allocator="cuda_async",
        xla_allocators=("default", "platform", "bfc", "cuda_async"),
        device_memory_api="cuda_driver",
        fft_plan_scratch="cufft",
        outside_pool_api="cuda_driver_mempool",
        memory_telemetry="nvidia-smi",
        collective_library="nccl", collective_net_plugin="libnccl-net.so",
        profiler=("nsys", "libToolsInjection", 0)),
    "ROCM": PlatformLeg(
        key="ROCM", vendor="rocm",
        so_name="liblorrax_ffi_rocm.so", so_env="LORRAX_FFI_ROCM_SO",
        build_subdir="build_rocm", c_abi_suffix="_rocm",
        abi_symbol="lorrax_ffi_rocm_abi_version", accelerator=True,
        jax_platforms=("rocm",),
        visibility_envs=("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"),
        device_globs=("/dev/kfd",),
        jax_plugin_prefix="xla_rocm",
        # Autotune 0: XLA compile -16 % (gnppm) / -12 % (exciton_bands) at P4,
        # GEMM steady state identical (d/zgemm n=1024..8192, 39.6 TF/s).  No
        # LLVM-module-parallelism row: XLA's AMDGPU compiler cannot link split
        # modules, so the flag is inert there (runtime lane 2026-10-14).
        # Synchronous collectives: an async all-reduce on XLA's collective
        # stream beside a collective of another communicator on the compute
        # stream deadlocks RCCL across ranks (GPUs at 100 %, every rank inside
        # the program).  Minimal repro hangs at iteration 1, 300/300 with this
        # flag; hsuite bisp_sc hung 3/3 and sp_export 1/1, PASS with it
        # (disteigh lane 2026-10-15, d76a2bee4; pool logs disteigh_repro_*,
        # disteigh_bisp3).  Part of the leg's collective policy beside
        # collective_library: it holds on one node and across nodes.
        xla_flag_defaults=(("--xla_gpu_autotune_level", "0"),
                           ("--xla_gpu_disable_async_collectives", "ALLCOLLECTIVES")),
        # The pool policy's arena: BFC (cuda_async does not exist on ROCm),
        # preallocated at runtime.pool_fraction (0.8952 of a 68.70 GB GCD,
        # 61.50 GB) instead of jaxlib's 0.75 (51.53 GB).  Outside the arena
        # after RCCL init, 512 MiB psum/all_to_all, a 3D FFT and a GEMM:
        # 0.99-1.08 GB per rank at P4, against the 7.20 GB left; hsuite walls
        # unchanged, planner budget 41.96 -> 50.08 GB (Runtime-LEGS lane 2026-10-15).
        xla_allocator="bfc",
        # cuda_async refused at client creation (MI250X, jax 0.9.1).
        xla_allocators=("default", "platform", "bfc"),
        device_memory_api="hip_runtime",
        # rocFFT compiles each new FFT length with hipRTC (MI250X: 1.9-7.6 s
        # per length per process, 0.02-0.04 s from its cache); cuFFT ships its
        # kernels, so CUDA has none.  One SQLite file per process: several
        # writers on one file on Lustre spin.  comgr's cache (~590 MB after one
        # hsuite) would otherwise grow in HOME.
        runtime_kernel_caches=(("ROCFFT_RTC_CACHE_PATH", "rocfft_kernel_cache", "process-file"),
                               ("AMD_COMGR_CACHE_DIR", "comgr", "shared-dir")),
        # XLA's ROCm FFT thunk takes the hipFFT plan workspace from the BFC
        # arena: on the seven c128 3D boxes probed the arena peak rose by
        # exactly hipfftMakePlanMany's workSize (23.3-131.1 MB, one copy of
        # the data; workSize 0 and no rise for 96^3 x 16).  An eighth case
        # (1D 1009 x 4096, workSize 65.5 kB) gave no reading: the peak did not
        # rise past an earlier one.  MI250X, Runtime-LEGS lane 2026-10-15.
        fft_plan_scratch="hipfft",
        outside_pool_api="hip_runtime_arena",
        # In process: amd-smi takes 1.0 s a sample and rocm-smi 0.45 s, and both
        # number all eight GCDs, not the one ROCR_VISIBLE_DEVICES leaves visible.
        memory_telemetry="hip_runtime",
        collective_library="rccl", collective_net_plugin="librccl-net.so",
        # rocprofv3 --kernel-trace deadlocks RCCL at P=4 (Screening lane, 2026-10-14).
        profiler=("rocprofv3", "librocprofiler-sdk-tool", 1)),
})

_BY_VENDOR = {leg.vendor: leg for leg in LEGS.values()}


def accelerator_legs() -> tuple:
    """The device legs, in :data:`LEGS` order."""
    return tuple(leg for leg in LEGS.values() if leg.accelerator)


def leg_for_vendor(vendor: str) -> Optional[PlatformLeg]:
    """The leg whose devices report ``vendor``, or ``None`` (no LORRAX leg)."""
    return _BY_VENDOR.get(vendor)


def leg_for_jax_platform(name: str) -> Optional[PlatformLeg]:
    """The device leg a ``JAX_PLATFORMS`` entry selects (``"cuda"``, ``"gpu"``,
    ``"rocm"``), or ``None`` (``"cpu"``, an unknown name).  First match in
    :data:`LEGS` order, the order JAX itself tries ``"gpu"`` in."""
    name = (name or "").strip().lower()
    return next((leg for leg in accelerator_legs() if name in leg.jax_platforms), None)


def _masked(leg: PlatformLeg, env: Mapping[str, str]) -> bool:
    return any(env.get(n) is not None and env.get(n).strip() == ""
               for n in leg.visibility_envs)


def leg_device_present(leg: PlatformLeg, *, environ: Mapping[str, str] | None = None,
                       exists: Callable[[str], bool] | None = None) -> bool:
    """A device of ``leg``'s vendor is visible to this process: a device node
    (or driver shim) exists and no visibility variable of the vendor is set
    empty.  Environment and filesystem only; never imports JAX.  ``exists``
    answers one glob pattern (tests pass a fake)."""
    env = os.environ if environ is None else environ
    if not leg.accelerator or _masked(leg, env):
        return False
    if exists is None:
        exists = _path_exists
    return any(exists(p) for p in leg.device_globs)


def _path_exists(pattern: str) -> bool:
    """A glob pattern matches, or a literal path exists."""
    if any(c in pattern for c in "*?["):
        return bool(glob.glob(pattern))
    return os.path.exists(pattern)


def process_accelerator_leg(*, environ: Mapping[str, str] | None = None,
                            exists: Callable[[str], bool] | None = None
                            ) -> Optional[PlatformLeg]:
    """The device leg this process can host, decided before any JAX backend
    exists, or ``None``.

    ``JAX_PLATFORMS`` first, as JAX picks its default backend: a first entry
    naming one leg selects that leg (and ``cpu`` selects none); then the leg
    must have a visible device (:func:`leg_device_present`).  With
    ``JAX_PLATFORMS`` unset, the first device leg with a visible device.
    """
    env = os.environ if environ is None else environ
    first = env.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if first:
        legs = [leg for leg in accelerator_legs() if first in leg.jax_platforms]
    else:
        legs = list(accelerator_legs())
    for leg in legs:
        if leg_device_present(leg, environ=env, exists=exists):
            return leg
    return None


def accelerator_leg_to_open_first(platform: str, opened, tried: set, *,
                                  environ: Mapping[str, str] | None = None,
                                  exists: Callable[[str], bool] | None = None
                                  ) -> Optional[str]:
    """The device-leg key a loader must open before it opens ``platform``, or ``None``.

    THE SONAME RULE (ffi-build.md, "Shared SONAMEs"): a device-built and a
    host-built SLATE/BLAS++ share SONAMEs, and whichever is mapped first wins
    for the whole process, so in a process that can host a device leg the
    device leg opens before the host leg.  In a CPU-platform process nothing
    opens first: there is no device handler to protect.

    One attempt per process per loader: ``tried`` is the loader's own set and
    is updated here.  ``opened`` is the loader's open-leg mapping.
    """
    if platform != HOST.key:
        return None
    leg = process_accelerator_leg(environ=environ, exists=exists)
    if leg is None or leg.key in opened or leg.key in tried:
        return None
    tried.add(leg.key)
    return leg.key
