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
  points (AMD comgr's code-object cache, ``AMD_COMGR_CACHE_DIR``, default
  ``~/.cache/comgr``).  :mod:`common.jax_compile_cache` points each beside
  the persistent compile cache (Precursors lane, 2026-10-14);
* ``runtime_kernel_stores``: the vendor kernel caches that are keyed by their
  own content and so outlive the compile-cache namespace, like LORRAX's own
  kconv image store (rocFFT's hipRTC kernels: a read-only snapshot through
  ``ROCFFT_RTC_SYS_CACHE_PATH``, new kernels through ``ROCFFT_RTC_CACHE_PATH``),
  and ``blas_host_warmup``: the dtypes whose first GEMM loads the BLAS
  library's kernel tables on the host (rocBLAS/Tensile).
  :mod:`runtime.vendor_kernels` implements both (Cold-start lane, 2026-10-15);
* the vendor facts the memory model, the network policy and the profiler
  guard implement (owner ``runtime``; Runtime-LEGS lane, 2026-10-15):
  ``fft_plan_scratch`` (whose plan-workspace query prices the FFT scratch
  XLA's buffer assignment cannot see, :mod:`runtime.aot_memory`),
  ``outside_pool_api`` (the reader of bytes outside the XLA pool,
  :mod:`runtime.xla_memory`), the memory model's rows (:mod:`lxkit.memory`),
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
import math
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
    #: ``(writer_env, reader_env, name)``: a content-keyed vendor kernel store
    #: in ``lxkit.user_cache_dir(name)``; the library reads the snapshot named
    #: by ``reader_env`` and writes new kernels to one file per process named
    #: by ``writer_env``, which ``runtime.vendor_kernels`` folds in at exit.
    runtime_kernel_stores: tuple = ()
    #: dtypes of one 8x8 GEMM each, compiled at runtime start-up and run on a
    #: helper thread so the BLAS library's host initialisation overlaps the
    #: driver's imports (() = none).
    blas_host_warmup: tuple = ()
    #: plan-workspace query in runtime.aot_memory (``"cufft"`` | ``"hipfft"``);
    #: "" = the leg's FFTs take no plan scratch outside buffer assignment.
    fft_plan_scratch: str = ""
    #: bytes-outside-the-pool reader in runtime.xla_memory ("" = none).
    outside_pool_api: str = ""
    #: ``(env, value)`` pairs: process-environment defaults of the leg's
    #: runtime libraries, ``setdefault`` by ``runtime.set_default_env`` before
    #: the backend starts (a caller's export wins).  () = none.
    runtime_env_defaults: tuple = ()
    #: Bytes a process may hold outside the allocator at ``linalg = local`` (context,
    #: modules, XLA's communicators, FFT plans): the measured maximum with its margin
    #: (``lxkit.memory.resolve``; provenance on each leg).
    outside_allowance: float = 3.6e9
    #: The allocator's fragmentation reserve, a fraction of its limit, that the planning
    #: budget leaves free (``lxkit.memory.resolve``; provenance on each leg).
    memory_reserve: float = 0.0
    #: Devices of this leg in one node of the site's GPU partition (``lxkit.memory.sizing``).
    node_devices: int = 1
    #: The allocator high-water reader of ``lxkit.memory``: "pool" (CUDA mempool
    #: USED_MEM_HIGH, resettable), "bfc" (``peak_bytes_in_use``, first raise exact, else a
    #: polled lower bound) or "live" (live arrays).
    peak_reader: str = "live"
    #: ``((library, op), MeasuredRow)``: vendor workspace measured where no query exists.
    vendor_rows: tuple = ()
    #: ``(bytes, provenance)``: what the distributed linear-algebra providers open outside the
    #: allocator under ``linalg = distributed`` (contexts, communicators, workspace), with the
    #: 1.2 margin: a route that opens them carries it as its ``outside`` term.
    distributed_context: tuple = (3.6e9, "cuSOLVERMp/cuBLASMp contexts + NCCL: 3.0 GB measured "
                                         "at P4 (5.9 vs 2.9 outside) x 1.2")
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


@dataclass(frozen=True)
class MeasuredRow:
    """A measured vendor workspace: ``ratio * unit(n, P, itemsize)`` bytes per rank on the
    measured ``n`` and ``P`` ranges (inclusive), with provenance (``lxkit.memory.vendor``);
    ``unit`` None: the caller computes the unit (its library's own layout)."""
    ratio: float
    unit: Callable
    unit_name: str
    n: tuple = (1, float("inf"))
    p: tuple = (1, float("inf"))
    source: str = ""


HOST = PlatformLeg(
    key="cpu", vendor="cpu",
    so_name="liblorrax_ffi_host.so", so_env="LORRAX_FFI_HOST_SO",
    build_subdir="build_host", c_abi_suffix="_host",
    abi_symbol="lorrax_ffi_host_abi_version", accelerator=False,
    jax_platforms=("cpu",), memory_reserve=0.18567)

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
        # Claim 683 (autotune 0).  XLA removed
        # --xla_gpu_enable_llvm_module_compilation_parallelism before jaxlib 0.12
        # (claim 4179 measured it on 0.9.1) and aborts on the unknown flag.
        # Triton GEMM off: jaxlib 0.12 at autotune 0 lowers a real f64 dot to an
        # untuned Triton GEMM fusion (8192^2 dgemm 3.74 TF/s on A100-80GB); with
        # Triton GEMM off it is cuBLASLt, 19.3 TF/s, as at autotune 4 and on
        # jaxlib 0.9.1.  c128 dots were cuBLASLt either way (21.9 TF/s), and
        # perf_check's 36-program cold compile fell 5.6 -> 4.9 s (A100-80GB,
        # n=2 each; docs/architecture/compilation.md, "The runtime's XLA flags").
        xla_flag_defaults=(("--xla_gpu_autotune_level", "0"),
                           ("--xla_gpu_enable_triton_gemm", "false")),
        xla_allocator="cuda_async",
        xla_allocators=("default", "platform", "bfc", "cuda_async"),
        device_memory_api="cuda_driver",
        fft_plan_scratch="cufft",
        outside_pool_api="cuda_driver_mempool",
        collective_library="nccl", collective_net_plugin="libnccl-net.so",
        profiler=("nsys", "libToolsInjection", 0),
        # 3.0 GB outside the pool (context, modules, XLA NCCL communicators, cuFFT
        # plans; sandbox claim 3978) x 1.2; 1.66 GB measured after warm-up, before the
        # Mp contexts (MEMORY_MODEL_AUDIT §2).
        outside_allowance=3.6e9,
        # PROVISIONAL: the 2026-10-08 rule's in-use overshoot phi 0.19 x margin 1.2,
        # as a share of the limit (0.228 / 1.228), so the budget is the rule's until
        # the gate legs measure max(peak - priced) over passing stages.
        memory_reserve=0.18567,
        node_devices=4,                  # Perlmutter GPU node: 4 A100
        peak_reader="pool",
        vendor_rows=(
            (("cusolvermp", "solve_lu"), MeasuredRow(
                11.2, None, "local LU panels (n/Px + mb) nb", n=(4096, 30000),
                p=(4, 4), source="lrx_lu_workspace_bytes 9.64-11.16 panels (getrf, getrs "
                "nrhs = n, pivots), 2x2 A100, 2026-10-19, max taken; 4x4 unmeasured")),)),
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
        # preallocated at runtime.pool_fraction (0.8760 of a 68.70 GB GCD,
        # 60.18 GB, after outside_allowance below) instead of jaxlib's 0.75 (51.53 GB).
        # Outside the arena
        # after RCCL init, 512 MiB psum/all_to_all, a 3D FFT and a GEMM:
        # 0.99-1.08 GB per rank at P4, against the 7.20 GB left; hsuite walls
        # unchanged, planner budget 41.96 -> 50.08 GB (Runtime-LEGS lane 2026-10-15).
        xla_allocator="bfc",
        # cuda_async refused at client creation (MI250X, jax 0.9.1).
        xla_allocators=("default", "platform", "bfc"),
        device_memory_api="hip_runtime",
        # comgr's cache (~590 MB after one hsuite) would otherwise grow in HOME.
        runtime_kernel_caches=(("AMD_COMGR_CACHE_DIR", "comgr", "shared-dir"),),
        # rocFFT compiles each new FFT kernel with hipRTC and the pip wheel
        # ships no system kernel cache; cuFFT ships its kernels, so CUDA has
        # none.  hsuite kmeans..dipole builds 16 kernels per process, 10.0 s of
        # the cold kmeans wall (rho_from_wfns, py-spy, P4); from the store the
        # first FFT of a length costs 0.02-0.04 s.  The kernels are keyed by
        # name, arch, HIP version and generator checksum, so they outlive a
        # compile-cache namespace.  New kernels go to one SQLite file per
        # process (several writers on one file on Lustre spin) and are folded
        # into the read-only snapshot at exit (Cold-start lane, 2026-10-15).
        runtime_kernel_stores=(("ROCFFT_RTC_CACHE_PATH", "ROCFFT_RTC_SYS_CACHE_PATH", "rocfft"),),
        # rocBLAS loads its Tensile tables on a process's first GEMM: 2.84 s
        # for an 8x8 dgemm, then 0.055 s for the first zgemm, then 0.001 s
        # (MI250X, autotune 0, P1 probe).  In the hsuite kmeans timeline that
        # first GEMM comes ~3 s after the communicator warm-up, all of it
        # driver imports, so a helper thread hides it (Cold-start lane,
        # 2026-10-15).  cuBLAS ships no such table load: CUDA declares none.
        blas_host_warmup=("float64", "complex128"),
        # XLA's ROCm FFT thunk takes the hipFFT plan workspace from the BFC
        # arena: on the seven c128 3D boxes probed the arena peak rose by
        # exactly hipfftMakePlanMany's workSize (23.3-131.1 MB, one copy of
        # the data; workSize 0 and no rise for 96^3 x 16).  An eighth case
        # (1D 1009 x 4096, workSize 65.5 kB) gave no reading: the peak did not
        # rise past an earlier one.  MI250X, Runtime-LEGS lane 2026-10-15.
        fft_plan_scratch="hipfft",
        # HSA scratch reclaim off.  With ROCr's default every RCCL collective
        # costs ~200 us of DEVICE time on MI250X (1 KB ppermute / all-gather /
        # psum in an in-graph chain at P4, not hidden behind a 2 ms ZGEMM),
        # unchanged by NCCL_PROTO, channel counts, SDMA, kernarg placement,
        # launch mode, MSCCL, command buffers or async collectives; with
        # HSA_NO_SCRATCH_RECLAIM=1 it is 9-29 us, and 16 MB ppermute 29 -> 40
        # GB/s, intra-package all-gather 26 -> 59 GB/s bus bandwidth.  Fe V2 P4
        # (10 879 + 3 301 collectives): bank sample_dyson 4.86 -> 3.03 s, Sigma
        # tau 9.61 -> 9.16 s, eqp1 identical (comm lane 2026-10-15; pool logs
        # comm_var2, comm_fe_base, comm_fe_nsr).  CUDA declares none.
        runtime_env_defaults=(("HSA_NO_SCRATCH_RECLAIM", "1"),),
        # O of the memory rule.  With scratch reclaim off ROCr keeps each queue's
        # largest scratch: outside the BFC arena (hipMemGetInfo total - free -
        # pool_bytes, sampled every 0.25 s) the max over P4 ranks was 4.07 GB in
        # hsuite kmeans (2.71 GB with HSA_NO_SCRATCH_RECLAIM=0) and 3.30 GB on Fe
        # V2 (2.13 GB), resident from the first ISDF stage on; 3 GB (the CUDA
        # figure) no longer covers it (comm lane 2026-10-16, pool logs comm_mhs_*,
        # comm_mfe_*).  4.1 GB x 1.2; R = M - (4.92 + 3.6) GB: 60.18 of 68.70 GB.
        outside_allowance=4.92e9,
        # PROVISIONAL, as on CUDA: 0.228 / 1.228 of the BFC bytes_limit (48.99 of 60.18 GB).
        memory_reserve=0.18567,
        node_devices=8,                  # Frontier node: 4 MI250X, 8 GCDs
        peak_reader="bfc",
        vendor_rows=(
            (("elpa", "eigh"), MeasuredRow(
                1.0, lambda n, p, item: (24.0 if item == 16 else 16.0) * (n / math.isqrt(p)) ** 2
                + 2048.0 * n / math.isqrt(p), "(24|16) (N/p)^2 + 2048 N/p", p=(4, 16),
                source="ELPA 2026.02.001 allocator-hook high-water, MI250X P 4 and 16 "
                       "(ELPA_ROCM_REPORT.md, table pool)")),),
        outside_pool_api="hip_runtime_arena",
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


def leg_for_platform(platform: str) -> PlatformLeg:
    """The leg of a mesh platform (``distrib_la.mesh_platform``: "CUDA", "ROCM"), else the host."""
    return LEGS.get(str(platform), LEGS["cpu"])


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

    THE LOAD-ORDER RULE (distrib_la.loader): in a process that can host a
    device leg the device leg opens before the host leg (introduced for a
    third-party SONAME race, kept until test_so_acceptance.py check 5 is re-run
    without it).  In a CPU-platform process nothing
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
