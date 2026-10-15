"""The platform legs are one table (``lxkit.legs.LEGS``) that every loader reads.

CPU only; the registry and the native-provider policy need no jax, the loader
cells import ``ffi.common.ffi_loader`` / ``distrib_la.loader`` and skip without it.
Run: ``PYTHONPATH=src:services/lxkit/src:services/distrib_la/src python3 -m pytest
tests/test_platform_legs.py``.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "services", "lxkit", "src"),
           os.path.join(_REPO, "services", "distrib_la", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lxkit                                                     # noqa: E402
from lxkit import legs                                           # noqa: E402
from lxkit import native_provider as native                      # noqa: E402


# ---- the table -----------------------------------------------------------------

def test_the_cuda_and_host_legs_keep_their_historical_facts():
    """Sealed bundles, site modules and LORRAX_FFI_SO pins depend on these values."""
    cuda, host = legs.LEGS["CUDA"], legs.LEGS["cpu"]
    assert (cuda.so_name, cuda.so_env, cuda.build_subdir, cuda.c_abi_suffix,
            cuda.abi_symbol) == ("liblorrax_ffi.so", "LORRAX_FFI_SO", "build", "",
                                 "lorrax_ffi_cuda_abi_version")
    assert (host.so_name, host.so_env, host.build_subdir, host.c_abi_suffix,
            host.abi_symbol) == ("liblorrax_ffi_host.so", "LORRAX_FFI_HOST_SO",
                                 "build_host", "_host", "lorrax_ffi_host_abi_version")


def test_the_runtime_vendor_facts_are_declared_per_leg():
    """The CUDA row carries the values the runtime hard-coded before they were leg facts;
    the host leg declares none (exact-zero scratch, no reader, no collective library)."""
    cuda, rocm, host = legs.LEGS["CUDA"], legs.LEGS["ROCM"], legs.LEGS["cpu"]
    fields = ("fft_plan_scratch", "outside_pool_api", "memory_telemetry",
              "collective_library", "collective_net_plugin")
    assert tuple(getattr(cuda, f) for f in fields) == (
        "cufft", "cuda_driver_mempool", "nvidia-smi", "nccl", "libnccl-net.so")
    assert tuple(getattr(rocm, f) for f in fields) == (
        "hipfft", "hip_runtime_arena", "hip_runtime", "rccl", "librccl-net.so")
    assert all(not getattr(host, f) for f in fields) and host.profiler == ()
    assert cuda.profiler[2] == 0 and rocm.profiler == ("rocprofv3", "librocprofiler-sdk-tool", 1)
    from runtime import aot_memory, xla_memory
    for leg in (cuda, rocm):
        assert leg.fft_plan_scratch in aot_memory._PLAN_APIS
        assert leg.outside_pool_api in xla_memory._OUTSIDE_POOL_READERS
        assert leg.memory_telemetry in xla_memory._TELEMETRY


def test_every_leg_fact_that_must_be_distinct_is_distinct():
    """Both libraries are dlopened RTLD_GLOBAL in one process: a shared file name,
    pin, C-ABI suffix or ABI symbol lets one leg answer for another."""
    rows = list(legs.LEGS.values())
    for field in ("key", "vendor", "so_name", "so_env", "build_subdir",
                  "c_abi_suffix", "abi_symbol"):
        values = [getattr(r, field) for r in rows]
        assert len(set(values)) == len(values), (field, values)
    assert all(k == r.key for k, r in legs.LEGS.items())


def test_vendor_map_is_derived_from_the_legs():
    assert dict(lxkit.FFI_PLATFORM_MAP) == {"cpu": "cpu", "cuda": "CUDA", "rocm": "ROCM"}
    assert legs.leg_for_vendor("rocm") is legs.LEGS["ROCM"]
    assert legs.leg_for_vendor("tpu") is None


# ---- hosting a device leg, before any backend exists ------------------------------

def _fs(*present):
    return lambda pattern: pattern in present


@pytest.mark.parametrize("env, nodes, want", [
    ({}, ("/dev/nvidiactl",), "CUDA"),
    ({}, ("/dev/kfd",), "ROCM"),
    ({}, (), None),
    ({"JAX_PLATFORMS": "cpu"}, ("/dev/nvidiactl", "/dev/kfd"), None),
    ({"JAX_PLATFORMS": "cuda,cpu"}, ("/dev/nvidiactl",), "CUDA"),
    ({"JAX_PLATFORMS": "gpu"}, ("/dev/nvidiactl",), "CUDA"),
    ({"JAX_PLATFORMS": "rocm,cpu"}, ("/dev/kfd",), "ROCM"),
    ({"JAX_PLATFORMS": "rocm"}, ("/dev/nvidiactl",), None),        # asked for ROCm, none here
    ({"CUDA_VISIBLE_DEVICES": ""}, ("/dev/nvidiactl",), None),     # masked
    ({"ROCR_VISIBLE_DEVICES": ""}, ("/dev/kfd",), None),
    ({"HIP_VISIBLE_DEVICES": ""}, ("/dev/kfd",), None),
    ({"CUDA_VISIBLE_DEVICES": "0"}, ("/dev/nvidiactl",), "CUDA"),
    ({}, ("/usr/lib/wsl/lib/libcuda.so.1",), "CUDA"),               # WSL2
])
def test_process_accelerator_leg(env, nodes, want):
    leg = legs.process_accelerator_leg(environ=env, exists=_fs(*nodes))
    assert (leg.key if leg else None) == want


def test_the_cuda_answer_is_the_one_the_loaders_used_to_give():
    """The removed ``process_can_use_cuda``: JAX_PLATFORMS first entry in
    {unset, cuda, gpu}, CUDA_VISIBLE_DEVICES not empty, an NVIDIA node."""
    def old(env, nvidia):
        first = env.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
        if first and first not in ("cuda", "gpu"):
            return False
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        return not (cvd is not None and cvd.strip() == "") and nvidia
    for env in ({}, {"JAX_PLATFORMS": "cuda"}, {"JAX_PLATFORMS": "gpu,cpu"},
                {"JAX_PLATFORMS": "cpu"}, {"CUDA_VISIBLE_DEVICES": ""},
                {"CUDA_VISIBLE_DEVICES": "1"}):
        for nvidia in (True, False):
            nodes = ("/dev/nvidiactl",) if nvidia else ()
            leg = legs.process_accelerator_leg(environ=env, exists=_fs(*nodes))
            assert ((leg is not None and leg.key == "CUDA") == old(env, nvidia)), (env, nvidia)


def test_the_device_leg_opens_before_the_host_leg_once():
    tried = set()
    kfd = dict(environ={"JAX_PLATFORMS": "rocm,cpu"}, exists=_fs("/dev/kfd"))
    assert legs.accelerator_leg_to_open_first("ROCM", {}, tried, **kfd) is None   # not the host leg
    assert legs.accelerator_leg_to_open_first("cpu", {}, tried, **kfd) == "ROCM"
    assert legs.accelerator_leg_to_open_first("cpu", {}, tried, **kfd) is None    # one attempt
    assert legs.accelerator_leg_to_open_first("cpu", {"CUDA": 1}, set(), environ={},
                                              exists=_fs("/dev/nvidiactl")) is None  # already open
    assert legs.accelerator_leg_to_open_first("cpu", {}, set(), environ={"JAX_PLATFORMS": "cpu"},
                                              exists=_fs("/dev/kfd")) is None    # CPU process


# ---- native-provider policy over the legs ---------------------------------------

@pytest.mark.parametrize("names, ok", [
    ({"CUDA", "cpu"}, True), ({"ROCM", "cpu"}, True), ({"CUDA", "ROCM"}, False),
    ({"CUDA", "ROCM", "cpu"}, False), ({"cpu"}, False), ({"CUDA"}, False),
    ({"tpu", "cpu"}, False),
])
def test_a_bundle_binds_the_host_leg_and_one_device_leg(names, ok):
    assert native._is_leg_pair(names) is ok


@pytest.mark.parametrize("name", [
    "libamdhip64.so.6", "libhsa-runtime64.so.1", "libamd_comgr.so.3", "libhiprtc.so.6",
    "libhipblas.so.2", "librocblas.so.4", "libhipfft.so.0", "librocfft.so.0",
    "librocsolver.so.0", "librccl.so.1", "libamd_smi.so.26",
    "libcudart.so.13", "libnccl.so.2",
])
def test_device_runtime_libraries_are_never_bundled(name):
    with pytest.raises(OSError, match="runtime-owned"):
        native.refuse_private_library(name)


def test_engine_private_libraries_still_bundle():
    native.refuse_private_library("libslate.so.2")


# ---- both loaders read the one table ----------------------------------------------

def _loaders():
    pytest.importorskip("jax")
    from ffi.common import ffi_loader
    from distrib_la import loader as dl
    return ffi_loader, dl


def test_both_loaders_derive_the_same_leg_specs():
    ffi_loader, dl = _loaders()
    fields = ("so_name", "env", "build_subdir")
    for loader in (ffi_loader, dl):
        assert set(loader._PLATFORMS) == set(legs.LEGS)
        for key, leg in legs.LEGS.items():
            assert (loader._PLATFORMS[key]["so_name"], loader._PLATFORMS[key]["env"],
                    loader._PLATFORMS[key]["build_subdir"]) == (
                        leg.so_name, leg.so_env, leg.build_subdir)
        assert loader._ABI_SYMBOLS == {k: l.abi_symbol for k, l in legs.LEGS.items()}
        assert loader._C_ABI_SUFFIX == {k: l.c_abi_suffix for k, l in legs.LEGS.items()}
    assert ({k: tuple(v[f] for f in fields) for k, v in ffi_loader._PLATFORMS.items()}
            == {k: tuple(v[f] for f in fields) for k, v in dl._PLATFORMS.items()})


def test_the_rocm_leg_is_declared_and_refused_by_its_file_name(monkeypatch):
    ffi_loader, dl = _loaders()
    monkeypatch.delenv("LORRAX_FFI_ROCM_SO", raising=False)
    with pytest.raises(ffi_loader.FfiLibraryNotBuilt, match="liblorrax_ffi_rocm.so"):
        ffi_loader.get_lib("ROCM")
    with pytest.raises(dl.LibraryNotBuilt, match="liblorrax_ffi_rocm.so"):
        dl.get_lib("ROCM")
    ok, reason = ffi_loader.probe_target("lorrax_phdf5_write", "ROCM")
    assert not ok and "ROCM" in reason


def test_the_rocm_leg_carries_exactly_its_ported_families():
    """The ROCM table is the CUDA one's shared families (the device legs compile the
    same sources), restricted to the families in CMakeLists.txt's ROCM list, plus the
    leg's own families (kdft: the CUDA leg serves those operations with mathdx)."""
    ffi_loader, dl = _loaders()
    rocm = ffi_loader._PLATFORMS["ROCM"]["targets"]
    own = ffi_loader._ROCM_OWN_TARGET_SYMBOLS
    shared = {t: s for t, s in rocm.items() if t not in own}
    assert shared and all(t.startswith("lorrax_phdf5_") for t in shared)
    assert all(ffi_loader._CUDA_TARGET_SYMBOLS[t] == s for t, s in shared.items())
    assert {t for t in ffi_loader._CUDA_TARGET_SYMBOLS if t.startswith("lorrax_phdf5_")} == set(shared)
    assert own and all(t.startswith("lorrax_kdft_") for t in own)
    assert not set(own) & set(ffi_loader._CUDA_TARGET_SYMBOLS)
    assert all(rocm[t] == s for t, s in own.items())
    assert dl._PLATFORMS["ROCM"]["targets"] == {}          # no distributed linalg yet
    cmake = open(os.path.join(_REPO, "src", "ffi", "cpp", "CMakeLists.txt")).read()
    block = cmake[cmake.index('STREQUAL "rocm"'):]
    block = block[:block.index("add_library(")]
    assert "phdf5/read_ffi.cc" in block and "cusolvermp/" not in block and "cufft/" not in block
    assert "kdft/kdft_klead_ffi.cc" in block and "common/rtc_build.cc" in block


def test_the_family_table_matches_the_loaders_routers_and_build():
    """ffi.common.families is the one per-leg declaration: every registered target belongs to
    a family built on its leg, every built family registers a target there, the ROCm k-leading
    router serves kdft targets only, and the ROCM CMake block compiles each ROCM-built family."""
    from ffi.common import families as fam
    ffi_loader, dl = _loaders()
    for leg in legs.LEGS:
        targets = set(ffi_loader._PLATFORMS[leg]["targets"]) | set(dl._PLATFORMS[leg]["targets"])
        for target in targets:
            name = fam.family_of(target)
            assert name is not None, target
            assert fam.state(name, leg)[0] == "built", (target, leg)
        for name, row in fam.FAMILIES.items():
            assert set(row.legs) == set(legs.LEGS) and all(s in fam.STATES for s, _ in row.legs.values())
            if row.legs[leg][0] == "built":
                assert any(t.startswith(row.prefixes) for t in targets), (name, leg)
            if row.legs[leg][0] in ("elsewhere", "refused", "host"):
                assert row.legs[leg][1], (name, leg)
    import ffi.fft as fft
    assert {fam.family_of(t) for t in fft._ROCM_KLEAD_TARGETS.values()} == {"kdft"}
    cmake = open(os.path.join(_REPO, "src", "ffi", "cpp", "CMakeLists.txt")).read()
    block = cmake[cmake.index('STREQUAL "rocm"'):]
    block = block[:block.index("add_library(")]
    for name, row in fam.FAMILIES.items():
        assert (f"{row.source}/" in block) == (row.legs["ROCM"][0] == "built"), name
