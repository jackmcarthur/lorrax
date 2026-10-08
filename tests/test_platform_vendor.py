"""The platform key comes from the device vendor, never from the string ``gpu``.

CPU only, no jax needed for the first group (fake devices carry the strings a
device client reports); the router cells import ``ffi.fft`` and skip without
jax.  Run: ``PYTHONPATH=src:services/lxkit/src python3 tests/test_platform_vendor.py``.
"""
import os
import sys
import tempfile
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "services", "lxkit", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lxkit                                         # noqa: E402
import runtime                                       # noqa: E402  (os + subprocess only)


def _dev(platform, client_platform="", version="", kind=""):
    client = types.SimpleNamespace(platform=client_platform, platform_version=version)
    return types.SimpleNamespace(platform=platform, client=client, device_kind=kind)


def _mesh(dev):
    return types.SimpleNamespace(devices=types.SimpleNamespace(flat=[dev]))


#: (device, vendor, FFI platform key) for every vendor string a client reports.
CASES = [
    (_dev("cpu", "cpu", "cpu", "cpu"), "cpu", "cpu"),
    (_dev("gpu", "cuda", "PJRT C API\ncuda 13020", "NVIDIA A100-SXM4-80GB"), "cuda", "CUDA"),
    (_dev("gpu", "gpu", "", "NVIDIA H100 80GB HBM3"), "cuda", "CUDA"),
    (_dev("cuda", "", "", ""), "cuda", "CUDA"),
    (_dev("gpu", "rocm", "rocm 60300", "AMD Instinct MI250X"), "rocm", "rocm"),
    (_dev("gpu", "gpu", "", "gfx90a"), "rocm", "rocm"),
    (_dev("rocm", "", "", ""), "rocm", "rocm"),
    (_dev("gpu", "gpu", "", "Some chipset"), "gpu", "gpu"),       # 'chipset' is not 'hip'
    (_dev("tpu", "tpu", "", "TPU v4"), "tpu", "tpu"),
]


def test_device_vendor_reads_the_client_strings():
    for dev, vendor, _ in CASES:
        assert lxkit.device_vendor(dev) == vendor, (dev, lxkit.device_vendor(dev))


def test_mesh_ffi_platform_maps_vendor_to_library_key():
    for dev, _, key in CASES:
        assert lxkit.mesh_ffi_platform(_mesh(dev)) == key, dev


def test_gpu_string_alone_is_not_cuda():
    """The defect this replaces: ``gpu`` was mapped to CUDA, so a ROCm mesh loaded CUDA."""
    assert lxkit.mesh_ffi_platform(_mesh(_dev("gpu"))) != "CUDA"


def test_platform_from_env_per_vendor():
    saved = os.environ.get("JAX_PLATFORMS")
    try:
        for value, want in (("cuda,cpu", "CUDA"), ("gpu", "CUDA"), ("rocm,cpu", "rocm"),
                            ("cpu", "cpu"), ("", "CUDA")):
            os.environ["JAX_PLATFORMS"] = value
            assert lxkit.platform_from_env() == want, (value, lxkit.platform_from_env())
    finally:
        if saved is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = saved


def _gate(platforms, **kw):
    def probe(target, platform):
        raise AssertionError(f"probe called for {target} on {platform}")
    return lxkit.Gate(env="LORRAX_TEST_VENDOR_GATE", target="lorrax_test", platforms=platforms,
                      modes=("off", "on"), default="on", off_label="xla", probe=probe, **kw)


def test_rocm_mesh_never_probes_a_cuda_or_host_library():
    rocm = _mesh(CASES[4][0])
    for platforms in (("cpu",), ("CUDA",)):
        g = _gate(platforms, silent_platform_demote="test")
        assert not g.platform_ok(rocm)
        assert g.enforce(rocm, announce=False) is None
        assert g.resolve(rocm) is None
        try:
            g.require(rocm)
        except RuntimeError as exc:
            assert "rocm" in str(exc)
        else:
            raise AssertionError("require() served a rocm mesh")


def test_cuda_mesh_still_probes():
    seen = []
    g = lxkit.Gate(env="LORRAX_TEST_VENDOR_GATE2", target="t", platforms=("CUDA",),
                   modes=("off", "on"), default="on", off_label="xla",
                   probe=lambda t, p: (seen.append((t, p)) or (True, "ok")))
    assert g.require(_mesh(CASES[1][0]), announce=False) == "CUDA"
    assert seen == [("t", "CUDA")]


def test_ffi_gate_is_the_lxkit_gate():
    if "ffi" not in sys.modules:                 # login node: no package __init__ (it imports jax)
        stub = types.ModuleType("ffi")
        stub.__path__ = [os.path.join(_REPO, "src", "ffi")]
        sys.modules["ffi"] = stub
    import importlib
    gate = importlib.import_module("ffi.gate")
    assert issubclass(gate.Gate, lxkit.Gate)
    assert gate.announce_once is lxkit.announce_once
    assert gate.mesh_ffi_platform is lxkit.mesh_ffi_platform
    assert gate.Gate.__dataclass_fields__["probe"].default is gate._probe


def _with_env(**env):
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_gpu_is_present_sees_amd_and_honours_masks():
    import glob as _glob
    real_exists, real_glob = os.path.exists, _glob.glob
    nodes = {"/dev/kfd"}
    os.path.exists = lambda p: p in nodes
    _glob.glob = lambda pat: []
    saved = _with_env(CUDA_VISIBLE_DEVICES=None, ROCR_VISIBLE_DEVICES=None,
                      HIP_VISIBLE_DEVICES=None)
    try:
        assert runtime._gpu_is_present()                 # AMD node, nothing masked
        os.environ["ROCR_VISIBLE_DEVICES"] = ""
        assert not runtime._gpu_is_present()             # AMD masked
        os.environ.pop("ROCR_VISIBLE_DEVICES")
        nodes.clear()
        assert not runtime._gpu_is_present()             # no device node at all
        nodes.add("/dev/nvidiactl")
        assert runtime._gpu_is_present()                 # NVIDIA node
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        assert not runtime._gpu_is_present()             # NVIDIA masked
    finally:
        os.path.exists, _glob.glob = real_exists, real_glob
        _restore(saved)


def test_visible_device_count_per_vendor_variable():
    for env, want in (({"CUDA_VISIBLE_DEVICES": "2"}, 1),
                      ({"ROCR_VISIBLE_DEVICES": "0,1"}, 2),
                      ({"HIP_VISIBLE_DEVICES": "3,4,5"}, 3),
                      ({}, 0)):
        full = {"CUDA_VISIBLE_DEVICES": None, "ROCR_VISIBLE_DEVICES": None,
                "HIP_VISIBLE_DEVICES": None, **env}
        saved = _with_env(**full)
        try:
            assert runtime._visible_device_count() == want, (env, want)
        finally:
            _restore(saved)


def test_gpu_plugin_platform_from_the_installed_plugin():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "jax_plugins", "xla_rocm7"))
        sys.path.insert(0, root)
        try:
            assert runtime._gpu_plugin_platform() == "rocm"
        finally:
            sys.path.remove(root)
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "jax_plugins", "xla_cuda13"))
        saved_path = list(sys.path)
        sys.path[:] = [root]
        try:
            assert runtime._gpu_plugin_platform() == "cuda"
        finally:
            sys.path[:] = saved_path


def _main():
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            failed.append(name)
            print("FAIL %s\n     %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed" % (len(fns) - len(failed), len(fns), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
