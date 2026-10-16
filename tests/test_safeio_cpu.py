"""I/O and environment safety gates: cgroup host memory, the SlabIO platform
key, the jax._src signature gate, the shared-directory nonce, the demotion
record and the compute-capability refusal.  CPU only."""
import os
import socket
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest


# ---- host memory under cgroups -------------------------------------------------

def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


_MEMINFO = "MemTotal:       263000000 kB\nMemAvailable:   250000000 kB\n"


def test_cgroup_v2_caps_through_ancestors(tmp_path):
    from common.gpu_utils import _meminfo_gb
    _tree(tmp_path, {
        "proc/meminfo": _MEMINFO,
        "proc/self/cgroup": "0::/job_1/step_0/task_0\n",
        "cg/job_1/memory.max": "100000000000\n",          # the job's limit
        "cg/job_1/memory.current": "30000000000\n",
        "cg/job_1/memory.stat": "anon 1\ninactive_file 10000000000\n",
        "cg/job_1/step_0/memory.max": "max\n",
        "cg/job_1/step_0/memory.current": "29000000000\n",
        "cg/job_1/step_0/task_0/memory.max": "max\n",
        "cg/job_1/step_0/task_0/memory.current": "1\n",
    })
    kw = dict(proc=str(tmp_path / "proc"), cgroup_root=str(tmp_path / "cg"))
    assert _meminfo_gb("MemTotal", **kw) == pytest.approx(100.0)
    # limit - (usage - reclaimable page cache) = 100 - (30 - 10)
    assert _meminfo_gb("MemAvailable", **kw) == pytest.approx(80.0)


def test_cgroup_v1_caps(tmp_path):
    from common.gpu_utils import _meminfo_gb
    _tree(tmp_path, {
        "proc/meminfo": _MEMINFO,
        "proc/self/cgroup": "12:cpu,cpuacct:/slurm/job_2\n5:memory:/slurm/job_2\n"
                            "1:name=systemd:/x\n",
        "cg/memory/memory.limit_in_bytes": "9223372036854771712\n",
        "cg/memory/memory.usage_in_bytes": "5\n",
        "cg/memory/slurm/job_2/memory.limit_in_bytes": "50000000000\n",
        "cg/memory/slurm/job_2/memory.usage_in_bytes": "45000000000\n",
        "cg/memory/slurm/job_2/memory.stat": "total_inactive_file 5000000000\n",
    })
    kw = dict(proc=str(tmp_path / "proc"), cgroup_root=str(tmp_path / "cg"))
    assert _meminfo_gb("MemTotal", **kw) == pytest.approx(50.0)
    assert _meminfo_gb("MemAvailable", **kw) == pytest.approx(10.0)


def test_unlimited_or_missing_cgroup_keeps_meminfo(tmp_path):
    from common.gpu_utils import _meminfo_gb
    _tree(tmp_path, {"proc/meminfo": _MEMINFO, "proc/self/cgroup": "0::/\n",
                     "cg/memory.max": "max\n", "cg/memory.current": "7\n"})
    kw = dict(proc=str(tmp_path / "proc"), cgroup_root=str(tmp_path / "cg"))
    assert _meminfo_gb("MemTotal", **kw) == pytest.approx(263000000 * 1024 / 1e9)
    assert _meminfo_gb("MemAvailable", **kw) == pytest.approx(250000000 * 1024 / 1e9)
    assert _meminfo_gb("MemTotal", proc=str(tmp_path / "absent")) is None


# ---- the SlabIO platform key ---------------------------------------------------

def _mesh_of(platform, client_platform, kind):
    dev = SimpleNamespace(platform=platform, device_kind=kind,
                          client=SimpleNamespace(platform=client_platform,
                                                 platform_version=""))
    return SimpleNamespace(devices=np.array([dev], dtype=object))


@pytest.mark.parametrize("args, want", [
    (("gpu", "cuda", "NVIDIA A100-SXM4-80GB"), "CUDA"),
    (("gpu", "rocm", "AMD Instinct MI250X"), "ROCM"),
    (("cpu", "cpu", "cpu"), "cpu"),
])
def test_slab_io_platform_follows_the_vendor(args, want):
    from ffi.io import _platform_for_mesh
    assert _platform_for_mesh(_mesh_of(*args)) == want


def test_slab_io_refuses_a_vendor_without_a_leg():
    from ffi.common.ffi_loader import get_lib
    with pytest.raises(ValueError, match="unknown platform 'tpu'"):
        get_lib("tpu")


def test_slab_io_refuses_a_rocm_leg_without_its_library(monkeypatch):
    """ROCM is a declared leg: a missing library is refused by naming its .so."""
    from ffi.common.ffi_loader import FfiLibraryNotBuilt, get_lib
    monkeypatch.delenv("LORRAX_FFI_ROCM_SO", raising=False)
    with pytest.raises(FfiLibraryNotBuilt, match="liblorrax_ffi_rocm.so"):
        get_lib("ROCM")


# ---- the jax._src signature gate -----------------------------------------------

def _rename_module_param(monkeypatch):
    from jax._src import compiler
    orig = compiler.backend_compile_and_load

    def renamed(backend, hlo_module, executable_devices, options, host_callbacks):
        return orig(backend, hlo_module, executable_devices, options, host_callbacks)
    monkeypatch.setattr(compiler, "backend_compile_and_load", renamed)


def test_the_gate_reads_names_not_counts(monkeypatch):
    from runtime import jax_support
    assert jax_support.check_private_params() == []
    _rename_module_param(monkeypatch)
    (arms, why), = jax_support.check_private_params()
    assert arms == "agreement" and "hlo_module" in why


def test_a_renamed_agreement_parameter_refuses_at_p2(monkeypatch):
    import jax
    from runtime import jax_support
    _rename_module_param(monkeypatch)
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    with pytest.raises(jax_support.JaxSupportError, match="backend_compile_and_load"):
        jax_support.enforce()


def test_a_renamed_agreement_parameter_warns_at_p1(monkeypatch, capsys):
    import jax
    from runtime import jax_support
    _rename_module_param(monkeypatch)
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    jax_support.enforce()
    assert "agreement patch will not install" in capsys.readouterr().err


def test_a_changed_cache_parameter_refuses_at_p1(monkeypatch):
    import jax
    from jax._src import compilation_cache
    from runtime import jax_support
    monkeypatch.setattr(compilation_cache, "get_executable_and_time",
                        lambda cache_key, compile_options, backend: None)
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    with pytest.raises(jax_support.JaxSupportError, match="get_executable_and_time"):
        jax_support.enforce()


def test_the_fit_gate_install_refuses_at_p2(monkeypatch):
    from jax._src.interpreters import pxla
    from common import jax_compile_cache as jcc

    class Moved:
        def __init__(self, name, xla_executable, backend):
            pass
    monkeypatch.setattr(pxla, "ExecuteReplicated", Moved)
    monkeypatch.setattr(pxla, "_lorrax_device_fit_gate_installed", False, raising=False)
    monkeypatch.setattr(jcc._STATE, "n_proc", 2)
    with pytest.raises(jcc._JaxSurfaceUnsupported, match="ExecuteReplicated"):
        jcc._install_device_fit_gate()
    monkeypatch.setattr(jcc._STATE, "n_proc", 1)
    jcc._install_device_fit_gate()                    # a notice, not a refusal
    assert Moved.__init__.__name__ == "__init__"


# ---- the route-demotion record -------------------------------------------------

def test_a_demotion_rides_the_receipt_and_the_stats(monkeypatch):
    from common import jax_compile_cache as jcc
    monkeypatch.setattr(jcc._STATE, "demotions", {})
    jcc.record_demotion("kconv", "nvidia-mathdx -> XLA: probe failed")
    line = jcc.compile_receipt("test", emit=lambda s: None)
    assert line.endswith(", demoted kconv")
    assert jcc.compile_cache_stats()["demotions"] == {
        "kconv": "nvidia-mathdx -> XLA: probe failed"}


# ---- compute capability --------------------------------------------------------

def test_a_device_below_sm80_refuses(monkeypatch):
    import jax
    from ffi import fft
    old = SimpleNamespace(compute_capability="7.0", device_kind="Tesla V100")
    monkeypatch.setattr(jax, "local_devices", lambda: [old])
    with pytest.raises(RuntimeError, match="GATE cuda-sm80"):
        fft._require_sm80()
    new = SimpleNamespace(compute_capability="8.0", device_kind="NVIDIA A100")
    monkeypatch.setattr(jax, "local_devices", lambda: [new])
    fft._require_sm80()


@pytest.mark.skipif(not os.environ.get("LORRAX_FFI_HOST_SO"),
                    reason="needs the host FFI library (lx run)")
def test_the_host_engines_run_at_startup():
    import jax
    from jax.sharding import Mesh
    from ffi import fft
    fft._probe_host_engines(Mesh(np.array(jax.devices("cpu")[:1]), ("x",)))


# ---- the shared-directory nonce, two real processes ----------------------------

_NONCE_CHILD = textwrap.dedent("""
    import os, sys
    import jax
    rank, port, cwd = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=2, process_id=rank)
    from file_io._slab_io_ffi import _assert_shared_directory
    os.chdir(cwd)   # one path string, per-rank storage: a node-local directory
    try:
        _assert_shared_directory("run/out.h5")
        print("SHARED-OK")
    except Exception as exc:
        print("REFUSED", exc)
""")


def _run_pair(tmp_path, dir0, dir1):
    script = tmp_path / "child.py"
    script.write_text(_NONCE_CHILD)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    procs = [subprocess.Popen([sys.executable, str(script), str(r), port, str(d)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env)
             for r, d in ((0, dir0), (1, dir1))]
    return [p.communicate(timeout=240)[0] for p in procs]


def test_one_shared_directory_passes_and_leaves_nothing(tmp_path):
    (tmp_path / "node" / "run").mkdir(parents=True)
    outs = _run_pair(tmp_path, tmp_path / "node", tmp_path / "node")
    assert all("SHARED-OK" in out for out in outs), outs
    assert os.listdir(tmp_path / "node" / "run") == []


def test_per_rank_directories_refuse_on_every_rank(tmp_path):
    a, b = tmp_path / "node0", tmp_path / "node1"
    (a / "run").mkdir(parents=True)
    (b / "run").mkdir(parents=True)
    outs = _run_pair(tmp_path, a, b)
    for out in outs:
        assert "REFUSED GATE io_global_commit" in out, outs
        assert "failing rank=1" in out and "not one filesystem" in out, outs
    assert os.listdir(a / "run") == [] and os.listdir(b / "run") == []
