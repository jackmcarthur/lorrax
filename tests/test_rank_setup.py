"""runtime.rank_setup / runtime.preflight: plain-srun pinning and the one-line refusals (CPU only)."""
from __future__ import annotations

from pathlib import Path

from runtime import preflight, rank_setup


def _step(**kw):
    env = {"SLURM_STEP_ID": "0", "SLURM_STEP_TASKS_PER_NODE": "4", "SLURM_NODEID": "0",
           "SLURM_LOCALID": "2"}
    env.update(kw)
    return env


def test_driver_process_detection():
    assert rank_setup.is_driver_process(["python", "-u", "-m", "gw.gw_jax", "-i", "x"])
    assert rank_setup.is_driver_process(["python", "-mbse.bse_jax"])
    assert not rank_setup.is_driver_process(["python", "-m", "pytest", "tests"])
    assert not rank_setup.is_driver_process(["python", "script.py"])


def test_pin_follows_select_gpu_rule():
    env = _step(CUDA_VISIBLE_DEVICES="0,1,2,3")
    assert "CUDA_VISIBLE_DEVICES=2" in rank_setup.pin_gpu(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    env = _step(SLURM_STEP_GPUS="4,5,6,7")
    rank_setup.pin_gpu(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
    env = _step(LORRAX_GPU_DEVICE="3,2,1,0")
    rank_setup.pin_gpu(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_leaves_pinned_single_task_and_unlaunched_alone():
    env = _step(CUDA_VISIBLE_DEVICES="1")                     # lorrax-run / --gpus-per-task
    assert rank_setup.pin_gpu(env) is None and env["CUDA_VISIBLE_DEVICES"] == "1"
    env = _step(ROCR_VISIBLE_DEVICES="5", SLURM_STEP_GPUS="0,1,2,3")   # --gpu-bind=closest
    assert rank_setup.pin_gpu(env) is None and "CUDA_VISIBLE_DEVICES" not in env
    env = _step(SLURM_STEP_TASKS_PER_NODE="1", CUDA_VISIBLE_DEVICES="0,1,2,3")
    assert rank_setup.pin_gpu(env) is None
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}                 # no Slurm step
    assert rank_setup.pin_gpu(env) is None


def test_tasks_per_node_spec():
    assert rank_setup._tasks_on_this_node({"SLURM_STEP_TASKS_PER_NODE": "4(x2),3",
                                           "SLURM_NODEID": "2"}) == 3
    assert rank_setup._tasks_on_this_node({"SLURM_STEP_NUM_TASKS": "8",
                                           "SLURM_STEP_NUM_NODES": "2"}) == 4


def test_threads_respect_caller_and_select_gpu():
    env = _step(OMP_NUM_THREADS="7")
    assert rank_setup.thread_defaults(env) is None and env["OMP_NUM_THREADS"] == "7"
    env = _step(LORRAX_RANK_THREADS="select_gpu: ...")
    assert rank_setup.thread_defaults(env) is None and "OMP_NUM_THREADS" not in env
    assert rank_setup.thread_defaults({}) is None


def _fake_tree(root: Path, abi: int) -> Path:
    (root / "src/runtime").mkdir(parents=True)
    (root / "src/runtime/__init__.py").write_text("")
    (root / "src/gw").mkdir()
    hdr = root / "src/ffi/cpp/common/lorrax_ffi_abi.h"
    hdr.parent.mkdir(parents=True)
    hdr.write_text(f"#define LORRAX_FFI_ABI_VERSION {abi}\n")
    so = root / "src/ffi/cpp/build_host/liblorrax_ffi_host.so"
    so.parent.mkdir(parents=True)
    so.write_bytes(b"\x7fELF")
    return root


def test_stale_overrides_name_the_unset(tmp_path):
    here = _fake_tree(tmp_path / "here", 11)
    old = _fake_tree(tmp_path / "old", 10)
    env = {"LORRAX_FFI_HOST_SO": str(old / "src/ffi/cpp/build_host/liblorrax_ffi_host.so"),
           "LORRAX_FFI_SO": str(tmp_path / "gone.so"),
           "PYTHONPATH": f"{here / 'src'}:{old / 'src'}:/opt/x",
           "LORRAX_ROOT": str(old)}
    refuse, warn = rank_setup.stale_overrides(env, tree=here, venv_tree=None)
    text = "\n".join(refuse)
    assert "ABI-10" in text and "`unset LORRAX_FFI_HOST_SO`" in text
    assert "`unset LORRAX_FFI_SO`" in text
    assert f"`export PYTHONPATH={here / 'src'}:/opt/x`" in text
    assert any("`unset LORRAX_ROOT`" in w for w in warn)
    ok = {"LORRAX_FFI_HOST_SO": str(here / "src/ffi/cpp/build_host/liblorrax_ffi_host.so"),
          "PYTHONPATH": str(here / "src")}
    assert rank_setup.stale_overrides(ok, tree=here, venv_tree=here) == ([], [])
    # one other tree on PYTHONPATH, imported by another tree's interpreter: refused unless
    # LORRAX_CHECKOUT says the overlay is deliberate (lx)
    over = {"PYTHONPATH": str(here / "src")}
    refuse, _ = rank_setup.stale_overrides(over, tree=here, venv_tree=old)
    assert refuse and "`unset PYTHONPATH`" in refuse[0]
    over["LORRAX_CHECKOUT"] = str(here)
    assert rank_setup.stale_overrides(over, tree=here, venv_tree=old) == ([], [])


def test_memory_rule_refuses_and_warns():
    refuse, _ = preflight.memory_findings((64, 20000, 2), 4, 40e9)
    assert refuse and "at least" in refuse[0]
    _, warn = preflight.memory_findings((8, 4000, 1), 4, 40e9)
    assert not warn
    _, warn = preflight.memory_findings((64, 4000, 2), 4, 40e9)
    assert warn and "expect OOM" in warn[0]


def test_stamp_refuses_only_an_owned_image(tmp_path):
    stamp = tmp_path / "verified_x.json"
    env = {"LORRAX_RUNTIME": "image", "LORRAX_PREFIX": str(tmp_path)}
    refuse, warn = preflight.stamp_findings(stamp, env)
    assert refuse and "lorrax-postinstall" in refuse[0] and not warn
    refuse, warn = preflight.stamp_findings(stamp, {})
    assert warn and not refuse
    stamp.write_text("{}")
    assert preflight.stamp_findings(stamp, env) == ([], [])


def test_missing_deck_refused(tmp_path):
    refuse, _, _ = preflight.deck_findings(str(tmp_path / "nope.in"), "gw.gw_jax")
    assert refuse and "not a readable file" in refuse[0]
