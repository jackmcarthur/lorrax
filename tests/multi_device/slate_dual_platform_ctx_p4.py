"""P4: CPU and CUDA SLATE contexts of one shape bind in their own libraries."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import numpy as np
    from jax.sharding import Mesh
    from distrib_la import _slate, loader

    assert jax.process_count() == 4
    gpu_devices, cpu_devices = jax.devices("gpu"), jax.devices("cpu")
    assert len(gpu_devices) == len(cpu_devices) == 4, (
        len(gpu_devices), len(cpu_devices))
    gpu = Mesh(np.asarray(gpu_devices).reshape(2, 2), ("x", "y"))
    cpu = Mesh(np.asarray(cpu_devices).reshape(2, 2), ("x", "y"))
    host_handle = _slate.get_or_init_context(cpu)
    host_key = _slate.context_key(cpu)
    assert host_handle == _slate.get_or_init_context(cpu)
    assert host_key == _slate.context_key(cpu)
    host_subrow = _slate._get_or_init_subrow_context(cpu)
    assert host_subrow == _slate._get_or_init_subrow_context(cpu)
    assert len(_slate._CACHE) == len(_slate._SUBROW_CACHE) == 1

    cuda_lib = loader.get_lib("CUDA")
    if hasattr(cuda_lib, "lrx_slate_context_create"):
        cuda_handle = _slate.get_or_init_context(gpu)
        assert cuda_handle != host_handle
        assert _slate.context_key(gpu) != host_key
        assert _slate._get_or_init_subrow_context(gpu) != host_subrow
        assert _slate._subrow_context_key(gpu) != _slate._subrow_context_key(cpu)
        assert len(_slate._CACHE) == len(_slate._SUBROW_CACHE) == 2
        cuda_verdict = "separate context and key"
    else:
        from distrib_la.loader import LibraryUnusable
        try:
            _slate.get_or_init_context(gpu)
        except LibraryUnusable as exc:
            assert "SLATE context unavailable on CUDA" in str(exc), str(exc)
        else:
            raise AssertionError("CUDA reused a CPU context despite absent CUDA SLATE")
        try:
            _slate._get_or_init_subrow_context(gpu)
        except LibraryUnusable as exc:
            assert "SLATE subrow context unavailable on CUDA" in str(exc), str(exc)
        else:
            raise AssertionError("CUDA reused a CPU subrow context")
        assert len(_slate._CACHE) == len(_slate._SUBROW_CACHE) == 1
        cuda_verdict = "named capability refusal; no foreign pointer reuse"
    _slate._atexit_teardown()
    assert not _slate._CACHE and not _slate._SUBROW_CACHE and not _slate._KEYS

    if jax.process_index() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "PASS", "job": os.environ.get("SLURM_JOB_ID"),
            "step": os.environ.get("SLURM_STEP_ID"),
            "source_root": str(Path(__file__).resolve().parents[2]),
            "cpu_provider": loader.loaded_lib_path("cpu"),
            "cuda_provider": loader.loaded_lib_path("CUDA"),
            "scope": "P4 host SLATE world/subrow context reuse, CUDA "
                     "platform isolation and teardown",
            "cuda_verdict": cuda_verdict,
        }, indent=2) + "\n")
    print(f"rank {jax.process_index()}: dual-platform ctx PASS", flush=True)


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    RUNTIME = initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
