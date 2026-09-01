"""Pure tests for rank-local nvidia-smi memory fallback."""

from types import SimpleNamespace


def test_nvidia_smi_memory_query_uses_this_ranks_visible_gpu(monkeypatch):
    from common import gpu_utils

    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        assert kwargs["timeout"] == 5
        return SimpleNamespace(returncode=0, stdout="2048\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(gpu_utils.subprocess, "run", run)
    assert gpu_utils.get_gpu_used_memory_bytes_nvidia_smi() == 2_000_000_000
    assert seen["argv"] == [
        "nvidia-smi", "--id=3", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
