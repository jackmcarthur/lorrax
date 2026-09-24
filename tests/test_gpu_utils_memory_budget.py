"""Device budget when the XLA client reports no bytes_limit (cuda_async).

Pure: jax, nvidia-smi and the live-array census are stubbed, so this runs
anywhere. GB is 1e9 bytes throughout.
"""
import sys
from types import SimpleNamespace

import pytest

from common import gpu_utils

TOTAL = 81920 * 2**20          # A100-SXM4-80GB, nvidia-smi memory.total


@pytest.fixture
def gpu(monkeypatch):
    """A GPU process whose client has no arena stats and 5 GB of live arrays."""
    smi = {'memory.total': TOTAL, 'memory.free': 3 * 2**30}
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(
        default_backend=lambda: 'gpu', device_count=lambda: 4))
    monkeypatch.setattr(gpu_utils, '_get_jax_gpu_memory_bytes',
                        lambda: (None, None, None))
    monkeypatch.setattr(gpu_utils, '_query_nvidia_smi_memory', smi.get)
    monkeypatch.setattr(gpu_utils, '_live_array_bytes', lambda: 5 * 10**9)
    for name in ('XLA_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_MEM_FRACTION'):
        monkeypatch.delenv(name, raising=False)
    return smi


@pytest.mark.parametrize('env,fraction', [
    ({}, 0.75),
    ({'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.85'}, 0.85),
    ({'XLA_CLIENT_MEM_FRACTION': '0.6'}, 0.6),
])
def test_missing_bytes_limit_uses_mem_fraction_of_total(gpu, monkeypatch, env, fraction):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    limit = int(fraction * TOTAL)
    assert gpu_utils.get_device_memory_gb() == pytest.approx(0.9 * limit / 1e9)
    info = gpu_utils.get_device_memory_info()
    assert info['total_gb'] == pytest.approx(limit / 1e9)
    assert info['available_gb'] == pytest.approx((limit - 5e9) / 1e9)
    assert info['budget_gb'] == pytest.approx(0.9 * (limit - 5e9) / 1e9)
    assert f'{fraction:g} x nvidia-smi memory.total' in info['source']


def test_budget_does_not_follow_nvidia_smi_free(gpu):
    # RED TWIN of the old fallback: 0.9 x memory.free, read in GiB and
    # labelled GB. The async pool keeps freed blocks reserved, so free memory
    # tracks the pool's high-water mark rather than what the next stage can use.
    before = gpu_utils.get_device_memory_gb(), gpu_utils.get_device_memory_info()
    gpu['memory.free'] = 70 * 2**30
    after = gpu_utils.get_device_memory_gb(), gpu_utils.get_device_memory_info()
    assert before[0] == after[0] and before[1] == after[1]


def test_arena_limit_still_wins(gpu, monkeypatch):
    monkeypatch.setattr(gpu_utils, '_get_jax_gpu_memory_bytes',
                        lambda: (60e9, 10e9, 50e9))
    assert gpu_utils.get_device_memory_gb() == pytest.approx(54.0)
    assert gpu_utils.get_device_memory_info()['budget_gb'] == pytest.approx(45.0)


def test_nvidia_smi_free_is_reported_in_gb(monkeypatch):
    monkeypatch.setattr(gpu_utils.subprocess, 'run', lambda argv, **kw:
                        SimpleNamespace(returncode=0, stdout='2048\n'))
    assert gpu_utils.get_gpu_memory_nvidia_smi() == 2048 * 2**20 / 1e9
