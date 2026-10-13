"""The persistent-cache key of an XLA:CPU program names no process-local device.

At P > 1 rank 0 is the only writer, so a peer hits only if it computes rank
0's key. A peer's copy of a program differs from rank 0's only in its own
device: here device 3 of a four-device host stands in for a peer's device.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import cache_key, compiler, xla_bridge
from jax._src.lib import xla_client

from common import jax_compile_cache as jcc


@pytest.fixture
def patched(monkeypatch):
    """Install the CPU key patch for one test; monkeypatch restores JAX's hashes."""
    for name in ("_hash_serialized_compile_options", "_hash_accelerator_config"):
        monkeypatch.setattr(cache_key, name, getattr(cache_key, name))
    monkeypatch.setattr(cache_key, "_lorrax_cpu_key_installed", False, raising=False)
    return jcc._install_rank_invariant_cpu_key


def _key(device):
    module = jax.jit(lambda x: 2 * x + 1).lower(jnp.ones(8)).compiler_ir("stablehlo")
    backend = xla_bridge.get_backend("cpu")
    options = compiler.get_compile_options(
        1, 1, device_assignment=np.array([[device]]), backend=backend)
    return cache_key.get(module, np.array([device]), options, backend)


def test_cpu_key_is_the_same_on_every_device(patched):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=4")
    assert _key(devices[0]) != _key(devices[-1])
    patched()
    assert _key(devices[0]) == _key(devices[-1])


def test_cpu_topology_keeps_the_target_machine_only():
    target = b"*\x07x86_64!"
    for records in (b"", b"\n\x00", b"\n\x02\x10\x01", b"\n\x00\n\x03\x08\x80\x10",
                    b"\n\x81\x01" + b"d" * 129):
        assert jcc._cpu_target_machine(records + target) == target
    for malformed in (b"\n\x81", b"\n\x05ab", b"\n\x02\x10\x01\n\xff\xff"):
        assert jcc._cpu_target_machine(malformed) == malformed
    blob = xla_client.get_topology_for_devices(jax.devices("cpu")).serialize()
    machine = jcc._cpu_target_machine(blob)
    assert 0 < len(machine) < len(blob) and blob.endswith(machine)


def test_gpu_hash_is_jax_own(patched, monkeypatch):
    seen = []
    monkeypatch.setattr(cache_key, "_hash_accelerator_config",
                        lambda hash_obj, devices: seen.append(devices))

    class Gpu:
        platform = "gpu"

    patched()
    devices = np.array([Gpu()], dtype=object)
    cache_key._hash_accelerator_config(None, devices)
    assert seen == [devices]
