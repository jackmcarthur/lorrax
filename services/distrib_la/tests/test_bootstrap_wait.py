"""Bootstrap arrival skew is a heartbeat, never a library deadline."""
import numpy as np
import pytest

from distrib_la import _collectives as C


def test_bootstrap_retries_only_client_wait_timeouts(capsys):
    from jax.errors import JaxRuntimeError

    class Client:
        calls = 0

        def blocking_key_value_get(self, key, timeout_ms):
            assert key == "test/nccl"
            assert timeout_ms == 60_000
            self.calls += 1
            if self.calls <= 2:
                raise JaxRuntimeError("DEADLINE_EXCEEDED: KV wait expired")
            return "00ff80"

    client = Client()
    assert C._wait_for_bootstrap(client, "test/nccl", 3) == "00ff80"
    assert client.calls == 3
    text = capsys.readouterr().err
    assert text.count("rank=3 still waiting for rank 0") == 2
    assert "NCCL ID key=test/nccl" in text
    assert "s elapsed" in text


@pytest.mark.parametrize("message", [
    "UNAVAILABLE: peer disconnected",
    "INTERNAL: coordinator timeout during shutdown",
])
def test_bootstrap_propagates_other_client_failures(message):
    from jax.errors import JaxRuntimeError

    error = JaxRuntimeError(message)

    class Client:
        def blocking_key_value_get(self, key, timeout_ms):
            raise error

    with pytest.raises(JaxRuntimeError) as caught:
        C._wait_for_bootstrap(Client(), "test/nccl", 1)
    assert caught.value is error


def test_broadcast_preserves_opaque_bytes_after_wait(monkeypatch):
    import jax
    from jax._src.distributed import global_state

    payload = np.arange(256, dtype=np.uint8)
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(jax, "process_index", lambda: 2)
    client = object()
    monkeypatch.setattr(global_state, "client", client)

    def wait(got_client, key, rank):
        assert got_client is client and key == "test/nccl" and rank == 2
        return payload.tobytes().hex()

    monkeypatch.setattr(C, "_wait_for_bootstrap", wait)
    got = C.broadcast_bytes(np.zeros_like(payload), key="test/nccl")
    assert got.dtype == np.uint8
    assert got.tobytes() == payload.tobytes()
