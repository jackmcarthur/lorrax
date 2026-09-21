import numpy as np


def test_host_reduction_is_bounded_and_cleans_keys(monkeypatch):
    from ffi.common import broadcast
    from jax._src.distributed import global_state

    class Client:
        def __init__(self):
            self.values = {
                "gate/value/1": b"b",
                "gate/ack/1": b"1",
            }
            self.deleted = []

        def key_value_set_bytes(self, key, value):
            self.values[key] = value

        def blocking_key_value_get_bytes(self, key, _timeout_ms):
            return self.values[key]

        def key_value_delete(self, key):
            self.deleted.append(key)
            self.values.pop(key, None)

    client = Client()
    monkeypatch.setattr(broadcast.jax, "process_count", lambda: 2)
    monkeypatch.setattr(broadcast.jax, "process_index", lambda: 0)
    monkeypatch.setattr(global_state, "client", client)

    reduced = broadcast.reduce_bytes_to_all(
        np.frombuffer(b"a", dtype=np.uint8), key="gate",
        reduce=lambda records: records[1])

    np.testing.assert_array_equal(reduced, np.array([98], np.uint8))
    assert set(client.deleted) == {
        "gate/value/0", "gate/value/1", "gate/result",
        "gate/ack/0", "gate/ack/1"}


def test_host_reduction_refuses_oversized_control():
    from ffi.common.broadcast import reduce_bytes_to_all

    with np.testing.assert_raises_regex(ValueError, "exceeds bound"):
        reduce_bytes_to_all(
            np.zeros(5, np.uint8), key="gate", reduce=lambda records: records[0],
            max_bytes=4)


def test_host_reduction_publishes_root_failure_to_peers(monkeypatch):
    from ffi.common import broadcast
    from jax._src.distributed import global_state

    class Client:
        def __init__(self):
            self.values = {"gate/value/1": b"bb", "gate/ack/1": b"1"}

        def key_value_set_bytes(self, key, value):
            self.values[key] = value

        def blocking_key_value_get_bytes(self, key, _timeout_ms):
            return self.values[key]

        def key_value_delete(self, key):
            self.values.pop(key, None)

    monkeypatch.setattr(broadcast.jax, "process_count", lambda: 2)
    monkeypatch.setattr(broadcast.jax, "process_index", lambda: 0)
    monkeypatch.setattr(global_state, "client", Client())
    with np.testing.assert_raises_regex(RuntimeError, "published sizes"):
        broadcast.reduce_bytes_to_all(
            np.frombuffer(b"a", dtype=np.uint8), key="gate",
            reduce=lambda records: records[0])
