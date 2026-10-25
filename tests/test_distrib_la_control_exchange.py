"""The host-only route exchange keeps rank order with a bounded RPC count."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
import time

import pytest


@pytest.fixture
def cohort(monkeypatch):
    import jax
    from jax._src.distributed import global_state
    owner = importlib.import_module('distrib_la.plan')
    local = threading.local()
    count = 4

    class Generations(dict):
        # Threads represent distinct processes, each with its own module counter.
        def get(self, key, default=None):
            return super().get((local.rank,key), default)
        def __setitem__(self, key, value):
            super().__setitem__((local.rank,key), value)

    class Client:
        def __init__(self):
            self.data = {}
            self.calls = Counter()
            self.condition = threading.Condition()
            self.omit = False
        def key_value_set_bytes(self, key, value):
            with self.condition:
                self.calls['set'] += 1
                self.data[key] = value
                self.condition.notify_all()
        def key_value_increment(self, key, value):
            with self.condition:
                self.calls['increment'] += 1
                self.data[key] = self.data.get(key,0)+value
                return self.data[key]
        def blocking_key_value_get(self, key, _timeout):
            with self.condition:
                self.calls['wait'] += 1
                if not self.condition.wait_for(lambda: key in self.data, timeout=3):
                    raise RuntimeError('test cohort did not complete')
                return self.data[key].decode()
        def key_value_dir_get_bytes(self, prefix):
            with self.condition:
                self.calls['directory'] += 1
                # Directory order is deliberately unrelated to process order.
                return [(key,self.data[key]) for key in sorted(self.data,reverse=True)
                        if key.startswith(prefix+'/') and not (self.omit and key.endswith('/2'))]

    client = Client()
    monkeypatch.setattr(global_state, 'client', client)
    monkeypatch.setattr(jax, 'process_count', lambda: count)
    monkeypatch.setattr(jax, 'process_index', lambda: local.rank)
    monkeypatch.setattr(owner, '_AGREE_GEN', Generations())
    for name in ('devices','local_devices','default_backend'):
        monkeypatch.setattr(jax, name, lambda: pytest.fail('host exchange initialized a client'))

    def run(rank):
        local.rank = rank
        outputs = []
        for generation in range(2):
            time.sleep(.002*((rank+generation)%count))
            # An error is ordinary control data; every rank must see it unchanged.
            payload = dict(rank=rank,generation=generation)
            if rank == 2:
                payload['error'] = 'provider query unavailable'
            outputs.append(owner._exchange('workspace_fixture',payload))
        return outputs
    return count, client, run


def test_bulk_exchange_orders_ranks_and_generations_with_linear_rpcs(cohort):
    count, client, run = cohort
    with ThreadPoolExecutor(max_workers=count) as pool:
        outputs = list(pool.map(run,range(count)))
    assert all(out == outputs[0] for out in outputs)
    for generation, rows in enumerate(outputs[0]):
        assert [row['rank'] for row in rows] == list(range(count))
        assert {row['generation'] for row in rows} == {generation}
        assert rows[2]['error'] == 'provider query unavailable'
    assert client.calls == dict(set=2*(count+1),increment=2*count,wait=2*count,directory=2*count)
    assert sum(client.calls.values()) == 2*(4*count+1)


def test_completed_directory_cannot_hide_a_missing_rank(cohort):
    count, client, run = cohort
    client.omit = True
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(run,rank) for rank in range(count)]
        for future in futures:
            with pytest.raises(RuntimeError, match='missing ranks'):
                future.result(timeout=5)
