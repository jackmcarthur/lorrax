"""Force a slow serial header reader before the next staged metadata writer."""
import json
import sys
import time
from pathlib import Path

from runtime import initialize_communicator_stack, finalize_process

runtime = initialize_communicator_stack(platform="gpu")
import h5py
import jax
from common.collectives import rank0_transaction
from file_io.commit_state import set_commit_state
from file_io import shared_pole_store as store


def main():
    path = Path(sys.argv[1])
    def create():
        with h5py.File(path, "w") as h5:
            h5.create_dataset('header_json', data=json.dumps({'round': 0}).encode())
            set_commit_state(h5, True)
    rank0_transaction(path, stage='header_fixture', write=create)
    read_header = store._read_header
    def delayed_read(path):
        if jax.process_index() == 2:
            time.sleep(0.3)
        return read_header(path)
    store._read_header = delayed_read
    try:
        header = store._read_staging_header(path)
        assert header == {'round': 0}
        # Hold rank zero's writer open past the deliberately delayed reader.
        # The old unfenced read fails at this boundary with HDF5 write-open.
        def append():
            with h5py.File(path, 'a') as h5:
                time.sleep(0.5)
                del h5['header_json']
                h5.create_dataset('header_json', data=json.dumps({'round': 1}).encode())
        rank0_transaction(path, stage='next_header_write', write=append)
        assert store._read_staging_header(path) == {'round': 1}
        assert store._read_staging_header(path.with_suffix('.absent')) is None
    finally:
        store._read_header = read_header
    if jax.process_index() == 0:
        print('PASS delayed-reader staged-header fence and absent-file branch', flush=True)


status = 1
try:
    main()
    status = 0
except BaseException:
    import traceback
    traceback.print_exc()
finalize_process(status)
