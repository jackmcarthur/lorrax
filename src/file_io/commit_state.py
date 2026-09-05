"""Persistent completion receipt for collective artifact mutations (L3).

Zero is the HDF5 fill value: even a new writer dying before its first data
write leaves an uncommitted file. Legacy artifacts without this dataset retain
their format-specific checks. This receipt authenticates completion, not physics.
"""
COMMIT_STATE = 'lorrax_io_committed'


def assert_committed(h5, *, path=None):
    """Refuse a readable HDF5 file left inside a collective write transaction."""
    if COMMIT_STATE in h5 and int(h5[COMMIT_STATE][0]) != 1:
        raise ValueError(
            f"GATE io_global_commit: path={path or h5.filename}; "
            "stage=restart read; artifact is not globally committed. "
            "Do not reuse this incomplete artifact; rebuild in a new run directory.")


def set_commit_state(h5, committed):
    """Write the small receipt through an already-open serial HDF5 handle."""
    if COMMIT_STATE not in h5:
        h5.create_dataset(COMMIT_STATE, shape=(1,), dtype='int32')
    h5[COMMIT_STATE][0] = int(committed)
    h5.flush()
