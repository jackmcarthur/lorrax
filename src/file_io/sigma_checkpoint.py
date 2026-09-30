"""The swept Σ(ω) state, kept between the τ sweep and finalize.

A one-shot shared-pole Σ sweep can take over an hour, and finalize (head
injection, at-DFT read, QSGW build, file writes) can still fail after it.
``gw.sigma_dispatch`` hands the swept state to this owner once the sweep
returns; a rerun with ``restart = true`` whose state authenticates goes
straight to finalize. It covers one-shots only: an SC rerun starts at map 0,
SC retention deletes later map directories, and map 0's sweep also plans the
held Σ windows the later maps reuse.

The cubes (``body`` and, when present, ``unextrap`` (a cube, or the raw
twin's band-diagonal slots) and ``odd``) go through
SlabIO from their own shards: no gather, no copy, and the file is closed
before finalize donates the body. The small host fields (the head diagonal's
closed-form fields, the band-extrapolation payload, the padded band axis, one
ratio) are written by rank 0 after the cubes, and the ``commit`` attribute, the record's digest,
is written last. A file without it, or whose identity differs, is removed
and the sweep recomputed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "sigma-sweep-checkpoint-v1"
CUBES = ("body", "unextrap", "odd")
#: Measured SlabIO write rate of a swept cube, one node at P4 (Na 8^3, 61.4 GB
#: in 47.7 s); more ranks write faster, so this predicts long.
WRITE_BYTES_PER_S = 1.3e9
#: The checkpoint is written only when the sweep it protects took at least
#: this many times its predicted write.
PAYOFF = 5.0


def checkpoint_pays(cubes, sweep_seconds):
    """Return (write?, predicted write seconds) for these cubes."""
    nbytes = sum(int(np.prod(c.shape)) * np.dtype(c.dtype).itemsize
                 for c in cubes.values() if c is not None)
    predicted = nbytes / WRITE_BYTES_PER_S
    return sweep_seconds >= PAYOFF * predicted, predicted


def _encode(value):
    # Floats go through repr, which round-trips float64 exactly.
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"__nd__": value.real.tolist(), "__im__": value.imag.tolist(),
                    "dtype": str(value.dtype), "shape": list(value.shape)}
        return {"__nd__": value.tolist(), "dtype": str(value.dtype),
                "shape": list(value.shape)}
    if isinstance(value, (complex, np.complexfloating)):
        return {"__c__": [float(value.real), float(value.imag)]}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if isinstance(value, dict):
        if "__nd__" in value:
            real = np.asarray(value["__nd__"])
            if "__im__" in value:
                real = real + 1j * np.asarray(value["__im__"])
            return real.astype(value["dtype"]).reshape(value["shape"])
        if "__c__" in value:
            return complex(*value["__c__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _json(value):
    return json.dumps(value, default=_encode, sort_keys=True, allow_nan=False)


def sweep_identity(**parts):
    """One digest of everything the swept state depends on.

    Arrays enter by dtype, shape and bytes; everything else by canonical JSON.
    """
    digest = hashlib.sha256()
    for key in sorted(parts):
        value = parts[key]
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            digest.update(f"{value.dtype}{value.shape}".encode())
            digest.update(value.tobytes())
        else:
            # Identity only: an option without a JSON form (an enum) enters by repr.
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False, default=lambda v: (
                _encode(v) if isinstance(v, (np.ndarray, np.generic, complex)) else repr(v))
            ).encode())
    return digest.hexdigest()


def _spec(cube):
    from jax.sharding import NamedSharding
    sharding = getattr(cube, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return None
    return [list(a) if isinstance(a, tuple) else a for a in sharding.spec]


def checkpointable(cubes):
    """True when every present cube is a NamedSharding jax array."""
    return all(cube is None or _spec(cube) is not None for cube in cubes.values())


def write_sigma_checkpoint(path, *, identity, cubes, host, mesh):
    """Write the swept state collectively; returns (bytes, seconds)."""
    import time
    import h5py
    from common.collectives import rank0_transaction
    from .slab_io import SlabIO

    path = Path(path)
    started = time.monotonic()
    rank0_transaction(path, stage="sigma_checkpoint.clear",
                      write=lambda: path.unlink(missing_ok=True))
    present = {name: cube for name, cube in cubes.items() if cube is not None}
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        for name, cube in present.items():
            io.create_dataset(name, shape=tuple(cube.shape), dtype=cube.dtype)
            io.write_slab(name, cube)
    record = dict(schema=SCHEMA, identity=identity,
                  cubes={name: dict(shape=list(cube.shape), dtype=str(cube.dtype),
                                    spec=_spec(cube))
                         for name, cube in present.items()},
                  host=host)
    text = _json(record)

    def commit():
        with h5py.File(path, "a") as f:
            f.create_dataset("record", data=np.bytes_(text))
            f.attrs["commit"] = hashlib.sha256(text.encode()).hexdigest()
    rank0_transaction(path, stage="sigma_checkpoint.commit", write=commit)
    nbytes = sum(int(np.prod(c.shape)) * np.dtype(c.dtype).itemsize for c in present.values())
    return nbytes, time.monotonic() - started


def discard_sigma_checkpoint(path):
    """Remove a checkpoint (collective; rank 0 unlinks)."""
    from common.collectives import rank0_transaction
    path = Path(path)
    rank0_transaction(path, stage="sigma_checkpoint.discard",
                      write=lambda: path.unlink(missing_ok=True))


def read_sigma_checkpoint(path, *, identity, mesh, print_fn=print):
    """Return ``(cubes, host)`` for an authenticated checkpoint, else None.

    Rank 0 decides; a partial or foreign file is removed so the sweep
    recomputes and writes a fresh one.
    """
    import h5py
    from jax.sharding import PartitionSpec as P
    from common.collectives import agree_io_error, rank0_transaction
    from .slab_io import SlabIO

    path = Path(path)

    def check():
        if not path.exists():
            return "absent"
        try:
            with h5py.File(path, "r") as f:
                text = f["record"][()]
                commit = f.attrs["commit"]
            text = text.decode() if isinstance(text, bytes) else str(text)
            commit = commit.decode() if isinstance(commit, bytes) else str(commit)
            ok = (hashlib.sha256(text.encode()).hexdigest() == commit
                  and json.loads(text).get("identity") == identity
                  and json.loads(text).get("schema") == SCHEMA)
        except (OSError, KeyError, ValueError):
            ok = False
        if ok:
            return "match"
        path.unlink()
        return "removed"
    verdict = rank0_transaction(path, stage="sigma_checkpoint.check", write=check,
                                return_value=True)
    if verdict == "removed":
        print_fn(f"WARNING Sigma checkpoint: removed {path} (partial, or another "
                 "identity); recomputing the sweep")
    if verdict != "match":
        return None
    record, error = None, None
    try:
        with h5py.File(path, "r") as f:
            text = f["record"][()]
            record = json.loads(text.decode() if isinstance(text, bytes) else str(text))
    except (OSError, KeyError, ValueError) as exc:
        error = exc
    agree_io_error(error, path=path, stage="sigma_checkpoint.record")
    cubes = dict.fromkeys(CUBES)
    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        for name, meta in record["cubes"].items():
            spec = P(*[tuple(a) if isinstance(a, list) else a for a in meta["spec"]])
            cubes[name] = io.read_slab(name, shape=tuple(meta["shape"]),
                                       dtype=np.dtype(meta["dtype"]), mesh=mesh,
                                       partition_spec=spec)
    host = _decode(record["host"])
    print_fn(f"Sigma checkpoint: authenticated swept state reused from {path}; sweep skipped")
    return cubes, host
