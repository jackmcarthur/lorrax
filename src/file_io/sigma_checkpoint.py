"""Checkpoints of sharded cubes plus a small host record, read back by authenticated identity.

Two owners use it. ``gw.sigma_dispatch`` keeps a one-shot shared-pole Σ sweep
(over an hour on large decks) between the τ sweep and finalize: a rerun with
``restart = true`` whose state authenticates goes straight to finalize. The SC
loop (``gw.sc_iteration``) keeps its Anderson window and carry beside each
warm seed, ``sc_seed/sc_checkpoint.h5`` (``SC_SCHEMA``), so a fresh process
continues the trajectory (docs/self_consistency.md §8).

The cubes go through SlabIO from their own shards: no gather, no copy. The
host record (JSON) and, for the SC loop, one ``state`` blob are written by
rank 0 after the cubes, and the ``commit`` attribute, the record's digest, is
written last; the file is built as a private sibling and published by
``os.replace``, so a kill mid-write leaves the previous checkpoint. The blob
is a pickle of plain Python and numpy containers only (dict, list, tuple,
str, numbers, None, ndarray); :func:`_plain_loads` refuses any other class,
and the writer round-trips it through that loader before publishing.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np

SCHEMA = "sigma-sweep-checkpoint-v1"
SC_SCHEMA = "sc-anderson-checkpoint-v1"
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


class _PlainLoader(pickle.Unpickler):
    """Unpickle plain containers and numpy arrays; refuse every other class."""

    def find_class(self, module, name):
        if ((module.split(".")[0] == "numpy"
             and name in ("_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"))
                or (module == "builtins" and name == "complex")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"checkpoint state holds {module}.{name}; only plain containers "
            "and numpy arrays are stored")


def _plain_loads(data):
    return _PlainLoader(io.BytesIO(data)).load()


def write_sigma_checkpoint(path, *, identity, cubes, host, mesh,
                           schema=SCHEMA, state=None):
    """Write the cubes, host record and optional ``state`` collectively.

    Returns (bytes, seconds). ``state`` is pickled (plain containers only).
    """
    import os
    import time
    import h5py
    from common.collectives import rank0_transaction
    from .slab_io import SlabIO

    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    started = time.monotonic()
    blob = None
    if state is not None:
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        _plain_loads(blob)
    rank0_transaction(partial, stage="checkpoint.clear",
                      write=lambda: partial.unlink(missing_ok=True))
    present = {name: cube for name, cube in cubes.items() if cube is not None}
    with SlabIO(str(partial), mode="w", mesh=mesh) as io:
        for name, cube in present.items():
            io.create_dataset(name, shape=tuple(cube.shape), dtype=cube.dtype)
            io.write_slab(name, cube)
    record = dict(schema=schema, identity=identity,
                  cubes={name: dict(shape=list(cube.shape), dtype=str(cube.dtype),
                                    spec=_spec(cube))
                         for name, cube in present.items()},
                  host=host,
                  state_sha256=None if blob is None else hashlib.sha256(blob).hexdigest())
    text = _json(record)

    def commit():
        with h5py.File(partial, "a") as f:
            if blob is not None:
                f.create_dataset("state", data=np.frombuffer(blob, np.uint8))
            f.create_dataset("record", data=np.bytes_(text))
            f.attrs["commit"] = hashlib.sha256(text.encode()).hexdigest()
        os.replace(partial, path)
    rank0_transaction(path, stage="checkpoint.commit", write=commit)
    nbytes = sum(int(np.prod(c.shape)) * np.dtype(c.dtype).itemsize for c in present.values())
    return nbytes + (0 if blob is None else len(blob)), time.monotonic() - started


def discard_sigma_checkpoint(path):
    """Remove a checkpoint (collective; rank 0 unlinks)."""
    from common.collectives import rank0_transaction
    path = Path(path)
    rank0_transaction(path, stage="sigma_checkpoint.discard",
                      write=lambda: path.unlink(missing_ok=True))


def read_sigma_checkpoint(path, *, identity, mesh, print_fn=print,
                          schema=SCHEMA, discard=True):
    """Return ``(cubes, host, state)`` for an authenticated checkpoint, else
    ``(None, reason)``.

    Rank 0 decides. A partial or foreign file is removed (``discard``) so the
    caller recomputes; with ``discard=False`` it is left in place and the
    reason names the identity fields that differ.
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
            record = json.loads(text)
            if (hashlib.sha256(text.encode()).hexdigest() != commit
                    or record.get("schema") != schema):
                reason = "partial or other schema"
            elif record.get("identity") == identity:
                return "match"
            elif isinstance(identity, dict) and isinstance(record.get("identity"), dict):
                got = record["identity"]
                reason = "identity differs: " + ", ".join(
                    sorted(k for k in set(got) | set(identity) if got.get(k) != identity.get(k)))
            else:
                reason = "identity differs"
        except (OSError, KeyError, ValueError):
            reason = "unreadable or partial"
        if discard:
            path.unlink()
            return f"removed ({reason})"
        return reason
    verdict = rank0_transaction(path, stage="checkpoint.check", write=check,
                                return_value=True)
    if verdict.startswith("removed"):
        print_fn(f"WARNING checkpoint: {verdict} {path}; recomputing")
    if verdict != "match":
        return None, verdict
    record, state, error = None, None, None
    try:
        with h5py.File(path, "r") as f:
            text = f["record"][()]
            record = json.loads(text.decode() if isinstance(text, bytes) else str(text))
            if record.get("state_sha256") is not None:
                blob = f["state"][()].tobytes()
                if hashlib.sha256(blob).hexdigest() != record["state_sha256"]:
                    raise ValueError("checkpoint state digest differs from its record")
                state = _plain_loads(blob)
    except (OSError, KeyError, ValueError) as exc:
        error = exc
    agree_io_error(error, path=path, stage="checkpoint.record")
    cubes = dict.fromkeys(CUBES) if schema == SCHEMA else {}
    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        for name, meta in record["cubes"].items():
            spec = P(*[tuple(a) if isinstance(a, list) else a for a in meta["spec"]])
            cubes[name] = io.read_slab(name, shape=tuple(meta["shape"]),
                                       dtype=np.dtype(meta["dtype"]), mesh=mesh,
                                       partition_spec=spec)
    host = _decode(record["host"])
    print_fn(f"checkpoint: authenticated {schema} state read from {path}")
    return cubes, host, state
