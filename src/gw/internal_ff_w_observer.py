"""Optional distributed Wc witness for the fit-free full-frequency oracle.

This module observes the Wc already produced by ``internal_ff_cd``.  It owns
neither response construction nor a Dyson solve, and its artifacts are never
read by the running self-energy calculation.  The fixed sketches are intended
for offline shared-pole fitting and independent certification.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


PROPOSAL_RANK = 16
HELDOUT_RANK = 32
PROBE_RANK = PROPOSAL_RANK + HELDOUT_RANK
OBSERVER_SCHEMA = 1
_PROBE_NAMESPACE = b"internal_ff_w_observer_v1\0"
_SPLITMIX_ADD = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_receipt(values) -> dict:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return {
        "dtype": array.dtype.str,
        "shape": [int(v) for v in array.shape],
        "sha256": digest.hexdigest(),
    }


def _complex_receipt(values) -> dict:
    values = np.asarray(values, np.complex128)
    return _array_receipt(values.view(np.float64))


def select_q_rows(q_full, q_irr_frac, bvec_cart) -> np.ndarray:
    """Choose Gamma, the nearest nonzero q, and the farthest q.

    Equal Cartesian norms are resolved by canonical full-q index and then
    wedge row.  The selection is entirely geometric and cannot leak W data.
    """
    q_full = np.asarray(q_full, np.int64)
    q_irr_frac = np.asarray(q_irr_frac, np.float64)
    bvec_cart = np.asarray(bvec_cart, np.float64)
    if q_full.ndim != 1 or q_irr_frac.shape != (q_full.size, 3):
        raise ValueError(
            "W observer q tables must have shapes (nq,) and (nq,3); got "
            f"{q_full.shape} and {q_irr_frac.shape}")
    if bvec_cart.shape != (3, 3):
        raise ValueError(
            f"W observer reciprocal basis must be (3,3), got {bvec_cart.shape}")
    gamma = np.flatnonzero(q_full == 0)
    if gamma.size != 1:
        raise ValueError(
            "W observer requires exactly one irreducible row with canonical "
            f"full-q index zero, found {gamma.size}")
    q_cart = q_irr_frac @ bvec_cart
    norm2 = np.einsum("qi,qi->q", q_cart, q_cart)
    nonzero = np.flatnonzero(norm2 > 64.0 * np.finfo(np.float64).eps)
    candidates = [(int(gamma[0]))]
    if nonzero.size:
        nearest = sorted(
            (int(i) for i in nonzero),
            key=lambda i: (float(norm2[i]), int(q_full[i]), i))
        farthest = sorted(
            (int(i) for i in nonzero),
            key=lambda i: (-float(norm2[i]), int(q_full[i]), i))
        candidates.extend((nearest[0], farthest[0]))
    selected = []
    for row in candidates:
        if row not in selected:
            selected.append(row)
    return np.asarray(selected, np.int32)


@dataclass(frozen=True)
class WObserverSpec:
    payload_path: str
    sidecar_path: str
    identity: dict
    identity_digest: str
    body_provenance: dict
    nmu_logical: int
    q_full: np.ndarray
    q_irr_frac: np.ndarray
    selected_q_rows: np.ndarray
    z_requested_ry: np.ndarray
    z_evaluated_ry: np.ndarray
    arm_code: np.ndarray
    arm_local_index: np.ndarray
    frequency_role: tuple[str, ...]
    arms: tuple[dict, ...]
    centroid_digest: str

    @property
    def nz(self) -> int:
        return int(self.z_requested_ry.size)

    def arm(self, name: str) -> dict:
        matches = [arm for arm in self.arms if arm["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"W observer does not know arm {name!r}")
        return matches[0]


def _normalise_arm(raw, *, code: int, kind: str) -> dict:
    if not isinstance(raw, dict):
        raise TypeError("W observer arm plans must be dictionaries")
    name = str(raw["name"])
    requested = np.asarray(raw["requested_z_ry"], np.complex128)
    evaluated = np.asarray(raw["evaluated_z_ry"], np.complex128)
    if requested.ndim != 1 or requested.shape != evaluated.shape:
        raise ValueError(
            f"W observer arm {name!r} requested/evaluated grids differ: "
            f"{requested.shape} vs {evaluated.shape}")
    if not requested.size or not (
            np.all(np.isfinite(requested)) and np.all(np.isfinite(evaluated))):
        raise ValueError(f"W observer arm {name!r} has an invalid frequency grid")
    return {
        "name": name,
        "kind": kind,
        "code": int(code),
        "requested": requested,
        "evaluated": evaluated,
    }


def plan_w_observer(*, input_dir, real_arms, imag_grid, q_full,
                    q_irr_frac, bvec_cart, nmu_logical,
                    centroid_identity, body_provenance) -> WObserverSpec:
    """Freeze all observer identities before any CD frequency is consumed."""
    q_full = np.asarray(q_full, np.int32)
    q_irr_frac = np.asarray(q_irr_frac, np.float64)
    selected = select_q_rows(q_full, q_irr_frac, bvec_cart)
    nmu_logical = int(nmu_logical)
    if nmu_logical <= 0:
        raise ValueError("W observer nmu_logical must be positive")
    if not isinstance(centroid_identity, dict):
        raise TypeError("W observer requires the centroid receipt dictionary")
    centroid_digest = str(
        centroid_identity.get("sha256", centroid_identity.get("digest", "")))
    if not centroid_digest:
        raise ValueError("W observer centroid receipt lacks a digest")

    raw_arms = [
        _normalise_arm(raw, code=i, kind="real")
        for i, raw in enumerate(real_arms)
    ]
    raw_arms.append(_normalise_arm(
        imag_grid, code=len(raw_arms), kind="imaginary"))
    if len({arm["name"] for arm in raw_arms}) != len(raw_arms):
        raise ValueError("W observer arm names must be unique")

    requested, evaluated, codes, local_indices, roles = [], [], [], [], []
    arms = []
    start = 0
    for arm in raw_arms:
        n = int(arm["requested"].size)
        stop = start + n
        requested.append(arm["requested"])
        evaluated.append(arm["evaluated"])
        codes.append(np.full(n, arm["code"], np.int16))
        local_indices.append(np.arange(n, dtype=np.int32))
        if arm["kind"] == "real":
            for iw in range(n):
                roles.append(("proposal_fit" if iw % 4 == 0 else
                              "proposal_validation" if iw % 4 == 2 else
                              "frequency_holdout"))
        else:
            roles.extend("moment_fit" if iw % 2 == 0 else "moment_holdout"
                         for iw in range(n))
        arms.append({
            "name": arm["name"], "kind": arm["kind"],
            "code": arm["code"], "start": start, "stop": stop,
            "n": n,
        })
        start = stop
    z_requested = np.concatenate(requested)
    z_evaluated = np.concatenate(evaluated)
    arm_code = np.concatenate(codes)
    arm_local_index = np.concatenate(local_indices)

    identity = {
        "schema": OBSERVER_SCHEMA,
        "policy": "internal_ff_w_observer_v1",
        "body_provenance": body_provenance,
        "nmu_logical": nmu_logical,
        "q_full": _array_receipt(q_full),
        "q_irr_frac": _array_receipt(q_irr_frac),
        "selected_q_rows": selected.tolist(),
        "selected_q_full_index": q_full[selected].tolist(),
        "z_requested_ry": _complex_receipt(z_requested),
        "z_evaluated_ry": _complex_receipt(z_evaluated),
        "arms": arms,
        "frequency_role": list(roles),
        "probe": {
            "family": "complex_rademacher_splitmix64_v1",
            "proposal_rank": PROPOSAL_RANK,
            "heldout_rank": HELDOUT_RANK,
            "centroid_digest": centroid_digest,
        },
    }
    identity_digest = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")).hexdigest()
    root = Path(input_dir)
    return WObserverSpec(
        payload_path=str(root / "internal_ff_cd_w_oracle.h5"),
        sidecar_path=str(root / "internal_ff_cd_w_oracle.json"),
        identity=identity, identity_digest=identity_digest,
        body_provenance=body_provenance, nmu_logical=nmu_logical,
        q_full=q_full, q_irr_frac=q_irr_frac,
        selected_q_rows=selected, z_requested_ry=z_requested,
        z_evaluated_ry=z_evaluated, arm_code=arm_code,
        arm_local_index=arm_local_index, frequency_role=tuple(roles),
        arms=tuple(arms), centroid_digest=centroid_digest)


def _probe_key(centroid_digest: str, role: str) -> np.uint64:
    digest = hashlib.sha256(
        _PROBE_NAMESPACE + centroid_digest.encode("ascii") + b"\0"
        + role.encode("ascii")).digest()
    return np.frombuffer(digest[:8], dtype="<u8")[0]


def _splitmix64(values) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = np.asarray(values, np.uint64) + _SPLITMIX_ADD
        values = (values ^ (values >> np.uint64(30))) * _SPLITMIX_MUL1
        values = (values ^ (values >> np.uint64(27))) * _SPLITMIX_MUL2
        return values ^ (values >> np.uint64(31))


def _slice_values(part, extent: int) -> np.ndarray:
    if isinstance(part, slice):
        start, stop, step = part.indices(extent)
        return np.arange(start, stop, step, dtype=np.int64)
    return np.asarray([int(part)], np.int64)


def _make_probes(spec: WObserverSpec, mesh_xy: Mesh,
                 nmu_storage: int) -> jax.Array:
    shape = (int(spec.q_full.size), int(nmu_storage), PROBE_RANK)
    sharding = NamedSharding(mesh_xy, P(None, "y", None))
    proposal_key = _probe_key(spec.centroid_digest, "proposal")
    heldout_key = _probe_key(spec.centroid_digest, "heldout")
    scale = np.sqrt(2.0 * spec.nmu_logical)

    def callback(index):
        qi = _slice_values(index[0], shape[0])[:, None, None]
        mu = _slice_values(index[1], shape[1])[None, :, None]
        col = _slice_values(index[2], shape[2])[None, None, :]
        qid = spec.q_full[qi]
        counter = ((qid.astype(np.uint64) * np.uint64(spec.nmu_logical)
                    + mu.astype(np.uint64)) * np.uint64(PROBE_RANK)
                   + col.astype(np.uint64))
        keys = np.where(col < PROPOSAL_RANK, proposal_key, heldout_key)
        bits = _splitmix64(counter ^ keys)
        real = 1.0 - 2.0 * (bits & np.uint64(1)).astype(np.float64)
        imag = 1.0 - 2.0 * ((bits >> np.uint64(1))
                            & np.uint64(1)).astype(np.float64)
        values = (real + 1j * imag) / scale
        return np.where(mu < spec.nmu_logical, values, 0.0).astype(
            np.complex128)

    return jax.make_array_from_callback(shape, sharding, callback)


def make_wc_action_kernel(mesh_xy, *, probe_rank=PROBE_RANK):
    """Return the one distributed Wc-times-probe action kernel."""
    if int(probe_rank) != PROBE_RANK:
        raise ValueError(
            f"W observer probe rank is fixed at {PROBE_RANK}, got {probe_rank}")
    from common.shard_map import shard_map

    def local(wc_wedge, probes):
        partial = jnp.einsum(
            "qmn,qnr->qmr", wc_wedge, probes, optimize=True)
        return jax.lax.psum(partial, "y")

    return jax.jit(shard_map(
        local, mesh=mesh_xy,
        in_specs=(P(None, "x", "y"), P(None, "y", None)),
        out_specs=P(None, "x", None), check_vma=False))


def _dataset_shapes(spec: WObserverSpec) -> dict:
    nq = int(spec.q_full.size)
    nsel = int(spec.selected_q_rows.size)
    nmu = int(spec.nmu_logical)
    return {
        "v_selected_qmunu": (nsel, nmu, nmu),
        "probe_qmur": (nq, nmu, PROBE_RANK),
        "wc_selected_zqmunu": (spec.nz, nsel, nmu, nmu),
        "wc_action_zqmur": (spec.nz, nq, nmu, PROBE_RANK),
    }


def _atomic_sidecar(path: str, state: dict) -> None:
    from common.collectives import barrier, process_rank

    if process_rank() == 0:
        target = Path(path)
        tmp = Path(str(target) + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, target)
    barrier("internal_ff_w_observer_sidecar")


def _new_sidecar(spec: WObserverSpec, *, status: str,
                 nmu_storage: int) -> dict:
    return {
        "schema": OBSERVER_SCHEMA,
        "status": status,
        "identity": spec.identity,
        "identity_digest": spec.identity_digest,
        "source_commit": os.environ.get(
            "LORRAX_SOURCE_COMMIT", "working-tree"),
        "jobid": os.environ.get("SLURM_JOB_ID", "unknown"),
        "payload_path": spec.payload_path,
        "sidecar_path": spec.sidecar_path,
        "centroid_carrier": {
            "logical": int(spec.nmu_logical),
            "storage": int(nmu_storage),
            "padding_policy": "zero_probe_and_slabio_logical_extent_clip",
        },
        "ready_prefix_by_arm": {arm["name"]: 0 for arm in spec.arms},
        "observer_seconds": {"action": 0.0, "enqueue": 0.0, "drain": 0.0},
        "datasets": {
            name: {"shape": list(shape), "dtype": "complex128"}
            for name, shape in _dataset_shapes(spec).items()
        },
        "arms": list(spec.arms),
        "checkpoint_paths": {
            arm["name"]: ("internal_ff_cd_checkpoints/imaginary.npz"
                           if arm["kind"] == "imaginary" else
                           f"internal_ff_cd_checkpoints/{arm['name']}.npz")
            for arm in spec.arms
        },
    }


def _create_datasets(io, spec: WObserverSpec) -> None:
    for name, shape in _dataset_shapes(spec).items():
        io.create_dataset(name, shape=shape, dtype=np.complex128)


def _validate_resume_state(state: dict, spec: WObserverSpec,
                           nmu_storage: int) -> None:
    if state.get("schema") != OBSERVER_SCHEMA:
        raise ValueError(
            f"W observer sidecar schema {state.get('schema')!r} is not "
            f"{OBSERVER_SCHEMA}")
    expected_carrier = {
        "logical": int(spec.nmu_logical),
        "storage": int(nmu_storage),
        "padding_policy": "zero_probe_and_slabio_logical_extent_clip",
    }
    if state.get("centroid_carrier") != expected_carrier:
        raise ValueError(
            "W observer centroid carrier changed across resume; start a new "
            "run variant")
    expected_datasets = {
        name: {"shape": list(shape), "dtype": "complex128"}
        for name, shape in _dataset_shapes(spec).items()
    }
    if state.get("datasets") != expected_datasets:
        raise ValueError(
            "W observer sidecar dataset schema changed; start a new run variant")
    ready = state.get("ready_prefix_by_arm")
    expected_arms = {arm["name"] for arm in spec.arms}
    if not isinstance(ready, dict) or set(ready) != expected_arms:
        raise ValueError("W observer sidecar readiness arms are incompatible")
    for arm in spec.arms:
        value = ready[arm["name"]]
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 0 or value > int(arm["n"])):
            raise ValueError(
                f"W observer sidecar has invalid {arm['name']} prefix {value!r}")
    timings = state.get("observer_seconds")
    if not isinstance(timings, dict) or set(timings) != {
            "action", "enqueue", "drain"} or any(
                not np.isfinite(value) or value < 0.0
                for value in timings.values()):
        raise ValueError("W observer sidecar has invalid timing accumulators")


def _write_allocation_metadata(io, spec: WObserverSpec) -> None:
    io.write_attr("observer_schema", np.asarray(OBSERVER_SCHEMA, np.int32))
    io.write_attr("observer_identity_json", np.bytes_(
        _canonical_json(spec.identity)))
    io.write_attr("body_provenance_json", np.bytes_(
        _canonical_json(spec.body_provenance)))
    io.write_attr("z_requested_ry", spec.z_requested_ry)
    io.write_attr("z_evaluated_ry", spec.z_evaluated_ry)
    io.write_attr("arm_code", spec.arm_code)
    io.write_attr("arm_local_index", spec.arm_local_index)
    io.write_attr("frequency_role", np.asarray(spec.frequency_role, dtype="S24"))
    io.write_attr("q_wedge_full_index", spec.q_full)
    io.write_attr("q_irr_frac", spec.q_irr_frac)
    io.write_attr("selected_q_row", spec.selected_q_rows)
    io.write_attr("selected_q_full_index", spec.q_full[spec.selected_q_rows])
    io.write_attr("probe_column_role", np.asarray(
        ["proposal"] * PROPOSAL_RANK + ["heldout"] * HELDOUT_RANK,
        dtype="S8"))
    io.write_attr("logical_shapes_json", np.bytes_(
        _canonical_json(_dataset_shapes(spec))))
    io.write_attr("partition_specs_json", np.bytes_(_canonical_json({
        "v_selected_qmunu": "P(None,x,y)",
        "probe_qmur": "P(None,y,None)",
        "wc_selected_zqmunu": "P(None,None,x,y)",
        "wc_action_zqmur": "P(None,None,x,None)",
    })))


class InternalFFWObserver:
    """Persistent collective writer owned by one CD body invocation."""

    def __init__(self, spec, *, io, probes, action_kernel, state,
                 nmu_storage):
        self.spec = spec
        self._io = io
        self._probes = probes
        self._action = action_kernel
        self._state = state
        self._nmu_storage = int(nmu_storage)
        self._retained = []
        self._pending = []
        self._closed = False

    @property
    def artifact_receipt(self) -> dict:
        return {
            "payload_path": self.spec.payload_path,
            "sidecar_path": self.spec.sidecar_path,
            "identity_digest": self.spec.identity_digest,
        }

    def require_checkpoint_prefix(self, arm: str, completed: int) -> None:
        plan = self.spec.arm(arm)
        completed = int(completed)
        ready = int(self._state["ready_prefix_by_arm"][arm])
        if completed < 0 or completed > int(plan["n"]):
            raise ValueError(
                f"W observer received invalid CD prefix {completed} for {arm}")
        if ready < completed:
            raise ValueError(
                f"W observer payload for {arm} is ready through {ready}, but "
                f"the CD checkpoint consumed {completed}; missing W cannot be "
                "reconstructed. Start a new run variant.")

    def observe(self, global_frequency_index: int, wc_wedge) -> None:
        if self._closed:
            raise RuntimeError("W observer is already closed")
        i = int(global_frequency_index)
        if i < 0 or i >= self.spec.nz:
            raise IndexError(f"W observer frequency index {i} is out of range")
        expected = (int(self.spec.q_full.size), self._nmu_storage,
                    self._nmu_storage)
        if tuple(wc_wedge.shape) != expected:
            raise ValueError(
                f"W observer Wc carrier {wc_wedge.shape} != {expected}")
        t0 = time.perf_counter()
        action = self._action(wc_wedge, self._probes)
        action.block_until_ready()
        self._state["observer_seconds"]["action"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        selected = jnp.take(
            wc_wedge, jnp.asarray(self.spec.selected_q_rows), axis=0)
        selected_view = selected[None, ...]
        action_view = action[None, ...]
        # Keep the mesh-divisible physical carrier.  The datasets have
        # logical nmu extents, so SlabIO's defined dataset-minus-offset
        # clipping drops padded rows/columns without a gather or an illegal
        # nondivisible JAX slice.  Probe padding is exactly zero.
        self._io.write_slab(
            "wc_selected_zqmunu", selected_view, offset=(i, 0, 0, 0))
        self._io.write_slab(
            "wc_action_zqmur", action_view, offset=(i, 0, 0, 0))
        self._retained.extend((selected, selected_view, action, action_view))
        self._pending.append(i)
        self._state["observer_seconds"]["enqueue"] += time.perf_counter() - t0

    def commit_prefix(self, arm: str, n: int) -> None:
        plan = self.spec.arm(arm)
        n = int(n)
        if n < 0 or n > int(plan["n"]):
            raise ValueError(f"W observer prefix {n} is invalid for {arm}")
        if any(not (int(plan["start"]) <= i < int(plan["stop"]))
               for i in self._pending):
            raise ValueError(
                f"W observer pending frequencies cross arm boundary for {arm}")
        pending_stop = int(plan["start"]) + n
        pending_start = pending_stop - len(self._pending)
        if self._pending != list(range(pending_start, pending_stop)):
            raise ValueError(
                f"W observer pending frequencies do not end at {arm} prefix {n}")
        t0 = time.perf_counter()
        self._io.sync_writes()
        self._state["observer_seconds"]["drain"] += time.perf_counter() - t0
        ready = int(self._state["ready_prefix_by_arm"][arm])
        self._state["ready_prefix_by_arm"][arm] = max(ready, n)
        self._state["status"] = "active"
        _atomic_sidecar(self.spec.sidecar_path, self._state)
        self._retained.clear()
        self._pending.clear()

    def close(self, *, body_complete: bool) -> None:
        if self._closed:
            return
        self._io.close()
        self._closed = True
        self._retained.clear()
        self._pending.clear()
        if body_complete:
            incomplete = {
                arm["name"]: (self._state["ready_prefix_by_arm"][arm["name"]],
                              arm["n"])
                for arm in self.spec.arms
                if int(self._state["ready_prefix_by_arm"][arm["name"]])
                != int(arm["n"])
            }
            if incomplete:
                raise ValueError(
                    f"W observer cannot complete with incomplete arms {incomplete}")
            self._state["status"] = "body_complete"
            _atomic_sidecar(self.spec.sidecar_path, self._state)


def open_w_observer(spec, *, mesh_xy, v_wedge) -> InternalFFWObserver:
    """Allocate or authenticate, then open the one persistent SlabIO owner."""
    from common.collectives import barrier
    from file_io.slab_io import SlabIO

    if tuple(v_wedge.shape[:1]) != (int(spec.q_full.size),):
        raise ValueError(
            f"W observer V q carrier {v_wedge.shape} does not match "
            f"{spec.q_full.size} irreducible rows")
    if v_wedge.ndim != 3 or v_wedge.shape[1] != v_wedge.shape[2]:
        raise ValueError(f"W observer requires square wedge V, got {v_wedge.shape}")
    nmu_storage = int(v_wedge.shape[-1])
    if nmu_storage < spec.nmu_logical:
        raise ValueError(
            f"W observer V carrier {nmu_storage} is shorter than logical "
            f"nmu={spec.nmu_logical}")
    payload = Path(spec.payload_path)
    sidecar = Path(spec.sidecar_path)
    payload_exists, sidecar_exists = payload.exists(), sidecar.exists()
    if payload_exists != sidecar_exists:
        raise ValueError(
            "W observer found only one transaction artifact; start a new run "
            f"variant instead of repairing {payload} / {sidecar}")

    probes = _make_probes(spec, mesh_xy, nmu_storage)
    if not payload_exists:
        state = _new_sidecar(
            spec, status="allocating", nmu_storage=nmu_storage)
        _atomic_sidecar(spec.sidecar_path, state)
        with SlabIO(spec.payload_path, mode="w", mesh=mesh_xy) as io:
            _create_datasets(io, spec)
            selected_v = jnp.take(
                v_wedge, jnp.asarray(spec.selected_q_rows), axis=0)
            io.write_slab("v_selected_qmunu", selected_v)
            io.write_slab("probe_qmur", probes)
            _write_allocation_metadata(io, spec)
        state["status"] = "active"
        _atomic_sidecar(spec.sidecar_path, state)
    else:
        state = json.loads(sidecar.read_text())
        if state.get("identity_digest") != spec.identity_digest or (
                state.get("identity") != spec.identity):
            raise ValueError(
                "W observer artifacts have an incompatible identity; start a "
                "new run variant")
        if state.get("status") == "body_complete":
            raise ValueError(
                "W observer artifact is complete and write-once; consume it "
                "offline or start a new run variant")
        if state.get("status") != "active":
            raise ValueError(
                f"W observer sidecar status {state.get('status')!r} is not "
                "resumable; start a new run variant")
        _validate_resume_state(state, spec, nmu_storage)

    barrier("internal_ff_w_observer_before_append")
    io = SlabIO(spec.payload_path, mode="a", mesh=mesh_xy)
    try:
        _create_datasets(io, spec)
    except BaseException:
        io.close()
        raise
    return InternalFFWObserver(
        spec, io=io, probes=probes,
        action_kernel=make_wc_action_kernel(mesh_xy), state=state,
        nmu_storage=nmu_storage)


__all__ = [
    "PROPOSAL_RANK", "HELDOUT_RANK", "PROBE_RANK", "OBSERVER_SCHEMA",
    "WObserverSpec", "InternalFFWObserver", "select_q_rows",
    "plan_w_observer", "make_wc_action_kernel", "open_w_observer",
]
