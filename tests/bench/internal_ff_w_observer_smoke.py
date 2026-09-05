#!/usr/bin/env python3
"""Tiny argv-driven P4 smoke for the internal-FF W observer.

This is deliberately not a pytest cell.  It exercises the real distributed
action and SlabIO paths with a padded centroid carrier, then validates the
logical on-disk payload after the collective writer has closed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.environ.get("LORRAX_SRC", "src"))

from runtime import initialize_communicator_stack  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import barrier  # noqa: E402
from gw.internal_ff_w_observer import (  # noqa: E402
    open_w_observer, plan_w_observer)


def _indices(part, extent):
    if isinstance(part, slice):
        return np.arange(*part.indices(extent), dtype=np.int64)
    return np.asarray([int(part)], np.int64)


def _matrix(mesh, shape, frequency):
    sharding = NamedSharding(mesh, P(None, "x", "y"))

    def callback(index):
        q = _indices(index[0], shape[0])[:, None, None]
        m = _indices(index[1], shape[1])[None, :, None]
        n = _indices(index[2], shape[2])[None, None, :]
        real = frequency + 0.1 * q + 0.01 * m + 0.001 * n
        imag = -0.5 * frequency + 0.02 * q - 0.003 * m + 0.004 * n
        return (real + 1j * imag).astype(np.complex128)

    return jax.make_array_from_callback(shape, sharding, callback)


def _host_matrix(shape, frequency):
    q = np.arange(shape[0])[:, None, None]
    m = np.arange(shape[1])[None, :, None]
    n = np.arange(shape[2])[None, None, :]
    real = frequency + 0.1 * q + 0.01 * m + 0.001 * n
    imag = -0.5 * frequency + 0.02 * q - 0.003 * m + 0.004 * n
    return (real + 1j * imag).astype(np.complex128)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mesh = RUNTIME.mesh
    if dict(mesh.shape) != {"x": 2, "y": 2}:
        raise RuntimeError(f"observer smoke requires a 2x2 mesh, got {mesh.shape}")
    if jax.process_count() != 4:
        raise RuntimeError(
            f"observer smoke requires four ranks, got {jax.process_count()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q_full = np.asarray([0, 9, 3, 7], np.int32)
    qfrac = np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
    ])
    spec = plan_w_observer(
        input_dir=output_dir,
        real_arms=[{
            "name": "real_eta_0.25000000",
            "requested_z_ry": np.asarray([0.0 + 0.25j, 0.25 + 0.25j]),
            "evaluated_z_ry": np.asarray([0.0 + 0.25j, 0.25 + 0.25j]),
        }],
        imag_grid={
            "name": "imaginary",
            "requested_z_ry": np.asarray([0.0j, 1.0j]),
            "evaluated_z_ry": np.asarray([1.0e-10j, 1.0j]),
        },
        q_full=q_full, q_irr_frac=qfrac, bvec_cart=np.eye(3),
        nmu_logical=5,
        centroid_identity={"sha256": "01" * 32},
        body_provenance={"smoke": "padded-P4"})
    if spec.selected_q_rows.tolist() != [0, 2, 3]:
        raise AssertionError(
            f"deterministic q selection failed: {spec.selected_q_rows}")

    physical_shape = (4, 8, 8)
    v_wedge = _matrix(mesh, physical_shape, 0.0)
    observer = open_w_observer(spec, mesh_xy=mesh, v_wedge=v_wedge)
    frequencies = (1.0, 2.0, 3.0, 4.0)
    for i in range(2):
        observer.observe(i, _matrix(mesh, physical_shape, frequencies[i]))
    observer.commit_prefix("real_eta_0.25000000", 2)
    for i in range(2, 4):
        observer.observe(i, _matrix(mesh, physical_shape, frequencies[i]))
    observer.commit_prefix("imaginary", 2)
    observer.close(body_complete=True)
    barrier("internal_ff_w_observer_smoke_closed")

    if jax.process_index() == 0:
        import h5py

        with h5py.File(spec.payload_path, "r") as handle:
            shapes = {
                name: tuple(handle[name].shape) for name in (
                    "v_selected_qmunu", "probe_qmur",
                    "wc_selected_zqmunu", "wc_action_zqmur")
            }
            expected_shapes = {
                "v_selected_qmunu": (3, 5, 5),
                "probe_qmur": (4, 5, 48),
                "wc_selected_zqmunu": (4, 3, 5, 5),
                "wc_action_zqmur": (4, 4, 5, 48),
            }
            if shapes != expected_shapes:
                raise AssertionError(f"logical dataset shapes: {shapes}")
            probes = handle["probe_qmur"][...]
            selected = handle["wc_selected_zqmunu"][...]
            actions = handle["wc_action_zqmur"][...]
            if np.array_equal(probes[..., :16], probes[..., 16:32]):
                raise AssertionError("proposal and held-out probes coincide")
            for i, frequency in enumerate(frequencies):
                logical = _host_matrix(physical_shape, frequency)[:, :5, :5]
                np.testing.assert_array_equal(
                    selected[i], logical[spec.selected_q_rows])
                np.testing.assert_allclose(
                    actions[i], np.einsum(
                        "qmn,qnr->qmr", logical, probes),
                    rtol=2.0e-13, atol=2.0e-13)
        state = json.loads(Path(spec.sidecar_path).read_text())
        if state["status"] != "body_complete" or (
                state["ready_prefix_by_arm"] != {
                    "real_eta_0.25000000": 2, "imaginary": 2}):
            raise AssertionError(f"bad terminal sidecar: {state}")
        print(json.dumps({
            "status": "pass", "mesh": dict(mesh.shape),
            "selected_q_rows": spec.selected_q_rows.tolist(),
            "dataset_shapes": {k: list(v) for k, v in shapes.items()},
            "centroid_carrier": state["centroid_carrier"],
        }, sort_keys=True), flush=True)
    barrier("internal_ff_w_observer_smoke_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
