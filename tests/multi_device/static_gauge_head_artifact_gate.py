"""P=4 SlabIO roundtrip and red-twin gate for ``StaticGaugeHead``.

Run on one four-GPU node with a new path in a registered run directory::

    lx run -G 4 -n 4 python3 -u \
      tests/multi_device/static_gauge_head_artifact_gate.py \
      --path /bounded/run/static_gauge_head_gate.h5
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
from types import SimpleNamespace

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TESTS))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import barrier, process_count, resolve_mesh  # noqa: E402
from file_io.static_gauge_head import (  # noqa: E402
    LoadedStaticGaugeHeadResponse,
    STATIC_GAUGE_HEAD_CONVENTION_ID,
    load_static_gauge_head_artifact,
    write_static_gauge_head_artifact,
)
from gw.gw_config import HeadCorrection  # noqa: E402
from gw.head_correction import StaticGaugeHeadResponse  # noqa: E402
from gw.photon_layout import (  # noqa: E402
    PhotonBasisLayout,
    pack_photon_response_tiles,
    unpack_photon_response_tiles,
)
from gw.w_isdf import compute_static_photon_response  # noqa: E402


_FINGERPRINT = "sha256:" + "9" * 64


def _put(value, mesh, spec):
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


def _assert_same_local(got, want, name):
    got_shards = {tuple((s.start, s.stop, s.step) for s in sh.index):
                  np.asarray(sh.data)
                  for sh in got.addressable_shards}
    want_shards = {tuple((s.start, s.stop, s.step) for s in sh.index):
                   np.asarray(sh.data)
                   for sh in want.addressable_shards}
    if got_shards.keys() != want_shards.keys():
        raise AssertionError(
            f"{name} local shard indices changed: "
            f"{got_shards.keys()} != {want_shards.keys()}")
    for index in got_shards:
        np.testing.assert_array_equal(
            got_shards[index], want_shards[index], err_msg=f"{name} {index}")


def _response(mesh):
    layout = PhotonBasisLayout.from_centroid_extents(4, 4, mesh)
    n_body = layout.packed_extent
    S = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    S[0, 0, 0, 0] = 0.31
    S[0, 1, 0, 0] = S[1, 0, 0, 0] = -0.07
    S[1, 1, 0, 0] = 0.43
    indices_Y = np.indices((2, 4, n_body))
    Y = (0.1 * indices_Y[0] + 0.01 * indices_Y[1]
         + 0.001j * indices_Y[2]).astype(np.complex128)
    indices_Z = np.indices((2, n_body, 4))
    Z = (-0.2 * indices_Z[0] + 0.003 * indices_Z[1]
         - 0.02j * indices_Z[2]).astype(np.complex128)
    return StaticGaugeHeadResponse(
        layout=layout,
        S_direct=_put(S, mesh, P()),
        sigma_H=np.asarray((0.0, 0.0, 0.19), dtype=np.float64),
        Y_x=_put(Y, mesh, P(None, None, "x")),
        Z_y=_put(Z, mesh, P(None, "y", None)),
        hamiltonian_config_operator_fingerprint=_FINGERPRINT,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=0.0,
        hermiticity_residual=0.0,
    )


def check(path: str):
    if process_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "static gauge artifact gate requires exactly four ranks/devices; "
            f"got ranks={process_count()}, devices={jax.device_count()}")
    mesh = resolve_mesh()
    source = _response(mesh)
    write_static_gauge_head_artifact(
        path, source, mesh_xy=mesh,
        source_write_ibz_only=True, source_low_mem_bands=True)
    loaded = load_static_gauge_head_artifact(
        path, mesh_xy=mesh,
        expected_hamiltonian_config_operator_fingerprint=_FINGERPRINT)

    if type(loaded) is not LoadedStaticGaugeHeadResponse:
        raise AssertionError(f"loader returned {type(loaded)!r}")
    if loaded.convention_id != STATIC_GAUGE_HEAD_CONVENTION_ID:
        raise AssertionError("loader returned the wrong convention ID")
    if not loaded.source_write_ibz_only or not loaded.source_low_mem_bands:
        raise AssertionError("IBZ/low-mem storage stamps did not roundtrip")
    _assert_same_local(loaded.S_direct, source.S_direct, "S_direct")
    _assert_same_local(loaded.Y_x, source.Y_x, "Y_x")
    _assert_same_local(loaded.Z_y, source.Z_y, "Z_y")
    np.testing.assert_array_equal(loaded.sigma_H, source.sigma_H)
    for array, wanted in (
        (loaded.S_direct, np.complex128),
        (loaded.Y_x, np.complex128),
        (loaded.Z_y, np.complex128),
        (loaded.sigma_H, np.float64),
    ):
        if np.dtype(array.dtype) != np.dtype(wanted):
            raise AssertionError(f"fixed dtype changed: {array.dtype}")

    try:
        replace(loaded, _loader_token=object())
    except TypeError as exc:
        if "issued only by" not in str(exc):
            raise
    else:
        raise AssertionError("caller fabricated a loader-issued response")
    try:
        load_static_gauge_head_artifact(
            path, mesh_xy=mesh,
            expected_hamiltonian_config_operator_fingerprint=(
                "sha256:" + "8" * 64))
    except ValueError as exc:
        if "fingerprint mismatch" not in str(exc):
            raise
    else:
        raise AssertionError("stale operator fingerprint was accepted")
    try:
        write_static_gauge_head_artifact(
            path, source, mesh_xy=mesh,
            source_write_ibz_only=False, source_low_mem_bands=False)
    except FileExistsError as exc:
        if "immutable" not in str(exc):
            raise
    else:
        raise AssertionError("immutable artifact was overwritten")

    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.FULL))
    try:
        compute_static_photon_response(
            None, None, None, None, None, mesh, config=config,
            gauge_head_response=loaded)
    except ValueError as exc:
        if "static_gauge_head_producer_unavailable" not in str(exc):
            raise
    else:
        raise AssertionError("loader-issued fixture opened FULL_SCREENED")

    nq = 3
    block_sharding = NamedSharding(mesh, P(None, "x", "y"))
    tile_01 = _put(
        np.arange(nq * 16, dtype=np.float64).reshape(nq, 4, 4).astype(
            np.complex128), mesh, P(None, "x", "y"))
    tile_32 = _put(
        np.full((nq, 4, 4), 2j, dtype=np.complex128),
        mesh, P(None, "x", "y"))
    packed = pack_photon_response_tiles(
        {(0, 1): tile_01, (3, 2): tile_32}, nq, source.layout, mesh)
    if not packed.sharding.is_equivalent_to(block_sharding, 3):
        raise AssertionError(f"packed response sharding changed: {packed.sharding}")
    tiles = unpack_photon_response_tiles(packed, source.layout, mesh)
    _assert_same_local(tiles[0][1], tile_01, "tile_01")
    _assert_same_local(tiles[3][2], tile_32, "tile_32")
    for A in range(4):
        for B in range(4):
            if (A, B) in ((0, 1), (3, 2)):
                continue
            for shard in tiles[A][B].addressable_shards:
                if np.any(np.asarray(shard.data) != 0):
                    raise AssertionError(f"absent tile ({A},{B}) is nonzero")

    barrier("static_gauge_head_gate_checked")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return {
        "ranks": process_count(),
        "mesh": tuple(int(v) for v in mesh.devices.shape),
        "ibz": loaded.source_write_ibz_only,
        "low_mem_bands": loaded.source_low_mem_bands,
        "lorentz_tiles": (4, 4),
    }


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    result = check(args.path)
    if jax.process_index() == 0:
        print(f"PASS static_gauge_head_artifact {result}", flush=True)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_main)
