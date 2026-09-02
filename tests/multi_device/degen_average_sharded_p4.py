"""P=4 gate: diagonal averaging never replicates a band-sharded Sigma.

Run as one process per GPU on a 2x2 mesh.  The expected full matrix is known
on every rank, but the assertion reads only each rank's addressable output
tile.  A replicated output or a changed off-diagonal therefore fails by
layout or value, without using a full-matrix gather in the gate itself.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def _fail(message: str) -> None:
    print(f"[degen-average-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def main() -> None:
    if jax.device_count() != 4 or jax.process_count() != 4:
        _fail(
            f"requires four processes/four devices, got "
            f"{jax.process_count()}/{jax.device_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    rank = jax.process_index()
    if rank == 0:
        print("[degen-average-p4] processes=4 devices=4 mesh=2x2", flush=True)

    from common.collectives import device_put_process_local
    from gw.degen_average import (
        apply_to_matrix_diagonals,
        average_sigma_components,
    )
    from gw.gw_output import GWResults, _result_matrix_diag
    from gw.qsgw_utils import build_qsgw_sigma_xc, static_sigma_diag_to_host

    rng = np.random.default_rng(20260831)
    nk, nb = 3, 8
    source = (rng.standard_normal((nk, nb, nb))
              + 1j * rng.standard_normal((nk, nb, nb)))
    energies = np.tile(np.arange(nb, dtype=np.float64), (nk, 1))
    energies[0, 1] = energies[0, 0]
    energies[1, 4] = energies[1, 3]
    expected = apply_to_matrix_diagonals(source, energies, 1.0e-8)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    operand = device_put_process_local(source, sharding)

    with mesh:
        result = average_sigma_components(
            operand, operand, operand, operand, operand, None, operand, None,
            energies_kn_ry=energies, tol_ry=1.0e-8, mesh_xy=mesh)

    for ic, component in enumerate(
            (result[0], result[1], result[2], result[3], result[4], result[6])):
        if tuple(component.sharding.spec) != (None, "x", "y"):
            _fail(f"component {ic} layout {component.sharding.spec}")
        for shard in component.addressable_shards:
            want = expected[shard.index]
            got = np.asarray(shard.data)
            if not np.array_equal(got, want):
                _fail(
                    f"component {ic} rank {rank} shard {shard.index}: "
                    f"max|delta|={np.max(np.abs(got - want)):.3e}")
    expected_diag = np.diagonal(expected, axis1=1, axis2=2)
    diagonals = [static_sigma_diag_to_host(component, mesh)
                 for component in (result[1], result[2], result[3], result[6])]
    for ic, diag in enumerate(diagonals):
        if not np.array_equal(diag, expected_diag):
            _fail(f"bounded diagonal {ic} changed values")

    zeros = result[3] - result[3]
    output = GWResults(
        sig_sx=result[1], sig_coh=result[2], sig_h=result[3],
        sig_h_scalar=result[3], h_transverse=zeros, sig_x=result[6],
        E_qp_ry=np.zeros((nk, nb)), U_qp=np.zeros((nk, nb, nb)),
        E_dft_ry=energies, kin_ion_ry=np.zeros((nk, nb, nb)),
        band_start=0, band_stop=nb,
        sig_sx_diag_ry=diagonals[0], sig_coh_diag_ry=diagonals[1],
        sig_h_diag_ry=diagonals[2], sig_h_scalar_diag_ry=diagonals[2],
        h_transverse_diag_ry=np.zeros_like(diagonals[2]),
        sig_x_diag_ry=diagonals[3],
    )
    if not np.array_equal(_result_matrix_diag(output, "sig_h"), expected_diag):
        _fail("GWResults writer seam did not use bounded diagonal")

    cube_source = np.stack((source, source, source), axis=0)
    cube = device_put_process_local(
        cube_source, NamedSharding(mesh, P(None, None, "x", "y")))
    qsgw, _ = build_qsgw_sigma_xc(
        cube, operand, np.array([-1.0, 0.0, 1.0]),
        np.zeros((nk, nb)), mesh, replicated_output=False)
    qsgw_expected = source + np.conj(np.swapaxes(source, -1, -2))
    if tuple(qsgw.sharding.spec) != (None, "x", "y"):
        _fail(f"QSGW fixed-point output layout {qsgw.sharding.spec}")
    for shard in qsgw.addressable_shards:
        if not np.array_equal(np.asarray(shard.data), qsgw_expected[shard.index]):
            _fail("QSGW fixed-point sharded matrix changed values")
    if rank == 0:
        print(
            "[degen-average-p4] PASS: sharding retained; exact matrix; "
            "bounded writer diagonal; sharded QSGW rebuild",
            flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
