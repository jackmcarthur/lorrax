"""Degenerate averaging preserves a band-sharded post-Sigma matrix."""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from gw.degen_average import average_sigma_components, apply_to_matrix_diagonals


def test_band_sharded_helper_keeps_exact_matrix_on_one_device():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(20260831)
    nk, nb = 2, 4
    matrix = (rng.standard_normal((nk, nb, nb))
              + 1j * rng.standard_normal((nk, nb, nb)))
    energies = np.array([[0.0, 0.0, 1.0, 2.0],
                         [0.0, 1.0, 1.0, 3.0]])
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    operand = device_put_process_local(matrix, sharding)

    with mesh:
        out = average_sigma_components(
            operand, operand, operand, operand, operand, None, operand, None,
            energies_kn_ry=energies, tol_ry=1.0e-8, mesh_xy=mesh)

    expected = apply_to_matrix_diagonals(matrix, energies, 1.0e-8)
    for component in (out[0], out[1], out[2], out[3], out[4], out[6]):
        # A size-one mesh canonicalizes every PartitionSpec to replication;
        # the genuine 2x2 layout contract lives in the P=4 gate.
        np.testing.assert_allclose(
            gather_to_host(component), expected, rtol=0.0, atol=0.0)
