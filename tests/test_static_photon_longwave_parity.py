"""Parity contract for the experimental static photon wing diagnostic.

The production coefficient is fitted from the odd part of a complete
``+q/-q`` shell.  An even finite-shell contribution is spatial dispersion,
not an error in that odd fit.  This test plants both pieces in the packed
response and executes the real sharded projection/fitter at P=1.
"""
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.photon_layout import PhotonBasisLayout
from gw.v_q_bispinor import G0_DIAGONAL_FULL_BZ_V1
from gw.w_isdf import (
    PhotonG0Vectors,
    _nearest_paired_inplane_shell,
    _w_solve_pref_scalar,
    fit_static_photon_longwave_coefficients,
)


def _full_grid_sym(kgrid):
    steps = np.asarray(list(np.ndindex(kgrid)), dtype=np.int64)
    row = {tuple(step): i for i, step in enumerate(steps)}
    negative = np.asarray(
        [row[tuple(np.mod(-step, kgrid))] for step in steps],
        dtype=np.int64,
    )
    return SimpleNamespace(
        nk_tot=int(np.prod(kgrid)),
        kvecs_asints=steps,
        kqfull_map=np.asarray([negative], dtype=np.int64),
    )


def test_odd_wing_fit_does_not_count_even_shell_dispersion_as_error():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    layout = PhotonBasisLayout.from_centroid_extents(1, 1, mesh)
    kgrid = (3, 3, 1)
    sym = _full_grid_sym(kgrid)
    bvec = np.eye(3, dtype=np.float64)
    meta = SimpleNamespace(
        nk_tot=9, nkx=3, nky=3, nkz=1,
        nspin=1, nspinor_wfnfile=2, cell_volume=1.0,
    )
    shell, q_shell, *_ = _nearest_paired_inplane_shell(
        sym, kgrid, bvec)

    # P(q) is Hermitian.  Its CT/TC pair contains an exactly q-linear odd
    # term and an exactly q-quadratic even term on every selected shell row.
    # The direct H+Q diagnostic can represent both.  The wing model is
    # deliberately linear and must compare only with the odd projection.
    response = np.zeros((meta.nk_tot, 4, 4), dtype=np.complex128)
    for index, (qx, qy) in zip(shell, q_shell):
        odd = (0.7 + 0.2j) * qx + (-0.4 + 0.1j) * qy
        even = ((1.3 - 0.2j) * qx * qx
                + (0.5 + 0.4j) * qx * qy
                + (-0.8 + 0.3j) * qy * qy)
        response[index, 1, 0] = odd + even
        response[index, 0, 1] = np.conj(odd + even)

    chi = response / _w_solve_pref_scalar(meta)
    chi = jax.device_put(
        chi, NamedSharding(mesh, P(None, "x", "y")))
    vectors_x = tuple(
        jax.device_put(
            np.ones((meta.nk_tot, 1), dtype=np.complex128),
            NamedSharding(mesh, P(None, "x")),
        )
        for _ in range(4)
    )
    vectors_y = tuple(
        jax.device_put(
            np.ones((meta.nk_tot, 1), dtype=np.complex128),
            NamedSharding(mesh, P(None, "y")),
        )
        for _ in range(4)
    )
    g0 = PhotonG0Vectors(
        x_by_channel=vectors_x,
        y_by_channel=vectors_y,
        provenance=G0_DIAGONAL_FULL_BZ_V1,
    )

    coefficients = fit_static_photon_longwave_coefficients(
        chi, g0, layout, meta, mesh,
        sym=sym,
        bvec_cart_bohr=bvec,
        sys_dim=2,
        material_class="insulator",
    )

    assert coefficients.direct_relative_residual < 2.0e-13
    assert coefficients.Y_relative_residual < 2.0e-13
    assert coefficients.Z_relative_residual < 2.0e-13
    assert coefficients.Y_even_shell_fraction > 0.2
    assert coefficients.Z_even_shell_fraction > 0.2
