"""Nested-parity contract for the no-pair static photon completion.

The production coefficients are extrapolated from the odd/even parts of a
symmetry-owned ``+q/-q`` radial union.  This test plants leading and
next-order terms simultaneously and executes the real sharded
projection/fitter at P=1.
"""
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.photon_layout import PhotonBasisLayout
from gw.v_q_bispinor import G0_DIAGONAL_FULL_BZ_V1
from gw.w_isdf import (
    PhotonG0Vectors,
    _nested_paired_inplane_shell,
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
    kgrid = (5, 5, 1)
    sym = _full_grid_sym(kgrid)
    bvec = np.eye(3, dtype=np.float64)
    meta = SimpleNamespace(
        nk_tot=25, nkx=5, nky=5, nkz=1,
        nspin=1, nspinor_wfnfile=2, cell_volume=1.0,
    )
    shell, q_shell, *_ = _nested_paired_inplane_shell(
        sym, kgrid, bvec)

    # P(q) is Hermitian.  Its CT/TC pair contains q+q^3 odd terms and q^2+q^4
    # even terms.  The diagonal TT block contains q^2+q^4 after the declared
    # uniform-A subtraction.  A leading-only one-shell fit aliases the higher
    # orders; the production nested fit must recover the planted q and q^2
    # coefficients instead.
    response = np.zeros((meta.nk_tot, 4, 4), dtype=np.complex128)
    linear_x = 0.7 + 0.2j
    linear_y = -0.4 + 0.1j
    tt_xx = 1.1
    tt_xy = -0.3
    tt_yy = 0.8
    for index, (qx, qy) in zip(shell, q_shell):
        odd = (linear_x * qx + linear_y * qy
               + (2.3 - 0.2j) * qx ** 3
               + (-1.7 + 0.6j) * qx ** 2 * qy
               + (0.9 + 0.4j) * qx * qy ** 2
               + (-1.1 - 0.3j) * qy ** 3)
        even = ((1.3 - 0.2j) * qx ** 2
                + (0.5 + 0.4j) * qx * qy
                + (-0.8 + 0.3j) * qy ** 2
                + (1.4 - 0.5j) * qx ** 4
                + (-0.7 + 0.2j) * qx ** 3 * qy
                + (0.6 + 0.1j) * qx ** 2 * qy ** 2
                + (-0.4 + 0.3j) * qx * qy ** 3
                + (0.8 - 0.6j) * qy ** 4)
        response[index, 1, 0] = odd + even
        response[index, 0, 1] = np.conj(odd + even)
        response[index, 1, 1] = (
            tt_xx * qx ** 2 + 2.0 * tt_xy * qx * qy + tt_yy * qy ** 2
            + 1.6 * qx ** 4 - 0.9 * qx ** 3 * qy
            + 0.5 * qx ** 2 * qy ** 2 + 0.7 * qx * qy ** 3
            - 1.2 * qy ** 4)

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

    # The Voronoi service may split equal FFT-index radii on nontrivial cell
    # metrics; the contract is the attained polynomial ranks, not a hard-coded
    # number of geometric levels.
    assert coefficients.radial_shell_count >= 4
    assert coefficients.odd_extrapolation_rank == 6
    assert coefficients.even_extrapolation_rank == 8
    np.testing.assert_allclose(
        np.asarray(coefficients.H_direct[:, 1, 0]),
        np.asarray((linear_x, linear_y)), rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        np.asarray(coefficients.Q_direct[:, :, 1, 1]),
        np.asarray(((tt_xx, tt_xy), (tt_xy, tt_yy))),
        rtol=2.0e-12, atol=2.0e-12)
    assert coefficients.direct_relative_residual < 2.0e-12
    assert coefficients.Y_relative_residual < 2.0e-12
    assert coefficients.Z_relative_residual < 2.0e-12
    assert coefficients.Y_even_shell_fraction > 0.2
    assert coefficients.Z_even_shell_fraction > 0.2
    assert coefficients.hall_topological_source.startswith(
        "no_external_hall_term")


def test_nested_no_pair_completion_refuses_an_underdetermined_mesh():
    kgrid = (3, 3, 1)
    sym = _full_grid_sym(kgrid)
    with np.testing.assert_raises_regex(
            ValueError,
            r"odd\(q\+q\^3\) rank=4/6, even\(q\^2\+q\^4\) rank=4/8"):
        _nested_paired_inplane_shell(sym, kgrid, np.eye(3))


def test_cri3_6x6_hexagonal_geometry_supplies_the_nested_design():
    # Small metadata read from the immutable CrI3 WFN used by the corrected
    # full arm.  This cell proves that arm has the runtime q geometry required
    # by the rank-6/rank-8 completion; it does not launch the material run.
    kgrid = (6, 6, 1)
    bvec = np.asarray(
        ((1.0, -0.5773502691896258, 0.0),
         (0.0, 1.1547005383792517, 0.0),
         (0.0, 0.0, 0.302908)),
        dtype=np.float64,
    )
    result = _nested_paired_inplane_shell(
        _full_grid_sym(kgrid), kgrid, bvec)
    shell, _, _, _, _, _, pair_first, _, odd_rank, even_rank, nlevels = result
    assert shell.size == 2 * pair_first.size
    assert nlevels >= 2
    assert odd_rank == 6
    assert even_rank == 8
