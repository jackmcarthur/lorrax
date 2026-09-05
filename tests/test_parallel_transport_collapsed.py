"""Collapsed (vacuum) k axes and short periodic axes in the finite-link
covariant derivative (2026-09-05).

A collapsed reduced axis (one mesh point) has no k derivative; its
connection is the band matrix of the position conjugate to it,
``Z_a = <m| b_a . r |n> = 2 pi <m| f_a |n>``, and the covariant derivative
there is ``-i[Z_a, O]`` exactly.  3- and 4-point axes take a second-order
stencil.  Kernels are the production ones; the real-fixture test checks the
velocity identity ``-i Z_mn (e_n - e_m) = v_mn`` along the MoS2 slab normal
against the exact ``p + i[r, V_NL]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.parallel_transport import (
    COLLAPSED_AXIS, build_forward_neighbor_table, collapsed_axis_coordinate,
    fourth_order_connection, fourth_order_covariant_derivative,
    link_stencil_orders, make_distributed_band_matmul,
)

FIXTURE_DIR = Path(__file__).parent / "regression" / "cohsex_debug"


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _herm(rng, *shape):
    a = _crand(rng, *shape)
    return a + np.conj(np.swapaxes(a, -1, -2))


def test_collapsed_axis_coordinate_is_a_centred_sawtooth():
    grid = (4, 6, 10)
    atoms = np.array([[0.1, 0.2, 0.30], [0.7, 0.9, 0.34], [0.3, 0.5, 0.38]])
    zeta, center = collapsed_axis_coordinate(grid, atoms, 2)
    assert zeta.shape == grid
    assert abs(center - 0.34) < 1e-12
    # constant along the other axes, sawtooth along z
    assert np.max(np.abs(zeta - zeta[:1, :1, :])) == 0.0
    line = zeta[0, 0]
    assert np.all(line > -np.pi) and np.all(line <= np.pi)
    # the cut is half a cell from the centre: f = 0.84 -> wrap -0.5 -> -pi ... +pi
    f = np.arange(10) / 10.0
    want = 2 * np.pi * (np.mod(f - 0.34 + 0.5, 1.0) - 0.5)
    np.testing.assert_allclose(line, want, atol=1e-12)
    # circular mean survives a wrap-around slab (atoms at 0.95 and 0.05)
    _, c2 = collapsed_axis_coordinate(grid, np.array([[0, 0, 0.95], [0, 0, 0.05]]), 2)
    assert abs(c2) < 1e-12 or abs(c2 - 1.0) < 1e-12


def test_collapsed_axis_derivative_is_the_exact_commutator_and_short_axes_are_second_order():
    mesh = _mesh()
    rng = np.random.default_rng(1205)
    grid = (3, 4, 1)                       # second, second, collapsed
    nk = int(np.prod(grid))
    nb = 5
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in grid],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    plus = build_forward_neighbor_table(coords, grid)
    orders = link_stencil_orders(grid)
    assert orders == (2, 2, COLLAPSED_AXIS)
    # identity links: transport is trivial, so the stencil acts on O itself
    links = np.broadcast_to(np.eye(nb, dtype=np.complex128), (3, nk, nb, nb)).copy()
    O = _herm(rng, nk, nb, nb)
    Z = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    Z[2] = _herm(rng, nk, nb, nb)
    spacing = 1.0 / np.asarray(grid, dtype=float)
    band_matmul = make_distributed_band_matmul(mesh, n_batch_axes=1)
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    got = np.asarray(fourth_order_covariant_derivative(
        put(O, P(None, None, None)), put(links, P(None, None, None, None)),
        plus, spacing, band_matmul=band_matmul, stencil_orders=orders,
        collapsed_position=put(Z, P(None, None, None, None))))
    # collapsed axis: exact commutator
    want_z = -1j * (np.einsum("kab,kbc->kac", Z[2], O) - np.einsum("kab,kbc->kac", O, Z[2]))
    np.testing.assert_allclose(got[2], want_z, atol=1e-13)
    # second-order axes: central difference of the operator over +/-1
    from common.parallel_transport import inverse_neighbor_table
    minus = inverse_neighbor_table(plus)
    for d in (0, 1):
        want = (O[plus[:, d]] - O[minus[:, d]]) / (2.0 * spacing[d])
        np.testing.assert_allclose(got[d], want, atol=1e-13)
    # connection: A_a = Z_a on the collapsed axis, Hermitian
    A = np.asarray(fourth_order_connection(
        put(links, P(None, None, None, None)), plus, spacing,
        band_matmul=band_matmul, stencil_orders=orders,
        collapsed_position=put(Z, P(None, None, None, None))))
    np.testing.assert_allclose(A[2], Z[2], atol=1e-13)
    np.testing.assert_allclose(A[0], 0.0, atol=1e-13)      # identity links: no connection
    # refusals
    with pytest.raises(ValueError, match="pt_collapsed_axis_needs_position"):
        fourth_order_covariant_derivative(
            put(O, P(None, None, None)), put(links, P(None, None, None, None)),
            plus, spacing, band_matmul=band_matmul, stencil_orders=orders)
    with pytest.raises(ValueError, match="no axis is collapsed"):
        fourth_order_covariant_derivative(
            put(O, P(None, None, None)), put(links, P(None, None, None, None)),
            plus, spacing, band_matmul=band_matmul, stencil_orders=(2, 2, 2),
            collapsed_position=put(Z, P(None, None, None, None)))


def test_default_call_is_the_historical_fourth_order_stencil():
    """No ``stencil_orders`` = the all-fourth-order kernel, bit for bit."""
    mesh = _mesh()
    rng = np.random.default_rng(7)
    grid = (5, 5, 5)
    nk, nb = 125, 3
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in grid],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    plus = build_forward_neighbor_table(coords, grid)
    links = np.broadcast_to(np.eye(nb, dtype=np.complex128), (3, nk, nb, nb)).copy()
    O = _herm(rng, nk, nb, nb)
    spacing = 1.0 / np.asarray(grid, dtype=float)
    band_matmul = make_distributed_band_matmul(mesh, n_batch_axes=1)
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    a = np.asarray(fourth_order_covariant_derivative(
        put(O, P(None, None, None)), put(links, P(None, None, None, None)),
        plus, spacing, band_matmul=band_matmul))
    b = np.asarray(fourth_order_covariant_derivative(
        put(O, P(None, None, None)), put(links, P(None, None, None, None)),
        plus, spacing, band_matmul=band_matmul, stencil_orders=(4, 4, 4)))
    np.testing.assert_array_equal(a, b)


def test_mos2_slab_position_operator_reproduces_the_exact_z_velocity():
    """On the MoS2 3x3x1 fixture the collapsed axis is z.  The position
    operator sweep must (a) give the identity for zeta = 1 (normalisation),
    (b) be Hermitian, and (c) satisfy ``-i Z_mn (e_n - e_m) / |b_3| =
    v_z,mn`` against the exact ``p + i[r, V_NL]`` on resolved pairs -- the
    velocity identity the production gate checks along that axis."""
    wfn_path = FIXTURE_DIR / "WFNsmall.h5"
    if not wfn_path.exists() or not list(FIXTURE_DIR.glob("*.upf")):
        pytest.skip("cohsex_debug fixture or its pseudopotentials missing")
    from common.mtxel_sweep import (
        SweepGeometry, dipole_operator, local_potential_operator,
        sweep_matrix_elements)
    from common.parallel_transport import collapsed_axes
    from common.wfn_layout import band_sphere_spec
    from psp import vnl_ops
    from psp.dft_operators import padded_gvectors
    from psp.pseudos import load_pseudopotentials
    from wfn_loader import WfnLoader

    mesh = _mesh()
    wfn = WfnLoader(str(wfn_path))
    try:
        grid = tuple(int(n) for n in np.asarray(wfn.kgrid).reshape(3))
        assert collapsed_axes(grid) == (2,), grid
        sym = wfn.symmetry()
        nb = 12
        with mesh:
            gtab = padded_gvectors(wfn, k="ibz")
            psi_G = wfn.load(bands=(0, nb), k="ibz",
                             sharding=band_sphere_spec(), bispinor=False)
            geom = SweepGeometry(
                mesh=mesh, fft_grid=tuple(int(s) for s in wfn.fft_grid),
                ngkmax=int(psi_G.shape[3]), nb=nb, ns=int(psi_G.shape[2]),
                nk=int(psi_G.shape[0]), cell_volume=float(wfn.cell_volume))
            kw = dict(geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask,
                      box_index=wfn.box_index(k="ibz"),
                      kvecs=np.asarray(gtab.kvecs))
            one = np.asarray(sweep_matrix_elements(
                psi_G, operator=local_potential_operator(
                    geom, np.ones(geom.fft_grid)), **kw))[:, :nb, :nb]
            zeta_r, center = collapsed_axis_coordinate(
                geom.fft_grid, wfn.atom_crys, 2)
            Z_sampled = np.asarray(sweep_matrix_elements(
                psi_G, operator=local_potential_operator(geom, zeta_r),
                **kw))[:, :nb, :nb]
            from common.mtxel_sweep import collapsed_position_operator
            Z = np.asarray(sweep_matrix_elements(
                psi_G, operator=collapsed_position_operator(
                    geom, axis=2, center=center), **kw))[:, :nb, :nb]
            # a centre shift by delta inside the vacuum-safe range moves
            # every matrix element by -2 pi delta on the identity
            Z_shift = np.asarray(sweep_matrix_elements(
                psi_G, operator=collapsed_position_operator(
                    geom, axis=2, center=center + 0.01), **kw))[:, :nb, :nb]
            setup = vnl_ops.build_vnl_setup(
                wfn, sym, None, load_pseudopotentials(str(FIXTURE_DIR)),
                nspinor=2, print_fn=lambda *a, **k: None)
            v = np.asarray(sweep_matrix_elements(
                psi_G, operator=dipole_operator(
                    geom, bvec=wfn.bvec, blat=wfn.blat,
                    vnl_setup=setup), **kw))[:, :, :nb, :nb]
        nk = one.shape[0]
        eye = np.broadcast_to(np.eye(nb), (nk, nb, nb))
        np.testing.assert_allclose(one, eye, atol=1e-9)          # (a)
        np.testing.assert_allclose(
            Z, np.conj(np.swapaxes(Z, -1, -2)), atol=1e-9)        # (b)
        np.testing.assert_allclose(
            Z_shift - Z, -2.0 * np.pi * 0.01 * eye, atol=2e-4)  # normalisation
        # the grid-sampled sawtooth is the aliased approximation of the same operator
        assert np.linalg.norm(Z_sampled - Z) / np.linalg.norm(Z) < 5e-2
        e = np.asarray(wfn.energies[0])[:nk, :nb]                 # Ry, IBZ rows
        B = np.asarray(wfn.bvec, dtype=float) * float(wfn.blat)
        assert abs(B[0, 2]) < 1e-12 and abs(B[1, 2]) < 1e-12, "b1, b2 in plane"
        de = e[:, None, :] - e[:, :, None]                        # e_n - e_m
        v_pred = -1j * Z * de / np.linalg.norm(B[2])
        resolved = np.abs(de) > 0.05                              # Ry
        num = np.linalg.norm((v_pred - v[:, 2])[resolved])
        den = np.linalg.norm(v[:, 2][resolved])
        rel = num / den
        # Along the in-plane axes the same construct is NOT the velocity
        # (those axes disperse); the red twin must fail by a wide margin.
        twin = np.linalg.norm((v_pred - v[:, 0])[resolved]) / np.linalg.norm(v[:, 0][resolved])
        v_pred_sampled = -1j * Z_sampled * de / np.linalg.norm(B[2])
        rel_sampled = (np.linalg.norm((v_pred_sampled - v[:, 2])[resolved])
                       / den)
        print(f"z velocity identity: rel L2 {rel:.3e} (grid-sampled sawtooth "
              f"{rel_sampled:.3e}) over {int(resolved.sum())} pairs; x twin {twin:.3e}")
        # The residual is the plane-wave basis's incompleteness in the
        # commutator (-i[z, PHP] vs P(-i[z, H])P), not the position
        # operator: on this 16 Ry fixture it is 1.6 %, and truncating the
        # sphere to 12 / 9 / 6 Ry raises it to 5.3 / 8.2 / 7.5 %
        # (runs/DEV/324_pt_collapsed_axis_20260905/analysis/
        # z_identity_cutoff_trend.txt).  The exact-coefficient and the
        # grid-sampled sawtooth agree to 4 digits here; the exact one is the
        # production operator because it has no grid aliasing to converge.
        assert rel < 3e-2, rel
        assert abs(rel - rel_sampled) < 2e-3
        assert twin > 0.5, twin
    finally:
        wfn.close()


def test_polar_unfold_time_even_differs_from_time_odd_only_on_antiunitary_rows():
    """The position operator is a time-EVEN polar vector; the velocity is
    time-odd.  On the MoS2 fixture (file wedge with time-reversal partners)
    the two unfolds must differ by exactly -1 on the antiunitary rows and
    agree elsewhere; the velocity's sign applied to r flipped Z on those k
    and broke the slab gate on the collapsed axis (2026-09-05)."""
    wfn_path = FIXTURE_DIR / "WFNsmall.h5"
    if not wfn_path.exists():
        pytest.skip("cohsex_debug fixture missing")
    from symmetry_maps import unfold_file_wedge_polar_matrix
    from wfn_loader import WfnLoader
    wfn = WfnLoader(str(wfn_path))
    try:
        sym = wfn.symmetry()
        rng = np.random.default_rng(3)
        nb = 4
        data = _herm(rng, int(sym.nk_red), 3, nb, nb)
        odd = np.asarray(unfold_file_wedge_polar_matrix(sym, data))
        even = np.asarray(unfold_file_wedge_polar_matrix(sym, data, time_odd=False))
        rows = np.asarray(sym.sym_idx_k, dtype=np.int32)
        act_odd = np.asarray(sym.cartesian_action(rows, axial=False, time_odd=True))
        act_even = np.asarray(sym.cartesian_action(rows, axial=False, time_odd=False))
        anti = np.array([not np.allclose(a, b) for a, b in zip(act_odd, act_even)])
        assert anti.any(), "the fixture must carry an antiunitary row for this test"
        assert (~anti).any()
        np.testing.assert_allclose(even[anti], -odd[anti], atol=1e-13)
        np.testing.assert_allclose(even[~anti], odd[~anti], atol=1e-13)
    finally:
        wfn.close()
