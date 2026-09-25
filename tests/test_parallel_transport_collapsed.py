"""Collapsed (vacuum) k axes and short periodic axes in the finite-link
covariant derivative.

A collapsed reduced axis (one mesh point) has no k derivative; its
connection is the band matrix of the position conjugate to it,
``Z_a = <m| b_a . r |n> = 2 pi <m| f_a |n>``, and the covariant derivative
there is ``-i[Z_a, O]`` exactly.  3- and 4-point axes take a second-order
stencil.  The branch cut of the position sawtooth sits at the centre of the
largest vacuum gap.  Kernels are the production ones; the fixture test
checks the velocity identity ``-i Z_mn (e_n - e_m) = v_mn`` along the MoS2
slab normal against the exact ``p + i[r, V_NL]``, its invariance under a
half-cell translation of the slab (the wraparound), and a red twin with the
cut forced through the slab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.parallel_transport import (
    COLLAPSED_AXIS, COLLAPSED_CUT_DENSITY_MAX,
    COLLAPSED_CUT_PROBE_GAP_FRACTION, build_forward_neighbor_table,
    collapsed_axis_center, collapsed_axis_coordinate,
    collapsed_axis_vacuum_gap, fourth_order_connection,
    fourth_order_covariant_derivative, link_stencil_orders,
    make_distributed_band_matmul,
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


def _atoms(*z):
    return np.array([[0.1, 0.2, zi] for zi in z])


def test_branch_cut_is_the_centre_of_the_largest_vacuum_gap():
    # a symmetric slab inside the cell: cut half a cell from its centre
    cut, gap = collapsed_axis_vacuum_gap(_atoms(0.30, 0.34, 0.38), 2)
    assert abs(cut - 0.84) < 1e-12 and abs(gap - 0.92) < 1e-12
    assert abs(collapsed_axis_center(_atoms(0.30, 0.34, 0.38), 2) - 0.34) < 1e-12
    # a slab straddling the cell boundary (atoms at -0.1 and +0.1)
    cut, gap = collapsed_axis_vacuum_gap(_atoms(0.9, 0.1, 0.0), 2)
    assert abs(cut - 0.5) < 1e-12 and abs(gap - 0.8) < 1e-12
    # an asymmetric slab with an adsorbate: the circular mean (0.38) is not
    # the slab centre, and half a cell from it (0.88) is not the gap centre
    atoms = _atoms(0.30, 0.34, 0.38, 0.50)
    cut, gap = collapsed_axis_vacuum_gap(atoms, 2)
    assert abs(cut - 0.90) < 1e-12 and abs(gap - 0.80) < 1e-12
    # two slabs in one cell: the larger of the two vacua
    cut, gap = collapsed_axis_vacuum_gap(_atoms(0.10, 0.15, 0.40, 0.45), 2)
    assert abs(cut - 0.775) < 1e-12 and abs(gap - 0.65) < 1e-12
    # the sampled reference form wraps at the same cut
    zeta, center = collapsed_axis_coordinate((4, 6, 10), atoms, 2)
    assert abs(center - 0.40) < 1e-12
    line = zeta[0, 0]
    f = np.arange(10) / 10.0
    np.testing.assert_allclose(
        line, 2 * np.pi * (np.mod(f - 0.40 + 0.5, 1.0) - 0.5), atol=1e-12)


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
    want_z = -1j * (np.einsum("kab,kbc->kac", Z[2], O) - np.einsum("kab,kbc->kac", O, Z[2]))
    np.testing.assert_allclose(got[2], want_z, atol=1e-13)
    from common.parallel_transport import inverse_neighbor_table
    minus = inverse_neighbor_table(plus)
    for d in (0, 1):
        want = (O[plus[:, d]] - O[minus[:, d]]) / (2.0 * spacing[d])
        np.testing.assert_allclose(got[d], want, atol=1e-13)
    A = np.asarray(fourth_order_connection(
        put(links, P(None, None, None, None)), plus, spacing,
        band_matmul=band_matmul, stencil_orders=orders,
        collapsed_position=put(Z, P(None, None, None, None))))
    np.testing.assert_allclose(A[2], Z[2], atol=1e-13)
    np.testing.assert_allclose(A[0], 0.0, atol=1e-13)
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
    links = _crand(rng, 3, nk, nb, nb)
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
    c = np.asarray(fourth_order_connection(
        put(links, P(None, None, None, None)), plus, spacing,
        band_matmul=band_matmul))
    d = np.asarray(fourth_order_connection(
        put(links, P(None, None, None, None)), plus, spacing,
        band_matmul=band_matmul, stencil_orders=(4, 4, 4)))
    np.testing.assert_array_equal(c, d)


def _fixture_blocks(nb, *, shift=0.0, center_override=None):
    """Z (production sawtooth), the cut-window diagonal, v, e on the MoS2
    fixture's file wedge.  ``shift`` translates the slab by that fraction of
    the cell along z (psi(G) e^{-2 pi i G_z shift}, atoms + shift): an exact
    relabelling.  ``center_override`` forces the sawtooth zero."""
    from common.mtxel_sweep import (
        SweepGeometry, axis_window_operator, collapsed_position_operator,
        dipole_operator, local_potential_operator, sweep_matrix_elements)
    from common.wfn_layout import band_sphere_spec
    from psp import vnl_ops
    from psp.dft_operators import padded_gvectors
    from psp.pseudos import load_pseudopotentials
    from wfn_loader import WfnLoader

    mesh = _mesh()
    wfn = WfnLoader(str(FIXTURE_DIR / "WFNsmall.h5"))
    try:
        sym = wfn.symmetry()
        with mesh:
            gtab = padded_gvectors(wfn, k="ibz")
            psi_G = wfn.load(bands=(0, nb), k="ibz",
                             sharding=band_sphere_spec(), bispinor=False)
            if shift:
                gz = np.asarray(gtab.gvecs)[..., 2].astype(np.float64)
                phase = np.exp(-2j * np.pi * gz * float(shift))
                psi_G = psi_G * jnp.asarray(phase)[:, None, None, :]
            atoms = np.asarray(wfn.atom_crys, dtype=np.float64).copy()
            atoms[:, 2] += float(shift)
            cut, gap = collapsed_axis_vacuum_gap(atoms, 2)
            center = (np.mod(cut + 0.5, 1.0) if center_override is None
                      else float(center_override))
            geom = SweepGeometry(
                mesh=mesh, fft_grid=tuple(int(s) for s in wfn.fft_grid),
                ngkmax=int(psi_G.shape[3]), nb=nb, ns=int(psi_G.shape[2]),
                nk=int(psi_G.shape[0]), cell_volume=float(wfn.cell_volume))
            kw = dict(geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask,
                      box_index=wfn.box_index(k="ibz"),
                      kvecs=np.asarray(gtab.kvecs))
            probe_cut = np.mod(center + 0.5, 1.0)
            Z, win, one = sweep_matrix_elements(
                psi_G, operator=(
                    collapsed_position_operator(geom, axis=2, center=center),
                    axis_window_operator(
                        geom, axis=2, center=probe_cut,
                        width=COLLAPSED_CUT_PROBE_GAP_FRACTION * gap),
                    local_potential_operator(geom, np.ones(geom.fft_grid))),
                **kw)
            Z = np.asarray(Z)[:, :nb, :nb]
            win = np.real(np.diagonal(np.asarray(win), axis1=-2, axis2=-1))[:, :nb]
            one = np.asarray(one)[:, :nb, :nb]
            setup = vnl_ops.build_vnl_setup(
                wfn, sym, None, load_pseudopotentials(str(FIXTURE_DIR)),
                nspinor=2)
            v = np.asarray(sweep_matrix_elements(
                psi_G, operator=dipole_operator(
                    geom, bvec=wfn.bvec, blat=wfn.blat,
                    vnl_setup=setup), **kw))[:, :, :nb, :nb]
        e = np.asarray(wfn.energies)
        e = (e[0] if e.ndim == 3 else e)[:Z.shape[0], :nb]
        occ = np.asarray(wfn.occs)
        occ = (occ[0] if occ.ndim == 3 else occ)[:Z.shape[0], :nb]
        B = np.asarray(wfn.bvec, dtype=float) * float(wfn.blat)
        return dict(Z=Z, win=win, one=one, v=v, e=e, occ=occ, B=B,
                    center=center, gap=gap)
    finally:
        wfn.close()


def _z_identity(blk, *, component=2):
    de = blk["e"][:, None, :] - blk["e"][:, :, None]              # e_n - e_m
    v_pred = -1j * blk["Z"] * de / np.linalg.norm(blk["B"][2])
    resolved = np.abs(de) > 0.05                                   # Ry
    v = blk["v"][:, component]
    return (np.linalg.norm((v_pred - v)[resolved])
            / np.linalg.norm(v[resolved])), int(resolved.sum())


def _need_fixture():
    if (not (FIXTURE_DIR / "WFNsmall.h5").exists()
            or not list(FIXTURE_DIR.glob("*.upf"))):
        pytest.skip("cohsex_debug fixture or its pseudopotentials missing")


def test_mos2_slab_position_operator_reproduces_the_exact_z_velocity():
    """(a) ``<m|1|n>`` is the identity, (b) Z is Hermitian, (c) the slab
    normal's velocity identity ``-i Z_mn (e_n - e_m) / |b_3| = v_z,mn``
    holds against the exact ``p + i[r, V_NL]`` on resolved pairs, (d) the
    in-plane twin fails, and (e) the occupied density at the cut is below
    the refusal threshold."""
    _need_fixture()
    nb = 12
    blk = _fixture_blocks(nb)
    nk = blk["one"].shape[0]
    eye = np.broadcast_to(np.eye(nb), (nk, nb, nb))
    np.testing.assert_allclose(blk["one"], eye, atol=1e-9)
    np.testing.assert_allclose(
        blk["Z"], np.conj(np.swapaxes(blk["Z"], -1, -2)), atol=1e-9)
    rel, npairs = _z_identity(blk)
    twin, _ = _z_identity(blk, component=0)
    occupied = float(np.sum(blk["occ"] * blk["win"]) / np.sum(blk["occ"]))
    print(f"z velocity identity: rel L2 {rel:.3e} over {npairs} pairs; "
          f"x twin {twin:.3e}; occupied density at the cut {occupied:.3e}, "
          f"band max {blk['win'].max():.3e}")
    # The residual is the plane-wave basis's incompleteness in the
    # commutator (-i[z, PHP] vs P(-i[z, H])P), not the position operator:
    # 1.6 % on this 16 Ry fixture, rising as the sphere is truncated.
    assert rel < 3e-2, rel
    assert twin > 0.5, twin
    assert occupied <= COLLAPSED_CUT_DENSITY_MAX, occupied


def test_half_cell_translation_leaves_the_position_operator_unchanged():
    """Wraparound: the same slab translated by half a cell along z (an exact
    relabelling of psi and the atoms) gives the same Z to round-off, so the
    cut placement follows the vacuum, not the cell boundary.  Red twin: the
    cut forced through the slab fails the identity and the density probe."""
    _need_fixture()
    nb = 12
    base = _fixture_blocks(nb)
    moved = _fixture_blocks(nb, shift=0.5)
    scale = np.max(np.abs(base["Z"]))
    assert np.max(np.abs(moved["Z"] - base["Z"])) < 1e-10 * max(scale, 1.0)
    rel_base, _ = _z_identity(base)
    rel_moved, _ = _z_identity(moved)
    assert abs(rel_moved - rel_base) < 1e-10
    red = _fixture_blocks(nb, center_override=np.mod(base["center"] + 0.5, 1.0))
    rel_red, _ = _z_identity(red)
    occupied_red = float(np.sum(red["occ"] * red["win"]) / np.sum(red["occ"]))
    print(f"cut through the slab: identity {rel_red:.3e}, occupied density "
          f"at the cut {occupied_red:.3e}")
    assert rel_red > 0.3, rel_red
    assert occupied_red > 100.0 * COLLAPSED_CUT_DENSITY_MAX, occupied_red


def test_polar_unfold_time_even_differs_from_time_odd_only_on_antiunitary_rows():
    """The position operator is a time-EVEN polar vector; the velocity is
    time-odd.  On the MoS2 fixture (file wedge with time-reversal partners)
    the two unfolds must differ by exactly -1 on the antiunitary rows and
    agree elsewhere."""
    if not (FIXTURE_DIR / "WFNsmall.h5").exists():
        pytest.skip("cohsex_debug fixture missing")
    from symmetry_maps import unfold_file_wedge_polar_matrix
    from wfn_loader import WfnLoader
    wfn = WfnLoader(str(FIXTURE_DIR / "WFNsmall.h5"))
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
