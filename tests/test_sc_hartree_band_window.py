"""SC exact Hartree reads ONE band-sharded QP window and never slices it.

VI3 12x12 bispinor density-SC died in map 0 on every rank with
``RESOURCE_EXHAUSTED ... 124.52GiB`` (2026-09-23,
``runs/VI3/09_monolayer_12x12_gw_prep_20260923/11_gnppm_sc_bispinor_ferroU6/
gwjax.log``: the 133707571200 B allocation, and module I/O of the psi(G)
shard plus that output).  ``rebuild_hartree_dft_basis`` loaded the whole
360-band ladder and then did ``psi_G[:, :carrier]`` eagerly on the
``('x','y')``-sharded band axis; the partitioner replicated the result, so
every rank held the whole ``(144, 200, 4, 72541)`` c128 window.

The fix reads only ``[b0, b3)`` (the loader shards it on read, at the
sweep's carrier) and builds rho/J from that window.  These cells pin:

1. the V_H and transverse matrix elements are unchanged by the band cut
   against the incumbent full-ladder recipe (rho over the whole ladder with
   U embedded in the identity, then the window sweep);
2. an occupied tail is refused, a negligible one is admitted;
3. the SAME resident buffer, still band-sharded over all P, reaches both
   the density scan and the sweep: no slice, no replica.

Patched out: the Poisson solve, the TT projection and the symmetry
unfold/projection -- all linear maps downstream of rho/J that the band cut
does not touch -- so the cells compare what the cut can change.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding                   # noqa: E402

from common.wfn_layout import band_sphere_spec                 # noqa: E402

pytestmark = pytest.mark.mesh(4)

NK, NB_FULL, NB_SIGMA, NS, NG = 2, 7, 3, 4, 8
GRID = (3, 4, 2)
VOLUME = 17.0
#: The two recipes partition the band sum differently over the mesh (window
#: carrier 4 = 1 band/device, full carrier 8 = 2 bands/device), so the
#: reductions reassociate; nothing else differs.  Measured values are
#: printed by the cells.
RTOL = 1.0e-12


def _mesh():
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"needs 4 devices, have {len(devs)}")
    return Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))


def _put(array, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        array.shape, sharding, lambda index: array[index])


def _haar(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))[None]


def _case(seed=20260923):
    rng = np.random.default_rng(seed)
    ngrid = int(np.prod(GRID))
    # Full ladder carrier 8 at P4 (7 logical + 1 pad); the QP window is 3
    # logical bands at carrier 4, so BOTH carriers carry a pad band.
    psi_full = np.zeros((NK, 8, NS, NG), dtype=np.complex128)
    psi_full[:, :NB_FULL] = (rng.standard_normal((NK, NB_FULL, NS, NG))
                             + 1j * rng.standard_normal((NK, NB_FULL, NS, NG)))
    # The per-k sphere index (common.gvec_fft_box.build_sphere_box_index):
    # slot g sits in box cell bidx[k, g].
    bidx = np.zeros((NK, NG), dtype=np.int32)
    coords = []
    for ik in range(NK):
        cells = rng.choice(ngrid, size=NG, replace=False)
        xyz = np.column_stack(np.unravel_index(cells, GRID))
        coords.append(xyz)
        bidx[ik] = cells
    occ = np.zeros((NK, 8), dtype=np.float64)      # the bundle carrier
    occ[:, :NB_SIGMA] = [[0.95, 0.55, 0.15], [0.85, 0.35, -0.02]]
    U = np.stack([_haar(rng, NB_SIGMA) for _ in range(NK)])
    return psi_full, bidx, np.asarray(coords, dtype=np.int32), occ, U


class _Recorder:
    """Wrap a module function, keep every psi it was handed."""

    def __init__(self, fn):
        self.fn, self.psi = fn, []

    def __call__(self, psi, *args, **kwargs):
        self.psi.append(psi)
        return self.fn(psi, *args, **kwargs)


def _install(monkeypatch, mesh, psi_window, bidx, coords):
    """Fake SCInputs + linear stand-ins for the unchanged downstream maps."""
    import symmetry_maps
    import psp.dft_operators as dft_operators
    import psp.get_DFT_mtxels as get_DFT_mtxels
    from gw import sc_iteration

    seen = {}
    monkeypatch.setattr(sc_iteration, "_dft_psi_sphere",
                        lambda _inputs: (psi_window, bidx))
    monkeypatch.setattr(
        symmetry_maps, "project_polar_fft_field",
        lambda field, _sym: types.SimpleNamespace(field=field))
    monkeypatch.setattr(symmetry_maps, "unfold_file_wedge_band_operator",
                        lambda _sym, H, trs_rule: H)
    monkeypatch.setattr(dft_operators, "padded_gvectors",
                        lambda _wfn, k: types.SimpleNamespace(
                            gvecs=coords, mask=np.ones((NK, NG)),
                            kvecs=np.zeros((NK, 3))))

    def v_h(rho_r, _wfn, *, truncation_2d, expected_electrons, print_fn):
        seen["expected_electrons"] = float(expected_electrons)
        return 0.7 * jnp.asarray(rho_r)

    monkeypatch.setattr(get_DFT_mtxels, "build_hartree_potential", v_h)
    monkeypatch.setattr(dft_operators, "transverse_potential_from_current",
                        lambda J, *_a, **_k: 0.3 * jnp.asarray(J))
    wfn = types.SimpleNamespace(
        kweights=np.full(NK, 1.0 / NK), occupation_state_capacity=1.0,
        nspinor=2, fft_grid=GRID, cell_volume=VOLUME,
        bdot=np.eye(3), bvec=np.eye(3), blat=1.0)
    sym = types.SimpleNamespace(
        parent_k_domain="ibz", kirr_fullids=np.arange(NK),
        active_symmetry_rows=np.arange(1),
        fft_grid_pullback=lambda _rows, _grid: None)
    inputs = types.SimpleNamespace(
        wfn=wfn, sym=sym, mesh_xy=mesh,
        band_slices=types.SimpleNamespace(nb_sigma=NB_SIGMA),
        wfns_dft=types.SimpleNamespace(enk=np.zeros((NK, 8))),
        config=types.SimpleNamespace(bispinor=True,
                                     bispinor_gw="bare_transverse",
                                     sys_dim=2),
        print_fn=lambda *_a, **_k: None)
    return inputs, seen


def _full_ladder_reference(mesh, psi_full, bidx, coords, occ, U):
    """The incumbent recipe (c17bb176): rho over the WHOLE ladder with U
    embedded in the identity, then the window sweep and the logical strip."""
    from common.mtxel_sweep import (SweepGeometry,
                                    four_current_potential_operator,
                                    sweep_matrix_elements)
    from gw.qsgw_density import rho_from_wfns
    from gw.sc_iteration import _hartree_density_embed_kernel

    U_full = _hartree_density_embed_kernel(mesh, 8)(jnp.asarray(U))
    fields = rho_from_wfns(
        _put(psi_full, mesh, band_sphere_spec()), occ, np.full(NK, 1 / NK),
        U=U_full, mesh=mesh, box_index=bidx, fft_grid=GRID,
        cell_volume=VOLUME, spin_degeneracy=1.0,
        include_dirac_current=True, charge_nspinor=NS)
    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NG, nb=NB_SIGMA,
                         ns=NS, nk=NK, cell_volume=VOLUME)
    H = sweep_matrix_elements(
        _put(np.ascontiguousarray(psi_full[:, :geom.nb]), mesh,
             band_sphere_spec()),
        operator=four_current_potential_operator(
            geom, 0.7 * fields[0], 0.3 * fields[1:], charge_nspinor=NS),
        geom=geom, gvecs=coords, gmask=np.ones((NK, NG)), box_index=bidx,
        kvecs=np.zeros((NK, 3)))
    H = np.asarray(H)[..., :NB_SIGMA, :NB_SIGMA]
    return np.asarray(fields), H[:, 0], H[:, 1]


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want))
                 / max(float(np.max(np.abs(want))), 1e-300))


def test_band_cut_leaves_hartree_and_transverse_matrix_elements_unchanged(
        monkeypatch):
    """Window rho/J and <m|V_H|n>, <m|A.alpha|n> match the full ladder."""
    from gw import qsgw_density, sc_iteration

    mesh = _mesh()
    psi_full, bidx, coords, occ, U = _case()
    ref_fields, ref_scalar, ref_transverse = _full_ladder_reference(
        mesh, psi_full, bidx, coords, occ, U)

    psi_window_np = np.ascontiguousarray(psi_full[:, :4]).copy()
    psi_window_np[:, NB_SIGMA:] = 0.0          # the loader's zero pad band
    psi_window = _put(psi_window_np, mesh, band_sphere_spec())
    inputs, seen = _install(monkeypatch, mesh, psi_window, bidx, coords)
    real_rho, got_fields = qsgw_density.rho_from_wfns, []

    def rho_and_keep(*a, **k):
        out = real_rho(*a, **k)
        got_fields.append(np.asarray(out))
        return out
    monkeypatch.setattr(qsgw_density, "rho_from_wfns", rho_and_keep)

    out = sc_iteration.rebuild_hartree_dft_basis(
        inputs, jnp.asarray(U), jnp.asarray(occ), 0.0)

    field_rel = _rel(got_fields[0], ref_fields)
    scalar_rel = _rel(out.scalar_dft, ref_scalar)
    transverse_rel = _rel(out.transverse_dft, ref_transverse)
    print(f"band-cut deviation: rho/J {field_rel:.3e}, "
          f"V_H {scalar_rel:.3e}, transverse {transverse_rel:.3e}")
    assert out.scalar_dft.shape == out.transverse_dft.shape == (
        NK, NB_SIGMA, NB_SIGMA)
    assert field_rel < RTOL
    assert scalar_rel < RTOL
    assert transverse_rel < RTOL
    # The Poisson electron check integrates the window occupations, which
    # equal the ladder's here (the tail is empty by construction).
    assert np.isclose(seen["expected_electrons"],
                      float(np.mean(occ.sum(axis=1))), rtol=0, atol=1e-14)


def test_occupied_tail_is_refused_and_a_negligible_one_admitted(monkeypatch):
    """The cut is exact only when the tail is empty; the gate says so."""
    from gw import sc_iteration

    mesh = _mesh()
    psi_full, bidx, coords, occ, U = _case()
    psi_window_np = np.ascontiguousarray(psi_full[:, :4]).copy()
    psi_window_np[:, NB_SIGMA:] = 0.0
    psi_window = _put(psi_window_np, mesh, band_sphere_spec())
    inputs, _ = _install(monkeypatch, mesh, psi_window, bidx, coords)

    occupied = occ.copy()
    occupied[1, NB_SIGMA + 1] = 1.0e-6         # a real state above the window
    with pytest.raises(ValueError, match="GATE sc_density_band_cut"):
        sc_iteration.rebuild_hartree_dft_basis(
            inputs, jnp.asarray(U), jnp.asarray(occupied), 0.0)

    fd_tail = occ.copy()
    fd_tail[:, NB_SIGMA:NB_FULL] = 1.0e-17     # FD tail class (bcc Fe)
    out = sc_iteration.rebuild_hartree_dft_basis(
        inputs, jnp.asarray(U), jnp.asarray(fd_tail), 0.0)
    assert np.all(np.isfinite(np.asarray(out.scalar_dft)))


def test_resident_window_reaches_density_and_sweep_unsliced_and_sharded(
        monkeypatch):
    """Both consumers get THE loader buffer: band-sharded over all P."""
    import common.mtxel_sweep as mtxel_sweep
    from gw import qsgw_density, sc_iteration

    mesh = _mesh()
    psi_full, bidx, coords, occ, U = _case()
    psi_window_np = np.ascontiguousarray(psi_full[:, :4]).copy()
    psi_window_np[:, NB_SIGMA:] = 0.0
    psi_window = _put(psi_window_np, mesh, band_sphere_spec())
    inputs, _ = _install(monkeypatch, mesh, psi_window, bidx, coords)
    rho_rec = _Recorder(qsgw_density.rho_from_wfns)
    sweep_rec = _Recorder(mtxel_sweep.sweep_matrix_elements)
    monkeypatch.setattr(qsgw_density, "rho_from_wfns", rho_rec)
    monkeypatch.setattr(mtxel_sweep, "sweep_matrix_elements", sweep_rec)

    sc_iteration.rebuild_hartree_dft_basis(
        inputs, jnp.asarray(U), jnp.asarray(occ), 0.0)

    ndev = mesh.devices.size
    for name, rec in (("rho_from_wfns", rho_rec),
                      ("sweep_matrix_elements", sweep_rec)):
        assert len(rec.psi) == 1, name
        psi = rec.psi[0]
        assert psi is psi_window, f"{name} was handed a copy/slice of psi"
        assert psi.sharding.spec == band_sphere_spec(), name
        per_device = {int(sh.data.nbytes) for sh in psi.addressable_shards}
        assert per_device == {psi.nbytes // ndev}, (name, per_device)


def test_eager_band_slice_of_the_sharded_sphere_is_the_replicating_pattern():
    """Why the fix READS the window instead of slicing a resident ladder.

    Pins the mechanism behind the VI3 OOM at small P: an eager slice of the
    ('x','y')-sharded band axis does not come back band-sharded.  If JAX
    ever partitions it in place this cell fails, and the docstrings that
    cite it (``_dft_psi_sphere``, ``sweep_matrix_elements``) must be revised
    rather than this cell relaxed.
    """
    mesh = _mesh()
    psi_full, *_ = _case()
    psi = _put(psi_full, mesh, band_sphere_spec())
    window = psi[:, :4]
    per_device = max(int(sh.data.nbytes) for sh in window.addressable_shards)
    print(f"eager slice: spec={getattr(window.sharding, 'spec', '?')}, "
          f"per-device {per_device} B of {window.nbytes} B")
    assert per_device > window.nbytes // mesh.devices.size
