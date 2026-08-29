"""Focused gates for the evolving-orbital scalar/current Hartree seam."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import NamedSharding

from common.collectives import resolve_mesh
from common.mtxel_sweep import (
    SweepGeometry,
    four_current_potential_operator,
    local_potential_operator,
    sweep_matrix_elements,
)
from common.wfn_layout import band_sphere_spec
from gw.qsgw_density import band_rotation_spec, rho_from_wfns
from psp.get_DFT_mtxels import valence_density_from_kpoint


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _haar(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))[None]


def _fixture():
    rng = np.random.default_rng(20260829)
    nk, nb, ns, ng = 2, 3, 4, 8
    grid = (3, 4, 2)
    ngrid = int(np.prod(grid))
    psi = (rng.standard_normal((nk, nb, ns, ng))
           + 1j * rng.standard_normal((nk, nb, ns, ng)))
    bidx = np.full((nk, *grid), ng, dtype=np.int32)
    coords = []
    for ik in range(nk):
        cells = rng.choice(ngrid, size=ng, replace=False)
        xyz = np.column_stack(np.unravel_index(cells, grid))
        coords.append(xyz)
        bidx[ik, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = np.arange(ng)
    return rng, psi.astype(np.complex128), bidx, coords, grid


def _put(array, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        array.shape, sharding, lambda index: array[index])


def test_qsgw_four_current_matches_the_shared_per_k_kernel():
    """Finite signed occupations weight rho and J identically in both plans."""
    _, psi, bidx, coords, grid = _fixture()
    mesh = resolve_mesh()
    occ = np.asarray([[0.75, 0.30, -0.05],
                      [0.65, 0.20, 0.00]], dtype=np.float64)
    weights = np.full(2, 0.5, dtype=np.float64)
    volume = 17.0
    got = np.asarray(rho_from_wfns(
        _put(psi, mesh, band_sphere_spec()), occ, weights,
        mesh=mesh, box_index=bidx, fft_grid=grid,
        cell_volume=volume, spin_degeneracy=1.0,
        include_dirac_current=True, charge_nspinor=2))

    expected = np.zeros((4, *grid), dtype=np.float64)
    for ik, xyz in enumerate(coords):
        box = np.zeros((3, 4, *grid), dtype=np.complex128)
        box[:, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = psi[ik]
        expected += np.asarray(valence_density_from_kpoint(
            jnp.asarray(box), nocc=None, weight=weights[ik],
            cell_volume=volume, spin_degeneracy=1.0,
            band_occupations=occ[ik], include_dirac_current=True,
            charge_nspinor=2))
    scale = max(float(np.max(np.abs(expected))), 1.0)
    assert np.max(np.abs(got - expected)) < 2.0e-12 * scale


def test_equal_occupation_unitary_preserves_the_whole_four_current():
    """A degenerate occupied gauge rotates neither charge nor spatial J."""
    rng, psi, bidx, _, grid = _fixture()
    mesh = resolve_mesh()
    occ = np.asarray([[0.7, 0.7, 0.0], [0.7, 0.7, 0.0]])
    weights = np.full(2, 0.5)
    psi_j = _put(psi, mesh, band_sphere_spec())
    kw = dict(mesh=mesh, box_index=bidx, fft_grid=grid,
              cell_volume=17.0, spin_degeneracy=1.0,
              include_dirac_current=True, charge_nspinor=2)
    baseline = np.asarray(rho_from_wfns(psi_j, occ, weights, **kw))
    rotations = np.stack([np.eye(3, dtype=np.complex128) for _ in range(2)])
    for ik in range(2):
        rotations[ik, :2, :2] = _haar(rng, 2)
    rotated = np.asarray(rho_from_wfns(
        psi_j, occ, weights,
        U=_put(rotations, mesh, band_rotation_spec()), **kw))
    scale = max(float(np.max(np.abs(baseline))), 1.0)
    assert np.max(np.abs(rotated - baseline)) < 2.0e-12 * scale


def test_packed_four_current_matrix_sweep_matches_two_independent_sweeps():
    """Packing shares FFT/reshard work without changing either component."""
    rng, psi, bidx, coords, grid = _fixture()
    mesh = resolve_mesh()
    nk, nb, _, ng = psi.shape
    volume = 17.0
    scalar = rng.standard_normal(grid)
    vector = rng.standard_normal((3, *grid))
    gvecs = np.asarray(coords, dtype=np.int32)
    gmask = np.ones((nk, ng), dtype=np.float64)
    kvecs = np.zeros((nk, 3), dtype=np.float64)

    geom4 = SweepGeometry(
        mesh=mesh, fft_grid=grid, ngkmax=ng, nb=nb, ns=4, nk=nk,
        cell_volume=volume)
    packed = sweep_matrix_elements(
        _put(psi, mesh, band_sphere_spec()),
        operator=four_current_potential_operator(
            geom4, scalar, vector, charge_nspinor=2),
        geom=geom4, gvecs=gvecs, gmask=gmask, box_index=bidx,
        kvecs=kvecs)

    geom2 = SweepGeometry(
        mesh=mesh, fft_grid=grid, ngkmax=ng, nb=nb, ns=2, nk=nk,
        cell_volume=volume)
    scalar_ref = sweep_matrix_elements(
        _put(psi[:, :, :2], mesh, band_sphere_spec()),
        operator=local_potential_operator(geom2, scalar), geom=geom2,
        gvecs=gvecs, gmask=gmask, box_index=bidx, kvecs=kvecs)
    vector_ref = sweep_matrix_elements(
        _put(psi, mesh, band_sphere_spec()),
        operator=local_potential_operator(geom4, vector, dirac_vector=True),
        geom=geom4, gvecs=gvecs, gmask=gmask, box_index=bidx,
        kvecs=kvecs)
    for got, ref in ((packed[:, 0], scalar_ref),
                     (packed[:, 1], vector_ref)):
        got_np, ref_np = np.asarray(got), np.asarray(ref)
        scale = max(float(np.max(np.abs(ref_np))), 1.0)
        assert np.max(np.abs(got_np - ref_np)) < 2.0e-12 * scale


def test_exact_hartree_basis_rotation_retains_two_axis_band_sharding():
    """The gspace replacement must not replicate while rotating into QP."""
    from gw.sigma_dispatch import _rotate_v_h_to_qp

    rng = np.random.default_rng(2026082902)
    mesh = resolve_mesh()
    nk, nb = 2, 4
    raw = (rng.standard_normal((nk, nb, nb))
           + 1j * rng.standard_normal((nk, nb, nb)))
    v_h = raw + np.swapaxes(np.conj(raw), -1, -2)
    U = np.stack([_haar(rng, nb) for _ in range(nk)])
    got = _rotate_v_h_to_qp(
        _put(v_h, mesh, band_rotation_spec()),
        _put(U, mesh, band_rotation_spec()), mesh=mesh)
    want = np.einsum("kmi,kmn,knj->kij", np.conj(U), v_h, U,
                     optimize=True)
    assert got.sharding.spec == band_rotation_spec()
    scale = max(float(np.max(np.abs(want))), 1.0)
    assert np.max(np.abs(np.asarray(got) - want)) < 2.0e-12 * scale


def test_density_sc_suppresses_both_frozen_direct_components():
    """The caller-owned four-current cannot coexist with frozen H_T."""
    path = ROOT / "src" / "gw" / "sigma_dispatch.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in module.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "compute_sigma_xc")
    omit = next(n for n in ast.walk(fn)
                if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name)
                and n.test.id == "omit_v_h")
    assigned = {
        target.id
        for node in ast.walk(omit)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    # Tuple assignment is the production spelling; inspect its names too.
    for node in ast.walk(omit):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    assigned.update(
                        elt.id for elt in target.elts if isinstance(elt, ast.Name))
    assert {"sig_h", "v_h_ext", "h_transverse"} <= assigned


def test_live_hartree_is_carried_to_both_final_output_seams():
    """The final SigmaResult and sigma_mnk writer both receive the live field."""
    text = (ROOT / "src" / "gw" / "sc_iteration.py").read_text(
        encoding="utf-8")
    assert "exact_hartree_dft=exact_hartree_dft" in text
    assert "exact_hartree_dft=state_final.outputs.exact_hartree_dft" in text
    assert "v_h_scalar = exact_hartree_dft.scalar_dft" in text
    assert "h_transverse = exact_hartree_dft.transverse_dft" in text
    assert "sig_h = exact_hartree_dft.total" in text


def test_live_gspace_hartree_cannot_cross_the_host_gather_boundary():
    """Driver requests a device result; only the legacy arm may gather it."""
    dispatch = ast.parse((ROOT / "src" / "gw" / "sigma_dispatch.py").read_text(
        encoding="utf-8"))
    resolve = next(n for n in dispatch.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "resolve_external_hartree")
    calls = [n for n in ast.walk(resolve)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "compute_hartree_matrix"]
    assert len(calls) == 1
    keyword = next((kw for kw in calls[0].keywords
                    if kw.arg == "return_sharded"), None)
    assert keyword is not None
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True

    driver = ast.parse((ROOT / "src" / "gw" / "kin_ion_io.py").read_text(
        encoding="utf-8"))
    compute = next(n for n in driver.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "compute_hartree_matrix")
    boundaries = [
        n for n in ast.walk(compute)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "return_sharded"
    ]
    assert len(boundaries) == 2  # scalar V_H and transverse alpha.A
    for boundary in boundaries:
        live_calls = [n for stmt in boundary.body for n in ast.walk(stmt)
                      if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name)
                      and n.func.id == "blocks_to_host"]
        legacy_calls = [n for stmt in boundary.orelse for n in ast.walk(stmt)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == "blocks_to_host"]
        assert not live_calls
        assert len(legacy_calls) == 1
