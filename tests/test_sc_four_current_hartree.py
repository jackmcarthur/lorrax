"""Focused gates for the evolving-orbital scalar/current Hartree seam."""

from __future__ import annotations

import ast
import dataclasses
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


def test_complex_nontrivial_rotation_matches_explicit_orbitals():
    """The inline scan rotation matches the column-convention reference."""
    rng, psi, bidx, _, grid = _fixture()
    mesh = resolve_mesh()
    occ = np.asarray([[0.85, 0.35, 0.05], [0.70, 0.20, -0.03]])
    weights = np.full(2, 0.5)
    rotations = np.stack([_haar(rng, 3) for _ in range(2)])
    psi_rotated = np.einsum(
        "kmn,kmsg->knsg", rotations, psi, optimize=True)
    kw = dict(mesh=mesh, box_index=bidx, fft_grid=grid,
              cell_volume=17.0, spin_degeneracy=1.0,
              include_dirac_current=True, charge_nspinor=2)
    inline = np.asarray(rho_from_wfns(
        _put(psi, mesh, band_sphere_spec()), occ, weights,
        U=_put(rotations, mesh, band_rotation_spec()), **kw))
    explicit = np.asarray(rho_from_wfns(
        _put(psi_rotated, mesh, band_sphere_spec()), occ, weights, **kw))
    scale = max(float(np.max(np.abs(explicit))), 1.0)
    assert np.max(np.abs(inline - explicit)) < 2.0e-12 * scale


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
    assert {"sig_h", "h_transverse"} <= assigned
    omit_text = ast.get_source_segment(path.read_text(encoding="utf-8"), omit)
    assert "zeros_like(sig_x)" not in omit_text
    assert "jnp.asarray(0, dtype=sig_x.dtype)" in omit_text


def test_live_hartree_addition_is_fused_and_donates_the_dead_base():
    """SC must not retain or materialise V_H+H_T before H assembly."""
    text = (ROOT / "src" / "gw" / "sc_iteration.py").read_text(
        encoding="utf-8")
    assert "v_h_dft_new = exact_hartree_dft.total" not in text
    assert "def _add_exact_four_current_hartree" in text
    module = ast.parse(text)
    fn = next(n for n in module.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_add_exact_four_current_hartree")
    decorator = ast.get_source_segment(text, fn.decorator_list[0])
    assert "donate_argnums=(0,)" in decorator


def test_live_hartree_addition_trims_mesh_padding_at_logical_consumer():
    """A logical SC carry consumes the leading block of a padded sweep."""
    from gw.sc_iteration import _add_exact_four_current_hartree

    base = jnp.arange(2 * 3 * 3, dtype=jnp.float64).reshape(2, 3, 3)
    expected = np.asarray(base) + 1.75
    scalar = jnp.full((2, 4, 4), 2.0, dtype=jnp.float64)
    transverse = jnp.full((2, 4, 4), -0.25, dtype=jnp.float64)
    got = np.asarray(_add_exact_four_current_hartree(
        base, scalar, transverse))
    np.testing.assert_array_equal(got, expected)


def test_live_hartree_addition_rejects_an_undersized_carrier():
    """Mesh padding may enlarge a sweep carrier; it may never lose bands."""
    from gw.sc_iteration import _add_exact_four_current_hartree

    base = jnp.zeros((2, 4, 4), dtype=jnp.float64)
    scalar = jnp.zeros((2, 3, 3), dtype=jnp.float64)
    transverse = jnp.zeros((2, 4, 4), dtype=jnp.float64)
    with pytest.raises(ValueError, match="no wider than"):
        _add_exact_four_current_hartree(base, scalar, transverse)


def test_hartree_omission_receipt_restores_full_output_matrices():
    """The compact internal sentinel cannot masquerade as a final field."""
    from gw.sigma_dispatch import SigmaResult

    matrix = jnp.ones((2, 4, 4), dtype=jnp.complex128)
    omitted = SigmaResult(
        v_h_kij_ry=jnp.asarray(0, dtype=jnp.complex128),
        v_h_scalar_kij_ry=jnp.asarray(0, dtype=jnp.complex128),
        hartree_omitted=True,
        sigma_x_kij_ry=-matrix,
        sigma_xc_kij_ry=-0.5 * matrix)
    restored = dataclasses.replace(
        omitted, v_h_kij_ry=2 * matrix, v_h_scalar_kij_ry=matrix,
        h_transverse_kij_ry=matrix, hartree_omitted=False)
    assert not restored.hartree_omitted
    assert restored.v_h_kij_ry.shape == matrix.shape
    assert restored.v_h_scalar_kij_ry.shape == matrix.shape
    assert restored.h_transverse_kij_ry.shape == matrix.shape
    with pytest.raises(ValueError, match="scalar-zero sentinel"):
        dataclasses.replace(omitted, v_h_kij_ry=matrix)


def test_live_hartree_is_carried_to_both_final_output_seams():
    """The final SigmaResult and sigma_mnk writer both receive the live field."""
    text = (ROOT / "src" / "gw" / "sc_iteration.py").read_text(
        encoding="utf-8")
    assert "exact_hartree_dft=exact_hartree_dft" in text
    assert "exact_hartree_dft=state_final.outputs.exact_hartree_dft" in text
    assert "v_h_scalar = exact_hartree_dft.scalar_dft" in text
    assert "h_transverse = exact_hartree_dft.transverse_dft" in text
    assert "sig_h = exact_hartree_dft.total" in text


def test_sc_density_applies_rotation_inside_the_scan():
    """SC must not materialise a full resident QP-wavefunction array."""
    module = ast.parse((ROOT / "src" / "gw" / "sc_iteration.py").read_text(
        encoding="utf-8"))
    fn = next(n for n in module.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "rebuild_hartree_dft_basis")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "rho_from_wfns"]
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "psi_G"
    rotation = next(kw for kw in calls[0].keywords if kw.arg == "U")
    assert isinstance(rotation.value, ast.Name)
    assert rotation.value.id == "U_qp"
    assert not any(isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name)
                   and n.func.id == "rotate_bands"
                   for n in ast.walk(fn))


def test_density_scan_reshards_only_the_singleton_k_slice():
    """Resident psi stays band-xy; only psi[k] is placed m-on-x."""
    module = ast.parse((ROOT / "src" / "gw" / "qsgw_density.py").read_text(
        encoding="utf-8"))
    fn = next(n for n in module.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "rho_from_wfns")
    constraints = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "with_sharding_constraint"
        and len(n.args) >= 2
    ]

    def _named_arg(call, value, sharding):
        return (isinstance(call.args[0], ast.Name)
                and call.args[0].id == value
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == sharding)

    assert sum(_named_arg(call, "psi_", "band_xy")
               for call in constraints) == 1
    assert not any(_named_arg(call, "psi_", "m_on_x")
                   for call in constraints)
    slice_reshards = [
        call for call in constraints
        if isinstance(call.args[0], ast.Subscript)
        and isinstance(call.args[0].value, ast.Name)
        and call.args[0].value.id == "psi_k"
        and isinstance(call.args[0].slice, ast.Constant)
        and call.args[0].slice.value is None
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "m_on_x"
    ]
    assert len(slice_reshards) == 1
    rotation_einsums = [
        call for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "einsum"
        and any(isinstance(arg, ast.Name) and arg.id == "psi_k_x"
                for arg in call.args)
    ]
    assert len(rotation_einsums) == 1


def test_live_gspace_hartree_cannot_cross_the_host_gather_boundary():
    """The sole GW caller requests a sharded device result."""
    dispatch = ast.parse((ROOT / "src" / "gw" / "sigma_dispatch.py").read_text(
        encoding="utf-8"))
    resolve = next(n for n in dispatch.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "_compute_live_hartree")
    calls = [n for n in ast.walk(resolve)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "compute_hartree_matrix"]
    assert len(calls) == 1
    keyword = next((kw for kw in calls[0].keywords
                    if kw.arg == "return_sharded"), None)
    assert keyword is not None
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
