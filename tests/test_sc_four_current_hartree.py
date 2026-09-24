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
    nk, nb, ns, ng = 2, 4, 4, 8
    grid = (3, 4, 2)
    ngrid = int(np.prod(grid))
    psi = (rng.standard_normal((nk, nb, ns, ng))
           + 1j * rng.standard_normal((nk, nb, ns, ng)))
    # The per-k sphere index (common.gvec_fft_box.build_sphere_box_index):
    # slot g sits in box cell bidx[k, g].
    bidx = np.zeros((nk, ng), dtype=np.int32)
    coords = []
    for ik in range(nk):
        cells = rng.choice(ngrid, size=ng, replace=False)
        xyz = np.column_stack(np.unravel_index(cells, grid))
        coords.append(xyz)
        bidx[ik] = cells
    return rng, psi.astype(np.complex128), bidx, coords, grid


def _put(array, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        array.shape, sharding, lambda index: array[index])


def test_qsgw_four_current_matches_the_shared_per_k_kernel():
    """Finite signed occupations weight rho and J identically in both plans."""
    _, psi, bidx, coords, grid = _fixture()
    mesh = resolve_mesh()
    occ = np.asarray([[0.75, 0.30, -0.05, 0.0],
                      [0.65, 0.20, 0.00, 0.0]], dtype=np.float64)
    weights = np.full(2, 0.5, dtype=np.float64)
    volume = 17.0
    got = np.asarray(rho_from_wfns(
        _put(psi, mesh, band_sphere_spec()), occ, weights,
        mesh=mesh, box_index=bidx, fft_grid=grid,
        cell_volume=volume, spin_degeneracy=1.0,
        include_dirac_current=True, charge_nspinor=2))

    expected = np.zeros((4, *grid), dtype=np.float64)
    for ik, xyz in enumerate(coords):
        box = np.zeros((4, 4, *grid), dtype=np.complex128)
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
    occ = np.asarray([[0.7, 0.7, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]])
    weights = np.full(2, 0.5)
    psi_j = _put(psi, mesh, band_sphere_spec())
    kw = dict(mesh=mesh, box_index=bidx, fft_grid=grid,
              cell_volume=17.0, spin_degeneracy=1.0,
              include_dirac_current=True, charge_nspinor=2)
    baseline = np.asarray(rho_from_wfns(psi_j, occ, weights, **kw))
    rotations = np.stack([np.eye(4, dtype=np.complex128) for _ in range(2)])
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
    occ = np.asarray([[0.85, 0.35, 0.05, 0.0], [0.70, 0.20, -0.03, 0.0]])
    weights = np.full(2, 0.5)
    rotations = np.stack([_haar(rng, 4) for _ in range(2)])
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


@pytest.mark.parametrize("transverse", [False, True])
def test_final_hartree_writer_uses_sigma_band_carrier(monkeypatch, transverse):
    """Uneven physical bands retain U† V U and zero padding on P16."""
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from gw import dynamic_sigma, qsgw_utils
    from gw.sc_iteration import SCExactHartree, dump_sigma_omega_h5_final
    from gw.sigma_dispatch import SigmaResult
    from runtime.padding import padded_axis

    mesh = resolve_mesh()
    rng = np.random.default_rng(20260921)
    nk, nb = 2, 26
    tag = padded_axis(nb, mesh, name="Sigma bands")
    U = np.stack([_haar(rng, nb) for _ in range(nk)])
    raw = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb))
    scalar = raw + raw.conj().swapaxes(-1, -2)
    current = 0.17 * scalar if transverse else None
    replicated = P(None, None, None)
    exact = SCExactHartree(_put(scalar, mesh, replicated),
        None if current is None else _put(current, mesh, replicated), 0.0)
    sigma = SigmaResult(v_h_kij_ry=jnp.asarray(0.0),
        sigma_x_kij_ry=jnp.asarray(0.0), sigma_xc_kij_ry=jnp.asarray(0.0),
        sigma_c_omega_kij_ry=jnp.asarray(0.0),
        omega_grid_ev=np.array([-1.0, 1.0]), sigma_band_axis=tag)
    config = dataclasses.make_dataclass("Config", [("sc_omega_grid_ev", tuple)])(())
    captured = {}
    def write(_cube, **kwargs):
        captured.update(kwargs)
        return "unused.h5"
    monkeypatch.setattr(dynamic_sigma, "write_sigma_omega", write)
    monkeypatch.setattr(qsgw_utils, "write_qsgw_sigma_cube", lambda *a, **kw: None)
    with mesh:
        dump_sigma_omega_h5_final(
            SimpleNamespace(outputs=SimpleNamespace(sigma_result=sigma)),
            config=config, meta=None, mesh_xy=mesh, input_dir="unused",
            exact_hartree_dft=exact, sigma_basis_U=_put(U, mesh, replicated),
            print_fn=lambda *_: None)
    expected = np.zeros((nk, tag.carrier, tag.carrier), dtype=np.complex128)
    expected[:, :nb, :nb] = np.einsum("kmi,kmn,knj->kij", U.conj(), scalar, U)
    for name, factor in (("v_h_scalar", 1.0), ("sig_h", 1.17 if transverse else 1.0)):
        got = captured[name]
        assert got.sharding.spec == band_rotation_spec()
        for shard in got.addressable_shards:
            np.testing.assert_allclose(shard.data, factor * expected[shard.index],
                                       rtol=2e-12, atol=2e-12)
    if transverse:
        for shard in captured["h_transverse"].addressable_shards:
            np.testing.assert_allclose(shard.data, 0.17 * expected[shard.index],
                                       rtol=2e-12, atol=2e-12)
    else:
        assert captured["h_transverse"] is None


def test_density_sc_suppresses_both_frozen_direct_components():
    """The caller-owned four-current cannot coexist with frozen H_T."""
    path = ROOT / "src" / "gw" / "sigma_dispatch.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in module.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_sigma_hartree_fields")
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


def test_density_scan_slices_k_inside_the_manual_region():
    """The k slice happens inside shard_map, where GSPMD cannot hoist it.

    The previous auto-sharded scan constrained ``psi[k]`` to m-on-x, and the
    partitioner propagated that constraint to the scan OPERAND: every trip
    all-gathered the whole ``(n_k, nb/P, s, G)`` stack, 16 GB/rank/trip at
    VI3 12x12 P16 (``runs/runtime/density_scan_20260923``, HLO census).  The
    body is now manual-axis code: one scan, one psum after it, and no
    sharding constraint for the partitioner to propagate.  The compiled
    census is ``tests/test_qsgw_density_scan.py``.
    """
    module = ast.parse((ROOT / "src" / "gw" / "qsgw_density.py").read_text(
        encoding="utf-8"))
    fns = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}

    def calls(fn, attr):
        return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Attribute, ast.Name))
                and getattr(n.func, "attr", getattr(n.func, "id", "")) == attr]

    body = fns["_density_scan_body"]
    assert len(calls(body, "scan")) == 1
    assert len(calls(body, "psum")) == 1
    assert not calls(body, "with_sharding_constraint")
    rho = fns["rho_from_wfns"]
    sm = calls(rho, "shard_map")
    assert len(sm) == 1
    assert all(not isinstance(c.args[0], ast.Name) or c.args[0].id != "psi"
               for c in calls(rho, "with_sharding_constraint"))


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


def test_reduced_current_requires_the_typed_vector_action():
    """Scalar FFT pullbacks cannot substitute for the polar current action."""
    with pytest.raises(ValueError, match="requires SymMaps"):
        rho_from_wfns(
            jnp.zeros((2, 4, 4, 1), dtype=jnp.complex128),
            np.ones((2, 4)), np.asarray([0.25, 0.75]),
            mesh=None, box_index=np.zeros((2, 1, 1, 1), dtype=np.int32),
            fft_grid=(1, 1, 1), cell_volume=1.0, spin_degeneracy=1.0,
            sym_perm=np.zeros((1, 1), dtype=np.int32),
            include_dirac_current=True, charge_nspinor=2)
