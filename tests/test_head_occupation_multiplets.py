"""The SC head occupations must be one value per exactly degenerate multiplet.

Planted case (SCMETAL 2026-09-17): on a metal the frozen DFT head response
carried step occupations by band index.  Where the index cut runs through an
exact multiplet, the static head gives its members different Sigma_x head
values, which broke the little group and inversion times time reversal of the
next map (Na bcc Fermi-Dirac: 1.3207 eV splits at three k, bank reciprocity
1e-13 -> 1e-4, shared_pole_gram_valid refused at q=0).
"""
import numpy as np
import pytest

from gw.efermi import fd_occupations
from gw.head_correction import (
    HEAD_MULTIPLET_OCCUPATION_TOL,
    compute_static_head_terms,
    refuse_split_multiplet_head_occupations,
)

KT_RY = 0.01


def _ladder():
    """Three k, six bands; k=1 has an exact doublet at bands 3,4 (1-based) on E_F."""
    e = np.array([[-1.0, -0.40, -0.10, 0.30, 0.60, 0.90],
                  [-1.0, -0.40, 0.05, 0.05, 0.60, 0.90],
                  [-1.0, -0.50, -0.20, 0.40, 0.70, 0.95]])
    return e, 0.05


def _index_step(e, n_occ):
    occ = np.zeros_like(e)
    occ[:, :n_occ] = 1.0
    return occ


def test_index_step_through_a_multiplet_refuses_and_names_it():
    e, _ = _ladder()
    with pytest.raises(ValueError, match="GATE head_occupations_split_multiplet") as err:
        refuse_split_multiplet_head_occupations(e, _index_step(e, 3), where="planted")
    assert "k=1, bands 3 and 4" in str(err.value)


def test_the_planted_step_splits_the_static_head_inside_the_multiplet():
    """The consequence the gate protects: Sigma_x head differs inside the doublet."""
    e, _ = _ladder()
    step = compute_static_head_terms(vc0=6000.0, wcoul0_static=10.0, occ=_index_step(e, 3),
                                     cell_volume=254.0, nk_tot=512)
    x = np.real(np.asarray(step.sigma_x_diag))
    assert abs(x[1, 2] - x[1, 3]) > 1.0e-2
    fd = compute_static_head_terms(vc0=6000.0, wcoul0_static=10.0,
                                   occ=np.asarray(fd_occupations(e, 0.05, KT_RY)),
                                   cell_volume=254.0, nk_tot=512)
    x = np.real(np.asarray(fd.sigma_x_diag))
    assert x[1, 2] == x[1, 3]


def test_fermi_dirac_state_passes():
    e, mu = _ladder()
    f = np.asarray(fd_occupations(e, mu, KT_RY))
    refuse_split_multiplet_head_occupations(e, f, where="fd")


def test_fermi_dirac_inside_the_degeneracy_tolerance_passes():
    e, mu = _ladder()
    e = e.copy()
    e[1, 3] = e[1, 2] + 0.9e-9      # inside the tolerance, on E_F
    f = np.asarray(fd_occupations(e, mu, KT_RY))
    assert abs(f[1, 2] - f[1, 3]) < HEAD_MULTIPLET_OCCUPATION_TOL
    refuse_split_multiplet_head_occupations(e, f, where="fd-near")


def test_insulator_step_at_a_gap_passes():
    e = np.array([[-1.0, -0.5, -0.5, 0.4, 0.4, 0.9],
                  [-1.1, -0.6, -0.6, 0.5, 0.5, 1.0]])
    refuse_split_multiplet_head_occupations(e, (e < 0).astype(float), where="gap")


def test_resolved_pair_is_not_a_multiplet():
    e, _ = _ladder()
    e = e.copy()
    e[1, 3] = e[1, 2] + 1.0e-6      # resolved: not one multiplet
    refuse_split_multiplet_head_occupations(e, _index_step(e, 3), where="resolved")


def test_shape_mismatch_refuses():
    e, _ = _ladder()
    with pytest.raises(ValueError, match="GATE head_occupations_split_multiplet"):
        refuse_split_multiplet_head_occupations(e, _index_step(e, 3)[:, :5], where="shape")


# ---------------------------------------------------------------------------
# Known answer at nonzero time: the map's static head must keep G(t)^T = G(t)
# ---------------------------------------------------------------------------
#
# A k that is its own image under inversion times time reversal (IT), with IT
# acting as complex conjugation on real site functions phi.  The two degenerate
# states are stored in a complex gauge psi = phi V (as QE stores a multiplet), so
# IT mixes them.  One map: H = diag(E) + Sigma_x head(occ), then
# G(t) = psi U diag(f(E') exp(-i E' t)) U^dagger psi^dagger.  IT invariance of
# the state makes G(t) complex symmetric at every t: the known answer is
# ||G - G^T|| = 0.  The unfixed SC map passed step occupations by band index;
# the fixed map passes the Fermi-Dirac state.

def _toy_map_green(occ_for_head, t):
    rng = np.random.default_rng(7)
    phi, _ = np.linalg.qr(rng.standard_normal((6, 3)))          # real orthonormal sites
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    V, _ = np.linalg.qr(a)                                       # complex gauge in the doublet
    gauge = np.eye(3, dtype=complex)
    gauge[:2, :2] = V
    psi = phi @ gauge
    E = np.array([0.10, 0.10, 0.40])                             # doublet on E_F
    head = compute_static_head_terms(vc0=12648.106, wcoul0_static=18.3,
                                     occ=occ_for_head[None, :], cell_volume=254.476, nk_tot=512)
    H = np.diag(E) + np.diag(np.real(np.asarray(head.sigma_x_diag))[0])
    E1, U = np.linalg.eigh(H)
    f1 = np.asarray(fd_occupations(E1[None, :], 0.10, KT_RY))[0]
    w = f1 * np.exp(-1j * E1 * t)
    G = (psi @ U) @ np.diag(w) @ (psi @ U).conj().T
    return G


@pytest.mark.parametrize("t", [-1.0j, 0.5 - 1.0j, -8.0j])
def test_index_step_head_breaks_green_reciprocity_at_nonzero_time(t):
    step = np.array([1.0, 0.0, 0.0])
    G = _toy_map_green(step, t)
    assert np.linalg.norm(G - G.T) / np.linalg.norm(G) > 1.0e-2


@pytest.mark.parametrize("t", [-1.0j, 0.5 - 1.0j, -8.0j])
def test_fermi_dirac_head_keeps_green_reciprocity_at_nonzero_time(t):
    fd = np.asarray(fd_occupations(np.array([[0.10, 0.10, 0.40]]), 0.10, KT_RY))[0]
    G = _toy_map_green(fd, t)
    assert np.linalg.norm(G - G.T) / np.linalg.norm(G) < 1.0e-13


def test_sc_map_has_one_occupation_owner_for_bundles_and_head():
    """Pin the routine in gw_iteration_map (sc_iteration.py).

    Both rotate_wavefunctions calls pass ``occupations=`` (the metal's entry
    state), and no ``head_occ_kn`` assignment reads the frozen DFT response's
    ``sigma_occupations`` or ``wfns_dft``.  On the unfixed routine the frozen
    branch read ``iteration_head_response.sigma_occupations`` (a step by band
    index on a metal) and the bundles rebuilt a midgap step; this test fails
    there."""
    import ast
    import pathlib
    import gw.sc_iteration as sc

    tree = ast.parse(pathlib.Path(sc.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "gw_iteration_map")
    rotations = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "rotate_wavefunctions"]
    assert len(rotations) == 2
    for call in rotations:
        keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert keywords.get("occupations") == "bundle_occupations", keywords
    heads = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "head_occ_kn" for t in n.targets)]
    assert heads
    for stmt in heads:
        source = ast.unparse(stmt.value)
        assert "sigma_occupations" not in source and "wfns_dft" not in source, (
            f"line {stmt.lineno}: head_occ_kn = {source}")
    owner = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "bundle_occupations"
                         for t in n.targets))
    assert "metal_occ_state.f_kn" in ast.unparse(owner.value)
    # The frozen DFT response's Sigma-side ladder is rebound to this map on a metal,
    # so the MPA dynamic head and finalized samples read the same owner.
    rebinds = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "replace" and n.args
               and ast.unparse(n.args[0]) == "iteration_head_response"]
    assert len(rebinds) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in rebinds[0].keywords}
    assert "wfns_qp.occ" in keywords["sigma_occupations"]
    assert "wfns_qp.enk" in keywords["sigma_energies_ry"]
    assert "metal_occ_state.mu_ry" in keywords["efermi_ry"]


def test_rotated_bundle_and_parent_carrier_carry_the_supplied_state(monkeypatch):
    """rotate_wavefunctions binds ``occupations`` into the bundle and its parent
    carrier, and refuses two owners or a mis-shaped table."""
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import distrib_la
    from gw import wavefunction_bundle as wb

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))

    def planned_einsum(mesh, *, nq, m, k, n, **kwargs):
        def gemm(a, b):
            return jnp.einsum("qmk,qkn->qmn", a, b)
        gemm.nq = nq
        gemm.mesh = mesh
        gemm.in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
        gemm.in_sharding_b = gemm.in_sharding_a
        return gemm

    monkeypatch.setattr(distrib_la, "gemm_plan", planned_einsum)
    wb._FACE_ROTATE_CACHE.clear()
    nk, npk, nb, ns, mu = 4, 3, 4, 1, 8
    rows = np.asarray([0, 1, 3])
    plan = SimpleNamespace(
        n_parent=npk, n_full=nk, nspinor=ns, n_centroid_packed=mu,
        parent_full_rows=rows, sym=object(), parent_rows=lambda value: value[rows])
    rng = np.random.default_rng(17)
    psi = rng.normal(size=(npk, nb, ns, mu)) + 1j * rng.normal(size=(npk, nb, ns, mu))
    energies = jnp.asarray(np.sort(rng.normal(size=(nk, nb)), axis=1))
    bare = wb.Wavefunctions(enk=energies, occ=jnp.zeros((nk, nb)),
                            slices=wb.BandSlices.from_band_edges(0, 0, 2, nb, nb),
                            layout="face")
    parent = wb.attach_packed_parent_green_carrier(
        bare, jnp.asarray(psi), jnp.asarray(psi.transpose(0, 2, 3, 1)),
        plan=plan, mesh_xy=mesh)
    u = jnp.asarray(np.broadcast_to(np.eye(nb), (nk, nb, nb)).astype(complex))
    state = np.asarray(fd_occupations(np.asarray(energies), 0.0, KT_RY))
    rotated = wb.rotate_wavefunctions(parent, u, enk_active_new=energies, efermi=None,
                                      mesh_xy=mesh, occupations=state)
    np.testing.assert_array_equal(np.asarray(rotated.occ), state)
    np.testing.assert_array_equal(np.asarray(rotated.green_parent.occ), state[rows])
    with pytest.raises(ValueError, match="GATE rotate_occupation_owner"):
        wb.rotate_wavefunctions(parent, u, enk_active_new=energies, efermi=0.0,
                                mesh_xy=mesh, occupations=state)
    with pytest.raises(ValueError, match="GATE rotate_occupation_owner"):
        wb.rotate_wavefunctions(parent, u, enk_active_new=energies, efermi=None,
                                mesh_xy=mesh, occupations=state[:, :3])
    wb._FACE_ROTATE_CACHE.clear()
