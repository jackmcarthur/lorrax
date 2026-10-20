"""The head's sum set: ``head_dense_file`` (docs/theory/s-tensor-convention.md#sum-set-and-cell-set), CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: device-count and arithmetic logic only).
The GW mesh is the hsuite fixture WFN (5x5x1, two spinor components, 3
occupied of the 8-band chi window) with a synthetic, provenance-stamped
``dipole.h5``; the centroid faces and W(Gamma) are random.

(a) ``head_dense_file`` naming the GW mesh's own ``dipole.h5`` is the GW-mesh
    head: S_d and both wings are bit for bit the unset-key response, with no
    fold scale.  Red twin: the same file restamped with another velocity
    operator refuses (``GATE head_dense_operator``).
(b) A synthetic denser k set (64 k, its own energies and velocities) gives
    S_d equal to ``common.chi_from_dipole.compute_S_omega`` on that file to
    1e-12 at every node, ``M = S_d,sum S_d,cell^-1``, and the folded
    S_eff = S_d,sum + (M F + F M^T)/2 with F the GW mesh's own fold, at each
    node and batched.  Red twins: the additive fold S_d,sum + F differs, and a
    metal refuses (``GATE head_dense_metal``).
"""
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parent / "hsuite" / "fixture" / "WFN.h5"
NB, NMU, NK_DENSE = 8, 8, 64
Z = np.array([0.0, 0.3j, 1.2j, 0.25 + 0.4j])
RNG = np.random.default_rng(7)


def _mesh():
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(cpu[:4]).reshape(2, 2), ("x", "y"))


def _hermitian_velocity(nk, nb):
    v = RNG.standard_normal((nk, 3, nb, nb)) + 1j * RNG.standard_normal((nk, 3, nb, nb))
    return 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))


def _write(path, velocity, energies, mesh, provenance):
    from file_io.dipole import DIPOLE_BASIS_ATTR, write_dipole
    nk, nb = energies.shape
    write_dipole(path, velocity, energies, mesh=mesh,
                 attrs={"nbands": nb, "nk": nk, "skip_vnl": False,
                        DIPOLE_BASIS_ATTR: "dft", **provenance})


@pytest.fixture(scope="module")
def gw_mesh(tmp_path_factory):
    """The GW mesh: fixture WFN, its stamped dipole.h5, faces, meta and deck."""
    from common.wfn_layout import psi_specs
    from file_io.dipole import band_energies_on_full_bz
    from gw.gw_config import HeadCorrection
    from gw.qsgw_head import expected_hubbard_stamp
    from psp.get_dipole_mtxels import dipole_provenance
    from wfn_loader import WfnLoader
    mesh = _mesh()
    tmp = tmp_path_factory.mktemp("head_dense")
    wfn = WfnLoader(str(FIXTURE))
    sym = wfn.symmetry()
    nk, n_occ = int(sym.nk_tot), int(wfn.nelec)
    assert 0 < n_occ < NB and int(wfn.nspinor) == 2
    config = SimpleNamespace(
        head=SimpleNamespace(wcoul0_eta=0.0, dense_file="",
                             correction=HeadCorrection.FULL),
        nval=n_occ, ncond=NB - n_occ, nband=NB, bispinor=False,
        bispinor_gw="bare_transverse", hubbard_input="",
        hubbard_occupations="", input_dir=str(tmp))
    energies = band_energies_on_full_bz(wfn, sym, NB)
    _write(tmp / "dipole.h5", _hermitian_velocity(nk, NB), energies, mesh,
           dipole_provenance(
               wfn=wfn, wfn_path=str(FIXTURE), nval=n_occ, ncond=NB - n_occ,
               nband=NB, nb_written=NB, bispinor=False, skip_vnl=False,
               vnl_mode="analytic", nspinor=2, soc=False,
               hubbard=expected_hubbard_stamp(config, wfn=wfn,
                                              fallback_dir=str(tmp), caller="test")))
    occ = np.zeros_like(energies); occ[:, :n_occ] = 1.0
    psi = RNG.standard_normal((nk, 2, NMU, NB)) + 1j * RNG.standard_normal((nk, 2, NMU, NB))
    nmu_spec, mun_spec = psi_specs("face")
    put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
    wfns = SimpleNamespace(layout="face", psi_mun=put(psi, mun_spec),
                           psi_nmu=put(np.transpose(psi, (0, 3, 1, 2)), nmu_spec),
                           enk=energies, occ=occ, slices=("one",))
    meta = SimpleNamespace(b_id_0=0, b_id_4_chi_user=NB, nk_tot=nk,
                           cell_volume=float(wfn.cell_volume), nspinor=2,
                           nspinor_wfnfile=2, nelec=n_occ, nb_sigma=NB)
    W = RNG.standard_normal((NMU, NMU)) + 1j * RNG.standard_normal((NMU, NMU))
    W = put(0.05 * (W + W.conj().T), P("x", "y"))
    return SimpleNamespace(mesh=mesh, tmp=tmp, wfn=wfn, wfns=wfns, meta=meta,
                           config=config, W=W, n_occ=n_occ)


def _build(g, dense_file="", **kw):
    from gw.qsgw_head import build_dft_head_response
    config = SimpleNamespace(**{**vars(g.config), "head": SimpleNamespace(
        **{**vars(g.config.head), "dense_file": dense_file})})
    return build_dft_head_response(
        g.wfns, Z, input_dir=str(g.tmp), mesh=g.mesh, wfn=g.wfn, meta=g.meta,
        config=config, **kw), config


def test_own_dipole_is_the_gw_mesh_bitwise_and_operator_red_twin(gw_mesh):
    import h5py
    g = gw_mesh
    base, _ = _build(g)
    same, _ = _build(g, "dipole.h5")
    assert base.fold_scale is None and same.fold_scale is None
    for name in ("S_direct", "Y_x", "Z_y"):
        assert np.array_equal(np.asarray(getattr(base, name)),
                              np.asarray(getattr(same, name))), name
    other = g.tmp / "dipole_p.h5"
    shutil.copy(g.tmp / "dipole.h5", other)
    with h5py.File(other, "a") as f:
        f.attrs["prov_skip_vnl"] = True
    with pytest.raises(ValueError, match="GATE head_dense_operator"):
        _build(g, os.fspath(other))


def test_dense_set_matches_compute_S_omega_and_scaled_fold(gw_mesh):
    from common.chi_from_dipole import compute_S_omega, read_dipole_h5
    from gw.head_correction import fold_cartesian_head_wings_sharded
    from gw.qsgw_head import HeadSource, iteration_head_sample_terms
    g = gw_mesh
    e = np.empty((NK_DENSE, NB))
    e[:, :g.n_occ] = np.sort(RNG.uniform(-1.0, -0.3, (NK_DENSE, g.n_occ)), 1)
    e[:, g.n_occ:] = np.sort(RNG.uniform(0.2, 1.5, (NK_DENSE, NB - g.n_occ)), 1)
    path = g.tmp / "dipole_dense.h5"
    _write(path, _hermitian_velocity(NK_DENSE, NB), e, g.mesh, {})
    source = HeadSource(velocity_path=os.fspath(path), wfn=None,
                        nk_sum=NK_DENSE, kgrid_sum=(4, 4, 4),
                        band_window=(0, NB), dense=True)
    cell, config = _build(g)
    dense, _ = _build(g, source=source)
    # (b) S_d of the sum set, at every node, on its own N_k.
    v, dE = read_dipole_h5(os.fspath(path))
    f = np.zeros((NK_DENSE, NB)); f[:, :g.n_occ] = 1.0
    ref = np.asarray(compute_S_omega(v, dE, f, g.meta.cell_volume, NK_DENSE, 1, 2, Z))
    S_sum = np.asarray(dense.S_direct)
    rel = np.abs(S_sum - ref).max(axis=(1, 2)) / np.abs(ref).max(axis=(1, 2))
    assert rel.max() < 1e-12, rel
    # The wings and the cell set stay the GW mesh's.
    assert np.array_equal(np.asarray(dense.Y_x), np.asarray(cell.Y_x))
    M = np.asarray(dense.fold_scale)
    S_cell = np.asarray(cell.S_direct)
    assert np.abs(M @ S_cell - S_sum).max() < 1e-12 * np.abs(S_sum).max()
    # The scaled fold at each node (iteration_head_sample_terms) ...
    for i in range(len(Z)):
        F = (iteration_head_sample_terms(cell, i, g.W, meta=g.meta, config=config,
                                         mesh=g.mesh)["S_effective"] - S_cell[i])
        got = iteration_head_sample_terms(dense, i, g.W, meta=g.meta, config=config,
                                          mesh=g.mesh)["S_effective"]
        want = S_sum[i] + 0.5 * (M[i] @ F + F @ M[i].T)
        scale = np.abs(want).max()
        assert np.abs(got - want).max() < 1e-12 * scale, i
        # Red twin: the additive fold is a different head.
        assert np.abs(got - (S_sum[i] + F)).max() > 1e-6 * scale, i
    # ... and batched over nodes (finalize_iteration_head_samples' call).
    Wz = jax.device_put(np.broadcast_to(np.asarray(g.W), (len(Z), NMU, NMU)).copy(),
                        NamedSharding(g.mesh, P(None, "x", "y")))
    fold = lambda r: np.asarray(fold_cartesian_head_wings_sharded(
        r.S_direct, r.Y_x, Wz, r.Z_y, g.meta.cell_volume, mesh_xy=g.mesh,
        fold_scale=r.fold_scale))
    F = fold(cell) - S_cell
    want = S_sum + 0.5 * (M @ F + F @ np.swapaxes(M, -1, -2))
    assert np.abs(fold(dense) - want).max() < 1e-12 * np.abs(want).max()
    # Red twin: a metal refuses the dense set before any read.
    metal = SimpleNamespace(mu_ry=0.0, f_kn=np.zeros((g.meta.nk_tot, NB)), occ_hash="m")
    with pytest.raises(ValueError, match="GATE head_dense_metal"):
        _build(g, source=source, occupation_state=metal)
