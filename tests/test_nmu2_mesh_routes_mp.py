"""The mesh backups whose providers need one process per device (ScaLAPACK on the CPU leg):
run under ``srun -n 4`` (CPU pool, one CPU device per process, 2x2 mesh); skipped otherwise.
Each compares the mesh route with the local route it backs up (value identity to the
solver's tolerance) and checks that the mesh route's n x n arrays are faces (global / P)."""
import os
import types

import numpy as np
import pytest

if int(os.environ.get("SLURM_NTASKS", "1")) != 4:
    pytest.skip("needs srun -n 4 (one CPU device per process)", allow_module_level=True)

from runtime import initialize_communicator_stack  # noqa: E402
STACK = initialize_communicator_stack(platform="cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
from common.collectives import device_put_process_local, gather_to_host  # noqa: E402

MESH = STACK.mesh
FACE3 = NamedSharding(MESH, P(None, "x", "y"))
REP = NamedSharding(MESH, P())


def _put(a, sharding):
    return device_put_process_local(np.asarray(a), sharding)


def _faces(array, n):
    """Every addressable shard of an n x n stack is a face: (.., n/2, n/2) on the 2x2 mesh."""
    return all(tuple(s.data.shape[-2:]) == (n // 2, n // 2) for s in array.addressable_shards)


def _rooms(monkeypatch, rooms, default=1 << 34):
    from common import gpu_utils
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: rooms.get(stage, default))


def test_transverse_mesh_inverse_matches_the_ridged_lu(monkeypatch):
    from isdf import core
    n, n_log = 32, 30
    rng = np.random.default_rng(21)
    a = rng.standard_normal((n_log, n_log)) + 1j * rng.standard_normal((n_log, n_log))
    w = np.r_[-np.linspace(1, 3, n_log // 2), np.linspace(0.5, 2, n_log - n_log // 2)]
    q, _ = np.linalg.qr(a)
    c_log = (q * w) @ q.conj().T                                  # Hermitian indefinite, kappa ~ 6
    c = np.zeros((1, n, n), complex); c[0, :n_log, :n_log] = c_log
    z = rng.standard_normal((1, n, 8)) + 1j * rng.standard_normal((1, n, 8)); z[:, n_log:] = 0
    ridge = 1e-12 * np.sign(np.trace(c_log).real) * abs(np.trace(c_log)) / n_log
    ref = np.zeros_like(z); ref[0, :n_log] = np.linalg.solve(c_log + ridge * np.eye(n_log), z[0, :n_log])
    # local route: today's hoisted LU (whole tiles), applied as route G applies it
    _rooms(monkeypatch, {})
    lu, piv = core.factor_c_q(_put(c, FACE3), MESH, vertex_mu_L=1, n_rmu_logical=n_log)
    apply_lu = core._zeta_logical_solvers(n_log)[1]
    local = np.asarray(gather_to_host(jax.jit(lambda L, p, Z: apply_lu(L[0], p[0], Z[0]))(
        lu, piv, _put(z, FACE3))))
    # mesh route: the local tiles over the room, the LU solve over one-device size
    _rooms(monkeypatch, {"transverse zeta factor": 40_000, "transverse zeta inverse": 1 << 14})
    inverse, none = core.factor_c_q(_put(c, FACE3), MESH, vertex_mu_L=1, n_rmu_logical=n_log)
    assert none is None and _faces(inverse, n)
    mesh = np.einsum("qmn,qng->qmg", np.asarray(gather_to_host(inverse)), z)
    tol = 1e3 * np.finfo(float).eps * np.abs(ref).max()          # kappa * eps, kappa ~ 6
    np.testing.assert_allclose(local, ref[0], atol=tol)
    np.testing.assert_allclose(mesh, ref, atol=tol)


def test_galerkin_mesh_solves_match_the_replicated_factor(monkeypatch):
    from isdf import galerkin as gk
    c, phys, nk, nb = 32, 30, 4, 8
    rng = np.random.default_rng(22)
    a = rng.standard_normal((c, 2 * c)) + 1j * rng.standard_normal((c, 2 * c))
    L = np.linalg.cholesky(a @ a.conj().T / c)
    _rooms(monkeypatch, {"Galerkin selected-state solve": 1 << 14})     # the mesh LU, not one device
    L_rep, L_face = _put(L, REP), _put(L, NamedSharding(MESH, P("x", "y")))
    assert _faces(L_face, c)
    # the row solves (basis at nodes, r chunks)
    rows = rng.standard_normal((c, 2, 24)) + 1j * rng.standard_normal((c, 2, 24))
    layout = NamedSharding(MESH, P(None, None, "y"))
    local = gk._make_basis_solve_kernel(mesh=MESH, rank=c, nspinor=2, r_carrier=24,
                                        row_layout=layout)(L_rep, _put(rows, layout))
    mesh = gk._mesh_lower_solve(L_face, _put(rows, layout), layout)
    np.testing.assert_allclose(np.asarray(gather_to_host(mesh)), np.asarray(gather_to_host(local)),
                               rtol=1e-11, atol=1e-11)
    # the coefficient projection and the rank receipts
    proj_sh = NamedSharding(MESH, P(None, None, ("x", "y")))
    proj = rng.standard_normal((nk, nb, c)) + 1j * rng.standard_normal((nk, nb, c))
    c_local = gk._solve_coefficient_projection(_put(proj, proj_sh), L_rep, MESH)
    c_mesh = gk._solve_coefficient_projection(_put(proj, proj_sh), L_face, MESH)
    np.testing.assert_allclose(np.asarray(gather_to_host(c_mesh)), np.asarray(gather_to_host(c_local)),
                               rtol=1e-11, atol=1e-11)
    sel = tuple(range(phys))
    m_local = gk._galerkin_rank_metrics(c_local, L_rep, selected_rows=sel, physical=phys)
    m_mesh = gk._galerkin_rank_metrics(c_local, L_face, selected_rows=sel, physical=phys)
    host = lambda m: [float(np.asarray(v.addressable_data(0))) for v in m]
    np.testing.assert_allclose(host(m_mesh), host(m_local), rtol=1e-13)


def test_vq_coarse_face_with_the_mesh_eigh(monkeypatch):
    from bse import vq_interp as vq
    nq, nmu, ngk, ng = 4, 32, 40, 12
    rng = np.random.default_rng(11)
    zg = rng.standard_normal((nq, nmu, ngk)) + 1j * rng.standard_normal((nq, nmu, ngk))
    a = rng.standard_normal((nq, nmu, 2 * nmu)) + 1j * rng.standard_normal((nq, nmu, 2 * nmu))
    c = _put(a @ np.conj(np.swapaxes(a, 1, 2)), FACE3)
    reader = types.SimpleNamespace(read_q_slab=lambda s, nb, sharding: _put(zg[s:s + nb], sharding))
    zx = dict(nq=nq, n_mu=nmu, ngkmax=ngk, ZG=reader, qfr=np.c_[rng.uniform(-.5, .5, (nq, 2)), np.zeros(nq)],
              rmu_frac=rng.uniform(0, 1, (nmu, 3)), zeta_cutoff=10.0)
    gs = np.r_[rng.integers(-2, 3, (2, ng)), np.zeros((1, ng), int)]
    slots = rng.integers(-1, ngk, (nq, ng))
    v = {None: rng.uniform(.5, 2, (nq, ngk)), "slab_lr": rng.uniform(.1, .4, (nq, ngk))}
    monkeypatch.setattr(vq, "lr_gset", lambda zx, alpha, degrees=None: gs)
    monkeypatch.setattr(vq, "v_sphere_padded", lambda zx, kind=None, alpha=None: v[kind])
    monkeypatch.setattr(vq, "_sphere_slot", lambda zx, q, GS: slots[q])
    monkeypatch.setattr(vq, "v_slab_on_set", lambda zx, q, GS, kind=None, alpha=None: np.ones(ng))
    _rooms(monkeypatch, {"BSE V(q) coarse eigh": 1 << 14, "BSE V(q) coarse GEMM": 1 << 10})
    kw = dict(degrees={0: 1}, keep_host_mirrors=True)
    batched = vq.prepare_coarse(zx, c, MESH, q_chunk=4, **kw)
    face = vq.prepare_coarse(zx, c, MESH, q_chunk=3, **kw)          # the whole-mesh eigh (room 16 KiB)
    for key in ("S", "V_SRc_np", "Fch"):
        np.testing.assert_allclose(face[key], batched[key], rtol=1e-9, atol=1e-9 * np.abs(batched[key]).max())
