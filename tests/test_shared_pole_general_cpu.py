"""General full-port shared poles: complex/signed residues, no PSD repair."""
import numpy as np
import pytest

from gw.shared_pole_constructor import reduce_general_pencil
from gw.shared_pole_pencil import general_tangential_pencil


def _fake_canonical_handoff(monkeypatch):
    """Two sequential hosts; preserve root's real metadata and uint64 bytes."""
    import common.collectives as cc
    from jax.experimental import multihost_utils as mh
    state = dict(rank=0, records=[], bits=[], g=0, b=0)
    monkeypatch.setattr(cc, 'process_count', lambda: 2)
    monkeypatch.setattr(cc, 'process_rank', lambda: state['rank'])
    def gather(a):
        a = np.asarray(a)
        i = state['g']; state['g'] += 1
        if state['rank'] == 0:
            state['records'].append(a.copy())
        root = state['records'][i]
        assert root.shape == a.shape
        return np.stack((root, np.zeros_like(root)))
    def broadcast(a, *, is_source):
        assert a.dtype == np.uint64 and is_source == (state['rank'] == 0)
        i = state['b']; state['b'] += 1
        if is_source:
            state['bits'].append(a.copy())
        return state['bits'][i].copy()
    monkeypatch.setattr(cc, 'all_gather_processes', gather)
    monkeypatch.setattr(mh, 'broadcast_one_to_all', broadcast)
    def select(rank):
        state.update(rank=rank, g=0, b=0)
        if rank == 0:
            state.update(records=[], bits=[])
    return state, select


def test_canonical_reduced_owner_avoids_heterogeneous_gauge_mixing(monkeypatch):
    import gw.shared_pole_constructor as owner
    state, select = _fake_canonical_handoff(monkeypatch)
    real_reduce = owner.reduce_general_pencil
    calls = []
    def reduction(*a, **kw):
        assert state['rank'] == 0, 'worker must never run its heterogeneous LAPACK'
        calls.append(kw.get('fixed_spaces'))
        return real_reduce(*a, **kw)
    monkeypatch.setattr(owner, 'reduce_general_pencil', reduction)
    poles, exact, E, A, B, C = _case()
    held = None
    for shift in (0., .001):
        select(0)
        root = owner.collective_general_pencil(E, A+shift*E, max_order=4, fixed_spaces=held)
        assert len(state['bits']) == 5
        select(1)
        # Poison worker's would-be frequency gauge; only the owner's spaces
        # and current model are allowed to form distributed full-port faces.
        wrong = None if held is None else {k: -v for k,v in held.items()}
        worker = owner.collective_general_pencil(E, A+shift*E, max_order=4, fixed_spaces=wrong)
        for key in ('poles_ry', 'output_map', 'input_map'):
            np.testing.assert_array_equal(root[key].view(np.uint64), worker[key].view(np.uint64))
        for key in ('left', 'right'):
            np.testing.assert_array_equal(root['spaces'][key], worker['spaces'][key])
        assert root['receipt'] == worker['receipt']
        z = .3+.2j
        actual = ((C@worker['output_map'])/(z-worker['poles_ry'])) @ (B.conj().T@worker['input_map']).conj().T
        np.testing.assert_allclose(actual, exact(z-shift), rtol=3e-12, atol=3e-12)
        if shift == 0.:
            # Individually equivalent whole-factor gauges become WRONG when
            # different hosts contribute different row faces to one factor.
            left, right = C@root['output_map'], B.conj().T@root['input_map']
            gauge = np.ones(len(root['poles_ry'])); gauge[0] = -1
            left[2:] *= gauge; right[2:] *= gauge
            mixed = (left/(z-root['poles_ry']))@right.conj().T
            assert np.linalg.norm(mixed-exact(z)) > .1
        held = worker['spaces']
    assert len(calls) == 2 and calls[1] is not None  # current E/A recomputed, spaces held


@pytest.mark.parametrize('zero', [False, True])
def test_canonical_owner_refusal_and_zero_model_share_collective_protocol(monkeypatch, zero):
    import gw.shared_pole_constructor as owner
    state, select = _fake_canonical_handoff(monkeypatch)
    E, A = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    if zero:
        select(0)
        root = owner.collective_general_pencil(E, A, max_order=2)
        select(1)
        worker = owner.collective_general_pencil(E, A, max_order=2)
        assert len(state['bits']) == 0
        for key in ('poles_ry','output_map','input_map'):
            np.testing.assert_array_equal(root[key], worker[key])
    else:
        def refused(*a, **kw):
            assert state['rank'] == 0
            raise ValueError('GATE owner_test: defective input')
        monkeypatch.setattr(owner, 'reduce_general_pencil', refused)
        errors = []
        for rank in (0, 1):
            select(rank)
            with pytest.raises(ValueError, match='shared_general_owner.*owner_test') as caught:
                owner.collective_general_pencil(E, A, max_order=2)
            errors.append(str(caught.value))
        assert errors[0] == errors[1] and not state['bits']


def test_canonical_rawbit_handoff_refuses_single_precision_before_payload(monkeypatch):
    import jax
    import gw.shared_pole_constructor as owner
    state, select = _fake_canonical_handoff(monkeypatch)
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', False)
    try:
        for rank in (0, 1):
            select(rank)
            with pytest.raises(ValueError, match='shared_general_precision'):
                owner.collective_general_pencil(np.eye(2), np.eye(2), max_order=2)
    finally:
        jax.config.update('jax_enable_x64', previous)
    assert not state['bits']


@pytest.mark.parametrize("which", ["k", "q"])
def test_direct_full_microscopic_fft_refuses_permuted_two_dimensional_rows(which):
    from types import SimpleNamespace
    from gw.centroid_k_unfold import authenticate_canonical_full_grid
    grid = np.array([2, 3, 1])
    ijk = np.stack(np.unravel_index(np.arange(6), tuple(grid)), axis=1)
    plan = SimpleNamespace(k_parent_frac=ijk/grid[None], irr_idx=np.arange(6),
                           parent_full_rows=np.arange(6))
    sym = SimpleNamespace(q_irr_kgrid_int=ijk.copy(), q_irr_full_idx=np.arange(6))
    assert authenticate_canonical_full_grid(plan, grid, sym=sym)["q_order"] == "canonical-C"
    perm = [0, 3, 1, 4, 2, 5]
    if which == "k":
        plan.k_parent_frac = plan.k_parent_frac[perm]
    else:
        sym.q_irr_kgrid_int = sym.q_irr_kgrid_int[perm]
    # Even identity index tables do not authenticate physical row order.
    with pytest.raises(ValueError, match="full_microscopic_fft_order"):
        authenticate_canonical_full_grid(plan, grid, sym=sym)


def _mm(a, b, transa=None):
    return (a.conj().T if transa == "C" else a) @ b


def _case(seed=9):
    rng = np.random.default_rng(seed)
    poles = np.array([-.8-.12j, .4-.07j, 1.2+.015j])
    left = rng.normal(size=(5, 3)) + 1j*rng.normal(size=(5, 3))
    right = rng.normal(size=(5, 3)) + 1j*rng.normal(size=(5, 3))
    value = lambda z: (left / (z-poles)) @ right.conj().T
    mu = np.array([-.7+.4j, .1+.5j, .9+.4j, 1.5+.6j])
    lam = np.array([-1+.6j, -.2+.3j, .6+.5j, 1.3+.3j])
    ql = rng.normal(size=(5, 4)) + 1j*rng.normal(size=(5, 4))
    qr = rng.normal(size=(5, 4)) + 1j*rng.normal(size=(5, 4))
    vl = np.stack([ql[:, i].conj() @ value(z) for i, z in enumerate(mu)])
    wr = np.stack([value(z) @ qr[:, i] for i, z in enumerate(lam)], axis=1)
    E, A, B, C = (np.asarray(a) for a in general_tangential_pencil(
        (mu, ql, vl), (lam, qr, wr), matmul=_mm))
    return poles, value, E, A, B, C


def test_general_tangential_poles_preserve_full_complex_response():
    poles, exact, E, A, B, C = _case()
    out = reduce_general_pencil(E, A, max_order=4)
    assert out["receipt"]["order"] == 3
    np.testing.assert_allclose(np.sort_complex(out["poles_ry"]), np.sort_complex(poles), atol=2e-13)
    left = C @ out["output_map"]
    right = B.conj().T @ out["input_map"]
    assert left.shape[0] == right.shape[0] == 5  # no spatial-port reduction
    for z in [-.3+.2j, .35+.11j, 2+.8j]:
        got = (left / (z-out["poles_ry"])) @ right.conj().T
        np.testing.assert_allclose(got, exact(z), rtol=3e-12, atol=3e-12)
    assert out["poles_ry"].imag.max() > 0  # physical gain is not projected away


def test_general_pencil_workspace_refuses_before_factorization():
    _, _, E, A, _, _ = _case()
    with pytest.raises(ValueError, match="shared_general_capacity"):
        reduce_general_pencil(E, A, max_order=4, max_workspace_bytes=1)


def test_general_fixed_frequency_spaces_update_current_response():
    _, _, E, A, B, C = _case()
    initial = reduce_general_pencil(E, A, max_order=4)
    same = reduce_general_pencil(E, A, max_order=4, fixed_spaces=initial["spaces"])
    np.testing.assert_array_equal(initial["poles_ry"], same["poles_ry"])
    changed = reduce_general_pencil(E, A+.001*E, max_order=4, fixed_spaces=initial["spaces"])
    np.testing.assert_allclose(changed["poles_ry"], initial["poles_ry"]+.001, atol=2e-14)
    z = .3+.2j
    actual = ((C@changed["output_map"])/(z-changed["poles_ry"])) @ (B.conj().T@changed["input_map"]).conj().T
    previous = ((C@initial["output_map"])/(z-initial["poles_ry"])) @ (B.conj().T@initial["input_map"]).conj().T
    assert np.linalg.norm(actual-previous) > 1e-3


def test_general_pencil_signed_residue_not_psd_constrained():
    E, A = np.eye(2), np.diag([-.4-.05j, .7-.03j])
    out = reduce_general_pencil(E, A, max_order=2)
    C, B = np.diag([1., -2.]), np.eye(2)
    left, right = C @ out["output_map"], B.T @ out["input_map"]
    z = .2+.3j
    np.testing.assert_allclose((left/(z-out["poles_ry"])) @ right.conj().T,
                               C @ np.linalg.solve(z*E-A, B), atol=1e-13)


def test_general_store_roundtrip_signed_complex_full_port(tmp_path, monkeypatch):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from types import SimpleNamespace
    import file_io.shared_pole_store as store
    if len(jax.devices()) < 4:
        pytest.skip("file roundtrip uses the documented four-device CPU emulated SlabIO mesh")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    nq, nmu, width = 2, 4, 4
    identity = dict(iteration_id="general-test", hamiltonian="h", energies="e",
                    occupations="rho", wavefunctions="psi", centroids="mu")
    basis = SimpleNamespace(mesh_xy=mesh, n_packed=nmu, n_logical=nmu, n_canonical=nmu,
        active_mask=np.ones(nmu, bool), canonical_indices=np.arange(nmu),
        unpack_axis=lambda a, axis: a, pack_axis=lambda a, axis, spec=None: a)
    meta = SimpleNamespace(mu_basis=basis)
    base = dict(schema=store.SCHEMA, identity=identity, n_q_irr=nq, n_q_full=nq,
        n_mu_logical=nmu, nspinor=2, finalized=False, recipe={}, recipe_hash="recipe",
        centroid_digest="mu", qirr={})
    monkeypatch.setattr(store, "_metadata", lambda *a, **k: dict(base))
    monkeypatch.setattr(store, "_write_metadata", lambda *a: None)
    monkeypatch.setattr(store, "_capacity", lambda *a: None)
    monkeypatch.setattr(store, "_admit", lambda *a, **k: None)
    monkeypatch.setattr(store, "_check_io_capacity", lambda *a: None)
    monkeypatch.setattr(store, "_conversion_bytes", lambda *a, **k: (0, 0, 0))
    monkeypatch.setattr(store, "_check_basis", lambda *a, **k: basis)
    rng = np.random.default_rng(5)
    left = rng.normal(size=(nq, nmu, 1, width))+1j*rng.normal(size=(nq, nmu, 1, width))
    right = rng.normal(size=left.shape)+1j*rng.normal(size=left.shape)
    poles = np.array([[-.8-.1j, .4+.02j, .9-.07j, 1.4-.1j]]*nq)
    counts = np.array([4, 2])
    left[1, :, :, 2:], right[1, :, :, 2:] = 0, 0
    path = str(tmp_path/"general.h5")
    face = NamedSharding(mesh, P(None, "x", None, "y"))
    for q in range(nq):
        header = store.write_general_shared_pole_model(path,
            jax.device_put(left[q:q+1], face), jax.device_put(right[q:q+1], face),
            poles[q:q+1], counts[q:q+1], q_span=(q, q+1), meta=meta, tables={},
            recipe={}, receipts=dict(identity=identity), kmax=width)
        assert header["finalized"] == (q == nq-1)
    checked = store.validate_general_shared_pole_model(path, expected_identity=identity,
                                                       mesh_xy=mesh, capacity=None)
    assert checked["digest"] == header["digest"]
    with store.open_shared_pole_model(path, mesh_xy=mesh) as io:
        lx, ry, pp, kk = store.read_general_shared_pole_faces(io, (0, nq), meta=meta, header=checked)
    np.testing.assert_array_equal(lx, left)
    np.testing.assert_array_equal(ry, right)
    np.testing.assert_array_equal(kk, counts)
    np.testing.assert_array_equal(pp[0], poles[0])
    np.testing.assert_array_equal(pp[1, :2], poles[1, :2])
