"""P4 accepted Coulomb roots: cross-backend reuse, certificates and lifetime."""


def main():
    import argparse
    import gc
    import json
    import weakref
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import resolve_device_budget_gb
    from gw.tangential_recipe import CapacityLedger
    from gw.w_isdf import response_coulomb_powers
    import gw.response_bank as owner

    assert mesh.size == 4
    budget = int(resolve_device_budget_gb(28, 'local') * 1e9)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    logical, nq = 37, 5
    centroids = np.column_stack((np.arange(logical), np.zeros((logical, 2), int)))
    basis = PackedCentroidBasis.build(centroids, None, (64, 1, 1), mesh, identity=True)
    n = basis.n_packed
    meta = SimpleNamespace(mu_basis=basis, nk_tot=8, nspinor=2, n_rmu=logical)
    meta.tangential_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=budget)
    meta.tangential_capacity.live_stages = ()
    index = np.arange(logical)
    u = np.exp(2j*np.pi*index[:, None]*index[None, :]/logical)/np.sqrt(logical)
    eigenvalues = np.linspace(.2, 3., logical)
    eigenvalues[:9] = 0

    def carrier(diagonal):
        value = np.zeros((n, n), complex)
        value[:logical, :logical] = (u*diagonal) @ u.conj().T
        return value

    values = np.stack([carrier(eigenvalues*(1 + .1*q)) for q in range(nq)])
    expected_h = np.stack([carrier(np.sqrt(eigenvalues*(1 + .1*q))) for q in range(nq)])
    expected_hi = np.stack([carrier(np.where(eigenvalues > 0,
        1/np.sqrt(np.where(eigenvalues > 0, eigenvalues*(1 + .1*q), 1)), 0)) for q in range(nq)])
    projector = carrier((eigenvalues > 0).astype(float))
    value = device_put_process_local(values, face)
    token = owner._operator_token(value)
    resource = dict(path=None, dataset=None, basis='canonical',
        q_irr_full_idx=list(range(nq)), operator=token, sha256=owner.operator_digest(value, mesh))
    bank = dict(identity={'test': 'accepted-root-reuse'}, coulomb=resource)
    original = owner._coulomb_roots
    solves = []

    def roots(*a, **k):
        solves.append(dict(layout=a[3], span=list(a[5])))
        return original(*a, **k)

    errors = []
    with patch.object(owner, '_coulomb_roots', roots):
        h, hi, receipt = response_coulomb_powers(meta, {'linalg': 'local'},
            mesh_xy=mesh, bank_io=bank, q_span=(0, nq))
        assert len(solves) == 1
        for layout, span in [('distributed', (0, nq)), ('local', (0, nq)),
                             ('distributed', (1, 4)), ('distributed', (4, 5)),
                             ('local', (0, 2)), ('distributed', (3, 5))]:
            a, b, got = response_coulomb_powers(meta, {'linalg': layout},
                mesh_xy=mesh, bank_io=bank, q_span=span)
            assert len(solves) == 1 and len(owner._COULOMB_ROOTS) == 1
            if span == (0, nq):
                assert a is h and b is hi, 'exact hit allocated root copies'
            lo, end = span
            ah, bh = map(lambda x: np.asarray(gather_to_host(x)), (a, b))
            np.testing.assert_allclose(ah, expected_h[lo:end], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(bh, expected_hi[lo:end], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(ah@ah, values[lo:end], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(ah@bh, np.broadcast_to(projector, ah.shape), atol=2e-10)
            assert min(np.linalg.eigvalsh(ah).min(), np.linalg.eigvalsh(bh).min()) >= -2e-10
            assert got['support_ranks'] == [logical-9]*(end-lo)
            assert got['coulomb_identity'] == resource
            assert a.sharding == b.sharding == face
            errors.append(float(np.max(np.abs(ah-expected_h[lo:end]))))
        del a, b, got, ah, bh

    # Mismatches must reach the existing solver, not reuse a nearby cache
    # entry. Stop at that boundary rather than execute an invalid fixture.
    class Miss(Exception):
        pass

    missed = []
    key = next(iter(owner._COULOMB_ROOTS))
    held = owner._COULOMB_ROOTS[key]

    def must_solve(label, *, test_meta=meta, test_mesh=mesh, test_bank=bank, span=(0, nq)):
        marker = Miss(label)
        with patch.object(owner, '_coulomb_roots', side_effect=marker):
            try:
                owner._coulomb_batch(test_meta, {'linalg': 'distributed'}, test_bank,
                                    test_mesh, span, lambda *a: None)
            except Miss as error:
                assert error is marker
                missed.append(label)
            else:
                raise AssertionError(f'{label}: reused incompatible roots')

    clone = PackedCentroidBasis.build(centroids, None, (64, 1, 1), mesh, identity=True)
    must_solve('distinct packed-basis identity', test_meta=SimpleNamespace(mu_basis=clone))
    # Same carrier, different logical extent means a different support cut.
    cut_basis = PackedCentroidBasis.build(np.zeros((logical+1, 3), int), None,
                                         (64, 1, 1), mesh, identity=True)
    assert cut_basis.n_packed == n
    must_solve('different logical cut', test_meta=SimpleNamespace(mu_basis=cut_basis))
    other_mesh = Mesh(np.asarray(mesh.devices).T, mesh.axis_names)
    assert other_mesh != mesh
    must_solve('different device mesh', test_mesh=other_mesh)
    must_solve('uncovered span', span=(nq-1, nq+1))
    must_solve('different operator', test_bank=dict(bank, coulomb=dict(resource, operator='other')))
    replicated = NamedSharding(mesh, P())
    bad_h = device_put_process_local(expected_h, replicated)
    owner._COULOMB_ROOTS[key] = (bad_h, hi, held[2])
    must_solve('wrong output sharding')
    del bad_h
    owner._COULOMB_ROOTS[key] = held
    dead = device_put_process_local(expected_h, face)
    dead.delete()
    owner._COULOMB_ROOTS[key] = (dead, hi, held[2])
    must_solve('deleted cached root')
    del dead
    owner._COULOMB_ROOTS[key] = held

    h_ref, hi_ref, value_ref = weakref.ref(h), weakref.ref(hi), weakref.ref(value)
    del h, hi, held, receipt, value
    owner.release_coulomb_cache()
    jax.effects_barrier()
    gc.collect()
    assert h_ref() is hi_ref() is value_ref() is None
    assert not owner._COULOMB_ROOTS and not owner._COULOMB_OPERATORS and not owner._OPERATOR_DIGESTS
    result = dict(status='PASS', rank=jax.process_index(), shape=[nq, n, n],
        logical=logical, solves=solves, max_h_error=max(errors), mismatch_guards=missed,
        exact_span_same_arrays=True, released=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COULOMB_ROOT_REUSE PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
