"""P4 retarded fallback and direct streamed moments against a Lehmann sum.

Run only as a compute-node script. The two projection schemas are distinct:
retarded mode combines its orientations; direct mode accepts explicit rows.
"""


def main():
    import argparse
    import json
    from pathlib import Path
    import runtime
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from common.collectives import gather_to_host
    from common.gpu_utils import device_budget_bytes
    from ffi import fft as F
    from gw import response_bank as owner
    from gw.tangential_recipe import CapacityLedger
    import test_response_bank_blocks_cpu as fixture
    import test_tdagw_dynsig_sigma as reference

    assert mesh.size == 4
    budget = int(device_budget_bytes())
    records = []
    for fractional in (False, True):
        wfns, meta, kmq, occupations = fixture.system(mesh, fractional=fractional)
        ledger = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=budget)
        ledger.live_stages = ()
        meta.tangential_capacity = ledger
        psi = np.asarray(gather_to_host(wfns.green_parent.psi_nmu))[:, :reference.NBF]
        energy = np.asarray(gather_to_host(wfns.green_parent.enk))[:, :reference.NBF]
        # Independent expansion of (f_a-f_b)/(E_a-E_b-z). Full Nk
        # normalization remains fixed when only selected output q rows run.
        qids = (0, 2)
        coefficients = [np.zeros((len(qids), reference.NMU, reference.NMU), complex)
                        for _ in range(4)]
        for row, q in enumerate(qids):
            for k in range(meta.nk_tot):
                kb = kmq[q, k]
                rho = np.einsum('asu,bsu->abu', psi[k], np.conj(psi[kb]))
                de = energy[k, :, None] - energy[kb, None, :]
                df = occupations[k, :, None] - occupations[kb, None, :]
                for power, value in enumerate(coefficients):
                    value[row] -= np.einsum('abu,ab,abv->uv', rho, df * de**power,
                                            np.conj(rho)) / meta.nk_tot
        expected = [coefficients[i] for i in (1, 3, 0, 2)]

        def execute(kernel, args, *a, **k):
            return owner._compiled(kernel, args)(*args)

        with F.xla_reference():
            for ordered in (False, True):
                count = 4 if ordered else 2
                direct = owner.exact_bare_moments(wfns, meta, mesh_xy=mesh,
                    q_ids=qids, execute=execute, ordered=ordered)[:-1]
                receipt = dict(correlation_count=0)
                streamed = owner.streamed_moment_totals(wfns, meta, mesh_xy=mesh,
                    qids=np.asarray(qids), width=1, execute=execute, ordered=ordered,
                    receipt=receipt, root=str(options.out))
                assert streamed is not None
                bank, finish = streamed
                assembled = [[] for _ in range(count)]
                try:
                    runs = bank.reader([(i * count, (i + 1) * count) for i in range(len(qids))],
                                       run=lambda program, *args: program(*args))
                    split = owner._moment_batch_split(mesh, count, 1)
                    for run in runs:
                        values = owner._batch_totals(run, finish, split)
                        for dst, value in zip(assembled, values):
                            dst.append(np.asarray(gather_to_host(value)))
                finally:
                    bank.release()
                errors = []
                for fallback, pieces, wanted in zip(direct, assembled, expected[:count]):
                    actual = np.asarray(gather_to_host(fallback))
                    together = np.concatenate(pieces, axis=0)
                    np.testing.assert_allclose(actual, wanted, rtol=2e-11, atol=2e-10)
                    np.testing.assert_allclose(together, wanted, rtol=2e-11, atol=2e-10)
                    np.testing.assert_allclose(actual, together, rtol=2e-11, atol=2e-10)
                    errors.append(float(np.max(np.abs(actual-wanted))))
                records.append(dict(fractional=fractional, ordered=ordered, qids=qids,
                                    max_abs_by_moment=errors))
    options.out.mkdir(exist_ok=True, parents=True)
    (options.out / f'rank{jax.process_index()}.json').write_text(
        json.dumps(dict(verdict='PASS',records=records), indent=2)+'\n')
    print('MOMENT_MODES PASS', jax.process_index(), flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
