"""P4 response face/axis streams against the existing explicit band-pair oracle.

Run as a compute-node SCRIPT. Covers fractional blocks and streamed-bank
segments without changing the response rule, FFT convention or arithmetic.
"""


def main():
    import runtime
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from common.gpu_utils import device_budget_bytes
    from gw import response_bank as RB
    from lxkit import memory
    from gw.w_isdf import _w_solve_pref_scalar
    from ffi import fft as F
    import runtime.tiles as tiles
    import test_response_bank_blocks_cpu as fixture
    import test_tdagw_dynsig_sigma as reference

    if mesh.size != 4:
        raise ValueError('This independent small oracle uses the existing P4 fixture')
    device_budget_bytes()
    saved_stream, saved_tile, saved_room = RB.response_stream, tiles.TILE_BYTES, memory.room
    receipts = []
    try:
        tiles.TILE_BYTES = 4096  # Multiple row passes and the separate segment program.
        for fractional in (False, True):
            wfns, meta, _, f = fixture.system(mesh, fractional=fractional)
            if fractional:
                meta.tangential_response_blocks = dict(active=fixture.ACTIVE, held={})
            psi = np.asarray(gather_to_host(wfns.green_parent.psi_nmu))[:, :reference.NBF]
            energy = np.asarray(gather_to_host(wfns.green_parent.enk))[:, :reference.NBF]
            z = np.array([2j / fixture.RY], np.complex128)
            plan = fixture.sample_plan(z, 0.0)
            receipt = dict(correlation_count=0)
            support = RB.response_support(wfns, meta, plan, receipt, print_fn=lambda *a, **k: None)
            rules = RB.response_quadrature(meta, plan, receipt, support, group_size=1,
                                           print_fn=lambda *a, **k: None)
            exact = np.stack([reference._dense_chi(psi, energy, f, value) for value in z])
            outputs = []
            for layout in (('axis', 'face', 'auto') if not fractional else ('axis', 'face')):
                if layout == 'auto':
                    # Refuse the incumbent executable before dispatch. The
                    # minimum face route keeps the established admit policy.
                    memory.room = lambda **_: 0
                handles = []
                def forced(*args, **kwargs):
                    kernel, fixed = saved_stream(*args, **kwargs,
                        _force_sum_layout=None if layout == 'auto' else layout)
                    handles.append(kernel)
                    return kernel, fixed
                RB.response_stream = forced
                result = np.zeros_like(exact)
                with F.xla_reference():
                    for group in rules['groups']:
                        raw = RB.integrate_response_group(wfns, meta, mesh, rules, group,
                            q_ids=list(range(meta.nk_tot)), execute=lambda kernel, args, *a, **k: RB._compiled(kernel, args)(*args),
                            receipt=receipt, ordered=True)
                        values = np.asarray(gather_to_host(raw))
                        for slot, member in enumerate(group['members']):
                            # The bank stores the raw FFT convention; the
                            # production Dyson prefactor converts it to the
                            # physical 1/N_k band-pair sum used by the oracle.
                            result[int(member)] = values[2*slot] * _w_solve_pref_scalar(meta)
                    # The bank's segment consumer takes the same row windows in
                    # separate calls; assemble its LOCAL rectangles independently.
                    if not fractional and layout == 'face':
                        group = rules['groups'][0]
                        full = RB.integrate_response_group(wfns, meta, mesh, rules, group,
                            q_ids=list(range(meta.nk_tot)), execute=lambda kernel, args, *a, **k: RB._compiled(kernel, args)(*args),
                            receipt=receipt, ordered=True)
                        expected_local = np.asarray(full.addressable_shards[0].data)
                        segments, _, _ = RB.stream_segments(wfns, meta, mesh, meta.nk_tot, None)
                        assert len(segments) > 1
                        class Bank:
                            shapes = tuple((rows, cols) for rows, cols, _ in segments)
                            value = np.zeros_like(expected_local)
                            def put(self, index, array, outputs):
                                tile = np.asarray(array.addressable_shards[0].data)
                                for r, c, R, C, nr, nc in segments[index][2]:
                                    self.value[..., R:R+nr, C:C+nc] = tile[..., r:r+nr, c:c+nc]
                        bank = Bank()
                        RB.integrate_response_group(wfns, meta, mesh, rules, group,
                            q_ids=list(range(meta.nk_tot)), execute=lambda kernel, args, *a, **k: RB._compiled(kernel, args)(*args),
                            receipt=receipt, ordered=True, bank=bank)
                        np.testing.assert_allclose(bank.value, expected_local, rtol=1e-12, atol=1e-12)
                memory.room = saved_room
                expected_layout = 'face' if layout == 'auto' else layout
                assert handles and all(RB._STREAM_PROGRAMS[h]["last_layout"] == expected_layout for h in handles)
                relative = float(np.max(np.abs(result-exact)) / np.max(np.abs(exact)))
                assert relative < 50*fixture.TOL, (fractional, layout, relative)
                receipts.append(dict(fractional=fractional, layout=layout, relative_to_pair_oracle=relative))
                outputs.append(result)
            np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-10, atol=1e-11)
        # Four-current uses the same Green backend with its existing Dirac
        # half plans and channel algebra. Compare every packed channel block
        # to the incumbent axis route, including the streamed-bank consumer.
        from gw.photon_layout import PhotonBasisLayout, PhotonFamilies
        saved_ns = reference.NS
        try:
            reference.NS = 4
            wfns, meta, _, _ = fixture.system(mesh, fractional=False)
            carrier = wfns.green_parent
            photon_layout = PhotonBasisLayout.from_centroid_extents(reference.NMU, reference.NMU,
                                                                       mesh, packed=True)
            vertex = RB.PhotonEndpoints(PhotonFamilies((carrier.plan, carrier.plan), photon_layout,
                photon_layout), (carrier.psi_mun, carrier.psi_mun),
                (carrier.psi_nmu, carrier.psi_nmu), carrier.enk)
            receipt = dict(correlation_count=0)
            support = RB.response_support(wfns, meta, plan, receipt, print_fn=lambda *a, **k: None)
            rules = RB.response_quadrature(meta, plan, receipt, support, group_size=1,
                                           print_fn=lambda *a, **k: None)
            photon_outputs = []
            for layout in ('axis', 'face'):
                def forced(*args, **kwargs):
                    return saved_stream(*args, **kwargs, _force_sum_layout=layout)
                RB.response_stream = forced
                with F.xla_reference():
                    raw = RB.integrate_response_group(wfns, meta, mesh, rules, rules['groups'][0],
                        q_ids=list(range(meta.nk_tot)), execute=lambda kernel, args, *a, **k: RB._compiled(kernel, args)(*args),
                        receipt=receipt, ordered=True, vertex=vertex)
                    photon_outputs.append(np.asarray(gather_to_host(raw)))
                    if layout == 'face':
                        expected_local = np.asarray(raw.addressable_shards[0].data)
                        segments, _, _ = RB.stream_segments(wfns, meta, mesh, meta.nk_tot, vertex)
                        bank = Bank()
                        bank.shapes = tuple((rows, cols) for rows, cols, _ in segments)
                        bank.value = np.zeros_like(expected_local)
                        RB.integrate_response_group(wfns, meta, mesh, rules, rules['groups'][0],
                            q_ids=list(range(meta.nk_tot)), execute=lambda kernel, args, *a, **k: RB._compiled(kernel, args)(*args),
                            receipt=receipt, ordered=True, vertex=vertex, bank=bank)
                        np.testing.assert_allclose(bank.value, expected_local, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(*photon_outputs, rtol=1e-10, atol=1e-11)
            receipts.append(dict(four_current=True, max_axis_face=float(np.max(
                np.abs(photon_outputs[0]-photon_outputs[1])))))
        finally:
            reference.NS = saved_ns
        # A released table reservation must also release every cached axis /
        # face executable and its route metadata; arrays are never cache keys.
        from gw import w_isdf
        owned = {program for state in RB._STREAM_PROGRAMS.values()
                 for program in state['variants'].values()}
        live = w_isdf.placed_kconv_tables_live
        try:
            w_isdf.placed_kconv_tables_live = lambda tail: False
            RB._release_evicted_programs()
            assert not RB._STREAM_PROGRAMS
            assert not any(entry[0] in owned for entry in RB._COMPILED)
        finally:
            w_isdf.placed_kconv_tables_live = live
    finally:
        RB.response_stream, tiles.TILE_BYTES, memory.room = saved_stream, saved_tile, saved_room
    if jax.process_index() == 0:
        import json
        print('RESPONSE_SUM_FACES PASS ' + json.dumps(receipts), flush=True)
    return 0


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
