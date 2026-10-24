"""The two Σ sum-band layouts against the explicit (q,n,μ,ν,s,t) sum.

Run on a square mesh, four GPU ranks. Includes padded bands, signed masks,
empty/overlapping brackets, scanned row passes, and a complex antiunitary
Green partner from a preformed 2D right operand.
"""
import json
from types import SimpleNamespace

import runtime
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def main():
    mesh = runtime.initialize_communicator_stack().mesh
    from jax.experimental.multihost_utils import process_allgather
    from common.collectives import device_put_process_local
    from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
    from gw.ppm_tau_kernel import (_get_sigma_kij_kernel,
                                  sigma_kconv_tables, sigma_pass_price)
    from gw.subtile_stream import prepare_green_operands, green_operand_program
    from gw.greens_function_kernel import (build_G_parents, green_right_operand,
                                           face_green_product)
    from distrib_la import gemm_plan
    from symmetry_maps import QirrOperator
    from test_tdagw_dynsig_sigma import _plan, _kmq, KGRID, NMU
    import runtime.tiles as tiles

    rng = np.random.default_rng(7041)
    nk, nb, ns, mu, nb_sig = int(np.prod(KGRID)), 256, 2, NMU, 4
    c = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    psi = c((nk, nb, ns, mu)) / np.sqrt(nb * mu)
    psi[:, -1] = 0  # carrier tail, deliberately nonzero energy below
    energy = rng.uniform(-1, 1, (nk, nb))
    weight = rng.uniform(-0.3, 1.2, (nk, nb))
    weight[:, -1] = 0
    w = c((nk, mu, mu))
    w = w + w.conj().transpose(0, 2, 1)
    put = lambda x, spec: device_put_process_local(np.ascontiguousarray(x), NamedSharding(mesh, spec))
    nmu = put(psi, PSI_NMU_SPEC)
    mun = put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC)
    proj_nmu = put(psi[:, :nb_sig], PSI_NMU_SPEC)
    proj_mun = put(psi[:, :nb_sig].transpose(0, 2, 3, 1), PSI_MUN_SPEC)
    faces = (mun, nmu, proj_nmu, proj_mun)
    rep = lambda x: put(np.asarray(x), P())
    e, select = rep(energy), rep(weight)
    ref, t = 0.17, 0.31 + 0.09j
    kf = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1) / np.asarray(KGRID)
    kplan, _ = _plan(mesh, kf, rng)
    interaction = QirrOperator.of(put(w, P(None, 'x', 'y'))).with_load(mesh)
    g_load = sigma_kconv_tables(mesh, kplan)
    brackets = ((0, 5), (3, nb-2), (0, 0))
    # Exercise the scanned row path rather than only the whole-row fast path.
    tiles.TILE_BYTES = 1024
    receipts = []
    outputs = []
    programs, needs = {}, {}
    for layout in ('axis', 'face'):
        kernel = _get_sigma_kij_kernel(
            mesh_xy=mesh, kgrid=KGRID, brackets=brackets, face_shape=(nk, nb, mu, ns),
            face_band_extent=nb_sig, k_unfold_plan=kplan, q_wedge=interaction,
            sum_layout=layout)

        @jax.jit
        def node(prepared, e, select, interaction, g_load):
            return kernel(*prepared, e, select, ref, t,
                          interaction.values, None, interaction.load, g_load)

        def build(chosen, abstract):
            assert chosen == layout
            ex = node.lower(abstract, e, select, interaction, g_load).compile()
            receipts.append(dict(layout=chosen, passes=sigma_pass_price(kernel)['passes'],
                                 temp=ex.memory_analysis().temp_size_in_bytes))
            return ex, [ex], sigma_pass_price(kernel)['scratch']
        chosen, prepared, ex = prepare_green_operands(faces, mesh, build, label="Sigma", _force_layout=layout)
        programs[layout] = ex
        from runtime.aot_memory import compiled_new_bytes
        place = green_operand_program(mesh, layout).lower(*faces).compile()
        stats = place.memory_analysis()
        needs[layout] = max(compiled_new_bytes(place),
                            int(stats.output_size_in_bytes-stats.alias_size_in_bytes)
                            + compiled_new_bytes(ex, extra=sigma_pass_price(kernel)['scratch']))
        value = ex(prepared, e, select, interaction, g_load)
        outputs.append(np.asarray(process_allgather(value, tiled=True)))
        assert chosen == layout

    # Current-room injection selects by actual compiled prices; no compiler
    # byte constants or artificial program prices. No axis bank executes.
    from lxkit import memory
    assert needs['face'] < needs['axis'], needs
    forced_room = (needs['face'] + needs['axis']) // 2
    real_room = memory.room
    try:
        memory.room = lambda: forced_room
        def bounded(layout, abstract):
            return programs[layout], [programs[layout]], sigma_pass_price(kernel)['scratch']
        chosen, prepared, ex = prepare_green_operands(faces, mesh, bounded, label="Sigma")
        assert chosen == 'face' and needs[chosen] <= forced_room
        bounded_value = np.asarray(process_allgather(
            ex(prepared, e, select, interaction, g_load), tiled=True))
        np.testing.assert_allclose(bounded_value, outputs[1], rtol=1e-12, atol=1e-13)
    finally:
        memory.room = real_room

    phase = weight * np.exp(-1j * (energy - ref) * t)
    ref_sigma = np.zeros_like(outputs[0])
    kmq = _kmq()
    for b, (lo, hi) in enumerate(brackets):
        for k in range(nk):
            for q in range(nk):
                green = np.einsum('nsm,n,ntr->mstr', psi[kmq[q,k]],
                                  phase[kmq[q,k]] * ((np.arange(nb) >= lo) & (np.arange(nb) < hi)),
                                  psi[kmq[q,k]].conj())
                ref_sigma[b, k] -= np.einsum('ism,mstr,mr,jtr->ij', psi[k, :nb_sig].conj(),
                                             green, w[q], psi[k, :nb_sig]) / nk
    errors = [float(np.max(np.abs(v-ref_sigma)) / max(np.max(np.abs(ref_sigma)), 1e-30))
              for v in outputs]
    assert max(errors) < 5e-12, errors

    # Independent complex-weight antiunitary partner, using the very same face
    # owner but a supplied right operand (the new admissible combination).
    anti = SimpleNamespace(mesh_xy=mesh, n_full=nk, sym_idx=np.array([0, 1]), n_sym_spatial=1)
    gm = gemm_plan(mesh, m=mu*ns, k=nb, n=mu*ns, nq=nk,
                   dtype=jnp.complex128, layout='face', warmup=False)
    def prepared_product(a, b, *, weights):
        return face_green_product(a, b, mesh, weights, None, n_full=nk)

    from functools import partial
    @partial(jax.jit, static_argnames=('prepared',))
    def partner(a, b, weights, *, prepared=False):
        result = build_G_parents(
            a, None, phases=weights, layout='face', gemm=gm,
            k_unfold_plan=anti, real_weights=False, right=b,
            prepared_active_gemm=prepared_product if prepared else None)
        return result.G, result.transpose
    expected = np.einsum('knsm,kn,kntr->kmsrt', psi, phase, psi.conj())
    expected_t = np.einsum('knsm,kn,kntr->kmsrt', psi.conj(), phase, psi)
    anti_error = 0.0
    for prepared in (False, True):
        g, gt = (np.asarray(process_allgather(a, tiled=True))
                 for a in partner(mun, green_right_operand(nmu), rep(phase), prepared=prepared))
        anti_error = max(anti_error, float(np.max(np.abs(g-expected))),
                         float(np.max(np.abs(gt-expected_t))))
    assert anti_error < 5e-12, anti_error
    print('RESULT '+json.dumps(dict(rank=jax.process_index(), errors=errors,
                                    anti_error=anti_error, receipts=receipts, needed=needs, forced_room=forced_room)), flush=True)
    runtime.finalize_process(0)


if __name__ == '__main__':
    main()
