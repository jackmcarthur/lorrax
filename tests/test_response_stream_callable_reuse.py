"""A rebuilt response stream must consume current wavefunctions and energies."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np


def check_reuse(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
    from gw.response_bank import response_stream
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face

    def put(a, spec):
        sharding = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(a.shape, sharding, lambda i: a[i])

    rng = np.random.default_rng(331)
    psi = (rng.normal(size=(8, 1, 8, 4)) + 1j*rng.normal(size=(8, 1, 8, 4)))*0.1
    energy = np.broadcast_to(np.array([-0.4, -0.2, 0.3, 0.7]), (8, 4)).copy()
    meta = SimpleNamespace(nk_tot=8, nkx=2, nky=2, nkz=2, nspinor=1,
                           mu_basis=SimpleNamespace(n_packed=8))

    def wavefunctions(scale, shift):
        return SimpleNamespace(layout='face', green_parent=None,
            slices=SimpleNamespace(nb_full=4),
            psi_mun=put(psi*scale, PSI_MUN_SPEC),
            psi_nmu=put((psi*scale).transpose(0, 3, 1, 2), PSI_NMU_SPEC),
            enk=put(energy + shift*np.array([0., 0., 1., 1.]), P()))

    first = wavefunctions(1., 0.)
    second = wavefunctions(1.1, 0.05)
    options = dict(mesh_xy=mesh, q_ids=(0, 1), n_outputs=1)
    old, args0 = response_stream(first, meta, **options)
    new, args1 = response_stream(second, meta, **options)
    assert old is new
    assert args1[0] is second.psi_mun and args1[-1] is second.enk
    occupation = put(np.broadcast_to(np.array([1., 1., 0., 0.]), (8, 4)), P())
    prefix = (jnp.array([0.2, 0.7]), jnp.array([[1.+0j, 0.3+0j]]))
    suffix = (occupation, 1.-occupation, jnp.array(0.))
    a = old(*prefix, *args0, *suffix)
    b = new(*prefix, *args1, *suffix)
    reference = _get_chi_fractional_contour_kernel_face(
        mesh, (2, 2, 2), 1, (8, 4, 8, 1), selected_q=(0, 1))
    c = reference(*prefix, *args1, *suffix)
    error = float(jnp.max(jnp.abs(b-c)))
    changed = float(jnp.max(jnp.abs(b-a)))
    assert error < 1e-12 and changed > 1e-10
    assert response_stream(first, meta, **(options | {'q_ids': (1, 2)}))[0] is not old
    assert response_stream(first, meta, **(options | {'bank_carry': True}))[0] is not old
    assert response_stream(first, meta, **(options | {'pair_mode': 'laplace'}))[0] is not old
    return dict(status='PASS',maximum_absolute=error,changed_input_difference=changed,
                scope='P4 finite-response stream, current psi/energy red and cache-key geometry/mode/carry controls')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys, jax
        from common.collectives import resolve_mesh, barrier
        assert jax.process_count() == 4
        result = check_reuse(resolve_mesh())
        result['job_step'] = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
        barrier('response-stream-reuse')
    run_main_and_finalize(main)
