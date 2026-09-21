"""P4/P16 shared-pole charge stream parity for both band-storage layouts."""
from types import SimpleNamespace

from runtime import initialize_communicator_stack, finalize_process
rt = initialize_communicator_stack()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host
from common.wfn_layout import psi_specs
from gw.response_bank import response_stream


def main():
    mesh = rt.mesh
    rng = np.random.default_rng(2536)
    nk, nb, n, ns = 8, 8, 8, 2
    psi = (rng.normal(size=(nk, ns, n, nb))
           + 1j * rng.normal(size=(nk, ns, n, nb))) / 8
    right = psi.conj().transpose(0, 3, 1, 2)
    energy = np.broadcast_to(np.linspace(-1, 2, nb), (nk, nb)).copy()
    occupied = 1 / (1 + np.exp(energy / .3))
    meta = SimpleNamespace(nkx=2, nky=2, nkz=2, nk_tot=nk,
                           nspinor=ns, nspinor_wfnfile=ns,
                           mu_basis=SimpleNamespace(n_packed=n))

    def put(value, spec=P()):
        value = np.asarray(value)
        return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),
                                            lambda index: value[index])

    values = []
    for layout in ('face', 'axis'):
        nmu, mun = psi_specs(layout)
        wfns = SimpleNamespace(layout=layout, green_parent=None,
            psi_mun=put(psi, mun), psi_nmu=put(right, nmu), enk=put(energy),
            slices=SimpleNamespace(nb_full=nb))
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh,
            q_ids=(0, 3, 7), n_outputs=2, ordered=True, bank_carry=True)
        carry = put(np.zeros((2, 3, n, n), complex), P(None, None, 'x', 'y'))
        result = kernel(put([0., .3, 1.]),
            put([[1., .4j, .2], [.2j, 1., .7j]]), *fixed,
            put(occupied), put(1-occupied), put(0.), carry)
        assert result.sharding.spec == P(None, None, 'x', 'y')
        values.append(result)
    relative = float(gather_to_host(jnp.linalg.norm(values[1]-values[0])
                                    / jnp.linalg.norm(values[0])))
    assert np.isfinite(relative) and relative < 1e-11, relative
    if jax.process_index() == 0:
        print(f'CHARGE_RESPONSE_LAYOUT_PARITY relative={relative:.12e}', flush=True)


status = 1
try:
    main()
    status = 0
finally:
    finalize_process(status)
