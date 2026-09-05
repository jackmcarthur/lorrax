"""Literal band-sum oracles for the shared Lorentz-block Sigma contraction."""
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
from gw.photon_sigma import contract_lorentz_blocks
from gw.wavefunction_bundle import BandSlices, Wavefunctions, PSI_MUN_SPEC, PSI_NMU_SPEC


def test_rectangular_lorentz_blocks_all_terms():
    """All sixteen blocks and their head sectors equal a literal k/band sum."""
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(90502)
    psi = [rng.normal(size=(2, 4, 4, mu)) + 1j * rng.normal(size=(2, 4, 4, mu))
           for mu in (4, 8)]
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    slices = BandSlices.from_band_edges(0, 0, 2, 3, 4)
    carriers = [Wavefunctions(
        psi_nmu=put(p, PSI_NMU_SPEC), psi_mun=put(p.transpose(0, 2, 3, 1), PSI_MUN_SPEC),
        enk=put(np.zeros((2, 4)), P()), occ=put(np.zeros((2, 4)), P()),
        slices=slices, layout="face") for p in psi]
    blocks = [(A, B) for A in range(4) for B in range(4)]
    gamma = [np.asarray(g) for g in (gamma0, gamma1, gamma2, gamma3)]
    weights = np.array([1.0, 0.4, 0.2, 0.0])
    Gij = put(np.broadcast_to(np.diag(weights[:3]), (2, 3, 3)), P())
    expected, heads = np.zeros((3, 2, 4, 4), complex), np.zeros((3, 3, 2, 3), complex)
    operators = {}
    for A, B in blocks:
        left, right = psi[int(A != 0)], psi[int(B != 0)]
        shape = (2, left.shape[-1], right.shape[-1])
        V = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        W = 1.3 * V + rng.normal(size=shape)
        operators[A, B] = V, W
        direct = np.einsum("st,kntm->knsm", gamma[A], left)
        conjugated = np.einsum("st,kntm->knsm", gamma[B], right).conj()
        sector = 0 if A == B == 0 else 1 if (A == 0) != (B == 0) else 2
        for term, interaction, weight, sign in ((0, V, weights, -0.5),
                (1, W, weights, -0.5), (2, W - V, np.ones(4), 0.25)):
            for k in range(2):
                for q in range(2):
                    g = np.einsum("nsm,ntv,n->smtv", direct[(k-q) % 2],
                                  conjugated[(k-q) % 2], weight)
                    value = sign * np.einsum("asm,smtv,btv,mv->ab",
                        left[k].conj(), g, right[k], interaction[q])
                    expected[term, k] += value
                    if q == 0:
                        heads[term, sector, k] += np.diag(value)[:3]

    def get_block(A, B):
        V, W = operators[A, B]
        return put(V, P(None, "x", "y")), put(W, P(None, "x", "y")), \
            put(V, P(None, "x", "y")), put(W, P(None, "x", "y"))

    with mesh:
        sig, sectors, head, total = contract_lorentz_blocks(
            blocks, carrier_C=carriers[0], carrier_T=carriers[1],
            plan_C=None, plan_T=None, term=(0, 1, 2), mesh_xy=mesh,
            meta=SimpleNamespace(kgrid=(2, 1, 1), nk_tot=2), Gij=Gij,
            get_block=get_block, with_q0_diagnostic=True, verbose=False)
    from jax.experimental.multihost_utils import process_allgather
    gather = lambda a: np.asarray(process_allgather(a, tiled=True))
    np.testing.assert_allclose(np.stack([gather(a) for a in sig]), expected, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(sum(gather(a) for a in sectors), expected[1] + expected[2], rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(np.array([[gather(a) for a in row] for row in head]), heads, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(np.array([gather(a) for a in total]), heads.sum(axis=1), rtol=1e-11, atol=1e-10)
