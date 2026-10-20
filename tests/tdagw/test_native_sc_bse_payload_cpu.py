"""Toy accepted-SC payload through real BSE/FFT/head/dipole owners, CPU4.

Only static screening is a declared toy; this is neither a CrI3 screening
test nor a GPU/multihost/material admission receipt.
"""
import importlib.util
import json
from pathlib import Path

import runtime  # noqa: F401
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture


def test_native_bse_callback_toy_static_payload_actual_solver(tmp_path, monkeypatch):
    from common.collectives import device_put_process_local
    from symmetry_maps import QirrOperator
    from gw.head_correction import HeadSample
    import gw.static_screening as static
    from ffi import fft

    spec = importlib.util.spec_from_file_location('native_bse_payload_callback',
        Path(__file__).parent/'cri3_full_microscopic/native_bse_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    case = _native_port_fixture(monkeypatch)
    mesh, meta, rng = case['mesh'], case['meta'], case['rng']
    nk, nb, nmu = case['nk'], case['nb'], case['nmu']
    put = lambda a, p: device_put_process_local(np.asarray(a), NamedSharding(mesh, p))
    # A real reciprocal toy body and a separate nonzero analytic toy head.
    bare = np.broadcast_to(np.eye(nmu)*.003, (nk, nmu, nmu)).copy().astype(complex)
    screened = np.broadcast_to(np.eye(nmu)*.025, bare.shape).copy().astype(complex)
    V = QirrOperator.whole_zone(put(bare, P(None, 'x', 'y')))
    W = QirrOperator.whole_zone(put(screened, P(None, 'x', 'y')))
    sample = HeadSample(vc0=.02+0.j, wcoul0=.01+0.j,
                        source='DECLARED_TOY_STATIC_NOT_MATERIAL', omega=0.j)
    calls = []
    def toy_static(wfns, interaction, **kwargs):
        assert wfns is None and interaction is V
        calls.append(kwargs)
        return W, sample
    monkeypatch.setattr(static, 'build_static_screened_w', toy_static)
    U = np.stack([np.linalg.qr(rng.normal(size=(nb, nb))+1j*rng.normal(size=(nb, nb)))[0]
                  for _ in range(nk)])
    psi_sc = np.einsum('kmsu,kma->kasu', case['psi'], U)
    wfns, _ = case['fixture']._gw_bundle(mesh, psi_sc[..., case['perm']],
        case['energy'], case['occ'], case['wfns'].green_parent.plan)
    a = rng.normal(size=(3, nk, nb, nb))+1j*rng.normal(size=(3, nk, nb, nb))
    velocity = a+a.conj().swapaxes(-1, -2)
    source = dict(mesh=mesh, meta=meta, config=case['fixture']._config(np.array([-2., 2.])),
        wfns=wfns, energy_ry=case['energy'], occupations=case['occ'],
        U_active=put(U, P()), velocity_dft_cart=put(velocity, P()),
        V_q=V, G0_canonical=np.array([.1, .2, -.1, .15], complex),
        frame=dict(reference_ready=True, frame='DECLARED_SYNTHETIC_ACCEPTED_INPUT'),
        W_by_role=dict(iteration_head='TOY_HEAD_ONLY',
                       shared_pole=dict(digest='TOY_STATIC_NO_PRODUCTION_BANK')))
    target = tmp_path/'toy_bse'
    with fft.xla_reference():
        result = probe.run_bse_probe(source, out=target, nval=1, ncond=1, nroots=1, maxiter=3)
    assert len(calls) == 1
    assert result['nk'] == nk and result['nval'] == result['ncond'] == 1
    assert np.asarray(result['energies_ev']).shape == (1,)
    assert np.isfinite(result['energies_ev']).all() and result['energies_ev'][0] > 0.
    assert max(result['residual_norm_ev']) < 1e-5
    assert np.sum(result['strength_xyz_bohr2_cell']) > 0.
    assert 'skipped' not in result['head']
    assert result['optical_approximation'] == 'full-protected U^H v_DFT U divided by current SC gap'
    saved = json.loads((target/'bright_subgap.json').read_text())
    assert saved['frame']['frame'] == 'DECLARED_SYNTHETIC_ACCEPTED_INPUT'
    assert saved['energies_ev'] == result['energies_ev']
