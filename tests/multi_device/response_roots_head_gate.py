"""Batched invariant Coulomb roots and the shared-pole direct-head envelope."""
from runtime import initialize_communicator_stack, finalize_process
stack = initialize_communicator_stack(platform="gpu")
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from types import SimpleNamespace as NS
from gw.response_bank import _coulomb_algebra
from gw.gw_config import (ComputeMode, QPSolver, BispinorGWMode, HeadCorrection,
    ScreeningDiagrams, refuse_unsupported_bispinor_gw, incumbent_bispinor_head_record)
from common.collectives import gather_to_host
mesh=stack.mesh
rng=np.random.default_rng(473)
a=rng.normal(size=(5,8,8))+1j*rng.normal(size=(5,8,8))
v=a@a.swapaxes(-1,-2).conj()+np.eye(8)[None]
x=jax.make_array_from_callback(v.shape,NamedSharding(mesh,P(None,'x','y')),lambda ix:v[ix])
root=_coulomb_algebra(mesh,8,8,'local')
h,hi,neg,ranks=root(x)
actual=gather_to_host(h)
error=np.max(abs(actual@actual-v))
assert error<1e-11 and not bool(neg),(error,neg)
for q in range(5):
    single=root(x[q:q+1])[0]
    assert float(jnp.max(jnp.abs(single-h[q:q+1])))<1e-11
assert h.sharding.spec == P(None,'x','y')
cfg=NS(bispinor=True,bispinor_gw=BispinorGWMode.BARE_TRANSVERSE,
    head=NS(correction=HeadCorrection.NO_LOCAL_FIELDS), compute_mode=ComputeMode.MPA,
    sigma=NS(w_model='shared_pole'),screening=NS(diagrams=ScreeningDiagrams.W_RPA),
    qp_solver=QPSolver.SELF_CONSISTENT,density_self_consistent=True,sys_dim=3)
refuse_unsupported_bispinor_gw(cfg)
assert 'no wing/body fold' in incumbent_bispinor_head_record(cfg)[1]
for model,mode in [('mpa',BispinorGWMode.BARE_TRANSVERSE),('shared_pole',BispinorGWMode.FULL_STATIC_COHSEX)]:
    cfg.sigma.w_model=model;cfg.bispinor_gw=mode
    try:
        refuse_unsupported_bispinor_gw(cfg)
    except ValueError as exc:
        assert 'bispinor_head_correction_no_local_fields_unavailable' in str(exc)
    else:
        raise AssertionError('unrelated head route accepted')
if stack.process_index==0:
    print(f'PASS batched roots versus parent roots: H H error={error:.3e}; all-P sharding; shared direct head admission and unrelated-route refusals',flush=True)
finalize_process()
