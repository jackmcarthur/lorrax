"""Host-only preparation checks, not native execution or material admission."""
import ast
from pathlib import Path
import runpy

import numpy as np

HARNESS=Path(__file__).with_name('verify_common_coherent_chi_p4.py')


def test_native_harness_has_typed_plans_real_execution_and_failure_safe_bootstrap():
    source=HARNESS.read_text();tree=ast.parse(source)
    namespace=runpy.run_path(str(HARNESS))
    assert {'literal','response_stage','main'}<=namespace.keys()
    assert 'xla_reference' not in source and 'monkeypatch' not in source
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    imports=[n for n in ast.walk(run) if isinstance(n,ast.Import)]
    bootstrap=next(n.lineno for n in imports if any(a.name=='gw.gw_jax' for a in n.names))
    assert all(bootstrap<n.lineno for n in imports if any(a.name in ('jax','numpy') for a in n.names))
    assert "jax.process_count()!=4" in source and "jax.local_device_count()!=1" in source
    assert "kconv_backend(driver.RUNTIME.mesh,(8,8,1))!='mathdx'" in source
    assert "k_unfold_plan=c['plan'],execute=execute" in source
    assert 'fresh_shared_pole_capacity' in source and '_bank_execution' in source
    assert 'moment_power' in source and 'literal_crop' in source
    for handler in (n for n in ast.walk(run) if isinstance(n,ast.ExceptHandler)):
        assert any(isinstance(n,ast.Raise) for n in handler.body)
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                       and n.func.id=='rank0_transaction' for n in ast.walk(handler))


def test_harness_literal_density_order_and_exact_moments_match_independent_indices():
    from tests._full_density_blocks_oracle import pair_spectrum,pair_time
    literal=runpy.run_path(str(HARNESS))['literal']
    rng=np.random.default_rng(911);grid=(3,1,1);coords=np.indices(grid).reshape(3,-1).T
    psi=.2*(rng.normal(size=(3,2,2,2))+1j*rng.normal(size=(3,2,2,2)))
    energy=np.array([[-.7,.3],[-.6,.4],[-.8,.5]])
    left=np.array([[[.6,.1j],[-.1j,.4]]]*3);right=np.eye(2)[None]-left
    times=np.array([.1+.2j,-.07+.13j])
    got=literal(psi,energy,left,right,coords,grid,times)
    for q in range(3):
        spectrum=pair_spectrum(psi,energy,left,right,(np.arange(3)-q)%3)
        np.testing.assert_allclose(got[q]/np.sqrt(3),pair_time(spectrum,times),rtol=2e-14,atol=2e-15)
        for power in (0,1):
            moments=literal(psi,energy,left,right,coords,grid,[0.],power)[q,0]
            delta,amplitude=spectrum
            want=np.einsum('t,tij->ij',delta**power,amplitude)
            np.testing.assert_allclose(moments/np.sqrt(3),want,rtol=2e-14,atol=2e-15)


def test_native_bank_execution_config_is_built_by_real_parser_not_empty_constructor():
    from gw.gw_config import LorraxConfig
    config=runpy.run_path(str(HARNESS))['bank_config']()
    assert isinstance(config,LorraxConfig)
    assert config.backend.linalg=='local'
    assert config.sigma.w_model=='shared_pole'
    assert config.bands.chi==750
