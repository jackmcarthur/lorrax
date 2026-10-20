"""Pure preflight checks; never initializes a communicator or material run."""
import ast
from pathlib import Path
import numpy as np
import pytest
import jax

from tests.tdagw import transition_galerkin_saved_state as cli


def _arguments():
    return cli.parser().parse_args(['--source','source.in','--snapshot','state.npz','--out','new',
        '--source-bound-bytes-per-rank',str(8*1024**3),
        '--lu-bound-bytes-per-rank',str(1024**3),
        '--device-bound-bytes-per-rank',str(32*1024**3)])


def test_declared_fixed_protocol_has_bounded_defaults():
    a=_arguments();cli.validate_args(a)
    assert a.basis_capacity<=512 and a.seed_width<=8 and a.wall_seconds<=600
    assert a.gamma_ev==.05 and a.eta_ev==.1 and a.b_lo==128


@pytest.mark.parametrize('key,value',[
    ('basis_capacity',513),('basis_capacity',0),('seed_width',9),('panel_width',33),
    ('wall_seconds',601),('b_lo',127),('gamma_ev',.1),('eta_ev',.2),
    ('fit_tolerance',.03),('source_bound_bytes_per_rank',0),
    ('lu_bound_bytes_per_rank',0),('host_bound_bytes_per_rank',0),
    ('device_bound_bytes_per_rank',9*1024**3),
])
def test_unsupported_or_unpriced_protocol_refuses(key,value):
    a=_arguments();setattr(a,key,value)
    with pytest.raises(ValueError,match='GATE transition_'):
        cli.validate_args(a)


def test_source_array_count_is_metadata_only_and_deduplicates_object_identity():
    a=jax.device_put(np.ones((2,3),dtype=np.complex128));tree=dict(first=a,second=a)
    tree['cycle']=tree
    assert cli.known_device_array_bytes(tree)==a.size*a.dtype.itemsize


def test_declared_device_bound_cannot_exceed_runtime_ledger():
    assert cli.bind_device_budget(16,32)==32
    with pytest.raises(ValueError,match='runtime budget'):
        cli.bind_device_budget(33,32)


def test_tiny_preflight_pipeline_on_four_CPU_devices():
    from jax.sharding import Mesh
    if len(jax.devices('cpu'))<4:
        pytest.skip('four CPU devices required for XY protocol smoke')
    mesh=Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2,2),('x','y'))
    receipt=cli.tiny_collective_preflight(mesh,linalg='local')
    assert receipt['status']=='PASS_TINY_ONLY'
    assert receipt['dense_backend']=='local'
    assert not receipt['material_admission']
    assert all(v<1e-8 for v in receipt['errors_absolute'].values())


def test_verification_driver_does_not_call_production_sigma_or_trajectory():
    tree=ast.parse(Path(cli.__file__).read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
           for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'window_sigma_c','make_native_snapshot_evaluator','screen_general_shared_poles',
                      'write_general_shared_pole_model','propagate'}
    assert {'transition_polarization','transition_dyson_value','dyson_transition_seeds',
            'transition_parent_model'}<=calls
    full=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='full' for t in n.targets)]
    assert any(isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.Add)
               and isinstance(n.value.right,ast.Name) and n.value.right.id=='wc' for n in full)
