"""Native-route version/cache plumbing; no material or GPU execution claim."""
import ast
from pathlib import Path

import numpy as np


def builder_tree():
    import gw.shared_pole_screening as owner
    tree=ast.parse(Path(owner.__file__).read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('screen_general_shared_poles','_general_current_fit')]
    assert len(functions)==2
    return owner,ast.Module(body=functions,type_ignores=[])


def test_production_metric_solver_is_explicit_only_for_eligible_equilibrium():
    import inspect
    from gw import shared_pole_constructor as constructor
    owner,tree=builder_tree()
    route=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='definite'
        and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name)
                and c.func.id=='reduce_equilibrium_pencil_device' for c in ast.walk(n)))
    code=compile(ast.Module(body=[route],type_ignores=[]),owner.__file__,'exec')
    calls=[]
    def reduce(kind):
        def record(*args,**kwargs):
            calls.append((kind,args,kwargs))
            return kind
        return record
    held=object()
    scope=dict(E=np.ones((1,2,2)),A=2*np.ones((1,2,2)),
        output_gram=3*np.ones((1,2,2)),iq=0,mesh_xy=object(),cap=2,
        recipe=dict(bank_rule_tolerance=1e-8,multiplet_relative_tolerance=1e-6),
        gamma=.03,spaces={0:held},current=dict(capacity=object()),eigh_backend='auto',
        moment_scale=10.,coordinate_units=(np.ones(2),np.ones(2)),
        reduce_equilibrium_pencil_device=reduce('equilibrium'),
        reduce_general_pencil_device=reduce('general'))
    for definite,expected in ((True,'equilibrium'),(False,'general')):
        scope['definite']=definite
        exec(code,scope)
        kind,args,kwargs=calls[-1]
        assert kind==scope['model']==expected
        np.testing.assert_array_equal(args[0],scope['E'])
        np.testing.assert_array_equal(args[1],scope['A'])
        assert kwargs['fixed_spaces'] is held
        assert kwargs['capacity'] is scope['current']['capacity']
        assert kwargs['rank_tolerance']==1e-8
        if definite:
            np.testing.assert_array_equal(args[2],scope['output_gram'])
            assert kwargs['gamma_ry']==.03
        else:
            assert len(args)==2 and kwargs['coordinate_units'] is scope['coordinate_units']
    # Neither production branch may silently retry the old CPU reducers.
    sources=[tree,ast.parse(inspect.getsource(constructor.reduce_equilibrium_pencil_device)),
             ast.parse(inspect.getsource(constructor.reduce_general_pencil_device))]
    forbidden={'collective_general_pencil_batch','collective_general_pencil',
               'reduce_general_pencil','_reduce_equilibrium_pencil','_equilibrium_accurate_metric',
               'np.linalg.eig','np.linalg.eigh','scipy.linalg.eig','scipy.linalg.eigh'}
    for source in sources:
        assert not forbidden & {ast.unparse(n.func) for n in ast.walk(source) if isinstance(n,ast.Call)}


def test_common_span_cache_never_reinterprets_legacy_whitened_or_general_spaces():
    owner,tree=builder_tree()
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='spaces')
    code=compile(ast.Expression(assignment.value),owner.__file__,'eval')
    legacy={0:dict(right=np.full((7,3),.7),left=np.full((7,3),.7))}
    general={0:dict(right=np.ones((5,2)),left=np.ones((5,2)))}
    old_columns={0:dict(right=np.eye(7)[:,:3],left=np.eye(7)[:,:3])}
    cache=dict(equilibrium_spaces=legacy,spaces=general,equilibrium_column_spaces_v1=old_columns)
    selected=eval(code,dict(fit_basis=cache,definite=True))
    assert selected is cache['equilibrium_device_spaces_v1']
    assert selected is not legacy and not selected
    common=np.eye(7)[:,:3][None]
    selected[0]=dict(right=common,left=common)
    device_general=eval(code,dict(fit_basis=cache,definite=False))
    assert device_general is cache['general_device_spaces_v1']
    assert device_general is not general and not device_general
    assert eval(code,dict(fit_basis=cache,definite=True)) is selected
    assert selected[0]['right'] is selected[0]['left'] is common
    assert cache['equilibrium_spaces'] is legacy
    assert cache['spaces'] is general and cache['equilibrium_column_spaces_v1'] is old_columns
    # A refresh passes CURRENT pencils into that held span, rather than
    # caching a model/whitening. Only the first returned original span wins.
    reuse=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='spaces.setdefault')
    eval_scope=dict(spaces=selected,iq=0,model=dict(spaces=dict(right=common*2,left=common*2)))
    exec(compile(ast.Module(body=[reuse],type_ignores=[]),owner.__file__,'exec'),eval_scope)
    assert selected[0]['right'] is common


def test_equilibrium_metric_and_cache_policy_are_scientific_identity_fields():
    from gw.shared_pole_constructor import EQUILIBRIUM_DEVICE_SOLVER
    owner,tree=builder_tree()
    update=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='general_gates' and n.func.attr=='update')
    keys={k.arg:ast.unparse(k.value) for k in update.keywords}
    assert keys['metric_solver']=='_EQUILIBRIUM_METRIC_SOLVER'
    assert EQUILIBRIUM_DEVICE_SOLVER=='same-state-device-metric-quotient-v1'
    assert keys['numerical_spaces']=="'original-coordinate-native-common-span-v1'"
    assert keys['common_state_selection']=="'source-tolerance-common-metric-quotient-v1'"
    assert keys['eigensolver']=="'native-positive-H-quotient-Hermitian-eigh'"
    source=ast.unparse(tree)
    assert 'EQUILIBRIUM_DEVICE_SOLVER as _EQUILIBRIUM_METRIC_SOLVER' in source
    assert "frequency_coordinates.update(metric_solver=_EQUILIBRIUM_METRIC_SOLVER" in source
    assert "recipe['recipe_hash'] = hashlib.sha256" in source
    assert 'full_density_bare_moments' in source
    assert "current['moments'] = (c1, c2)" in source
