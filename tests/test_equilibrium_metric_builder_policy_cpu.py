"""Host-only version/cache plumbing; no material or device claim."""
import ast
from pathlib import Path

import numpy as np


def builder_tree():
    import gw.shared_pole_screening as owner
    tree=ast.parse(Path(owner.__file__).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                  and n.name=='screen_general_shared_poles')
    return owner,function


def test_production_metric_solver_is_explicit_only_for_eligible_equilibrium():
    from gw.shared_pole_constructor import EQUILIBRIUM_METRIC_SOLVER
    owner,tree=builder_tree()
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='collective_general_pencil_batch')
    expression=next(k.value for k in call.keywords if k.arg=='structure')
    code=compile(ast.Expression(expression),owner.__file__,'eval')
    scope=dict(definite=True,gamma=.03,common_state_units=np.array([1.,.1]),moment_scale=10.,
               _EQUILIBRIUM_METRIC_SOLVER=EQUILIBRIUM_METRIC_SOLVER)
    policy=eval(code,scope)
    assert policy['metric_solver']=='original-columns-extended-newton-v1'
    assert policy['state_selection']=='principal-rrqr-v1'
    scope['definite']=False
    assert eval(code,scope) is None


def test_selector_cache_never_reinterprets_legacy_whitened_or_general_spaces():
    owner,tree=builder_tree()
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='spaces')
    code=compile(ast.Expression(assignment.value),owner.__file__,'eval')
    legacy={0:dict(right=np.full((7,3),.7),left=np.full((7,3),.7))}
    general={0:dict(right=np.ones((5,2)),left=np.ones((5,2)))}
    cache=dict(equilibrium_spaces=legacy,spaces=general)
    selected=eval(code,dict(fit_basis=cache,definite=True))
    assert selected is cache['equilibrium_column_spaces_v1']
    assert selected is not legacy and not selected
    selected[0]=dict(right=np.eye(7)[:,:3],left=np.eye(7)[:,:3])
    assert eval(code,dict(fit_basis=cache,definite=False)) is general
    assert eval(code,dict(fit_basis=cache,definite=True)) is selected
    assert cache['equilibrium_spaces'] is legacy
    # Retained host admission counts ALL families, even an old unused one.
    price=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='retained'
        and "equilibrium_column_spaces_v1" in ast.unparse(n.value))
    actual=eval(compile(ast.Expression(price),owner.__file__,'eval'),dict(fit_basis=cache))
    expected=sum(a.nbytes for family in cache.values() for space in family.values() for a in space.values())
    assert actual==expected


def test_equilibrium_metric_and_cache_policy_are_scientific_identity_fields():
    owner,tree=builder_tree()
    update=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='general_gates' and n.func.attr=='update')
    keys={k.arg:ast.unparse(k.value) for k in update.keywords}
    assert keys['metric_solver']=='_EQUILIBRIUM_METRIC_SOLVER'
    assert keys['numerical_spaces']=="'original-column-selector-P_I-v1'"
    source=ast.unparse(tree)
    assert 'EQUILIBRIUM_METRIC_SOLVER as _EQUILIBRIUM_METRIC_SOLVER' in source
    assert "frequency_coordinates.update(metric_solver=_EQUILIBRIUM_METRIC_SOLVER" in source
    assert "recipe['recipe_hash'] = hashlib.sha256" in source
