"""An explicit restart can recover a committed body after its head failed."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def test_interrupted_head_body_reuse_requires_explicit_restart_and_current_identity(tmp_path):
    from file_io.commit_state import assert_committed, set_commit_state

    source = Path(__file__).parents[1] / 'src/gw/tangential_screening.py'
    tree = ast.parse(source.read_text())
    functions = {n.name:n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    root = tmp_path / 'oneshot_tangential'
    root.mkdir()
    identity = dict(hamiltonian='current-H', recipe='same-recipe')
    recipe = dict(eta_ev=.25, z_ry=np.array([.1+.2j]), n=4)
    digest = 'current-bare-V'
    namespace = dict(root=root, run_dir=tmp_path, Path=Path, json=json, np=np,
                     assert_committed=assert_committed, _SWEPT=[True], photon=False,
                     identity=identity, sc_scratch=False, config=SimpleNamespace(restart=False),
                     recipe=recipe, coulomb=dict(sha256=digest),
                     log=SimpleNamespace(info=lambda *_: None))
    for name in ('_json', '_authenticate_committed_body', 'prepare_output'):
        exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(source), 'exec'), namespace)
    header = dict(identity=identity, recipe=json.loads(namespace['_json'](recipe)), n_q_irr=1,
                  construction_receipts=[dict(receipt=dict(q_receipts=[dict(identity=identity,
                    constructor=dict(coulomb=dict(sha256=digest)))]))])
    model = root / 'model.h5'
    with h5py.File(model, 'w') as h5:
        set_commit_state(h5, 1)
        h5.create_dataset('final_commit', data=1)
        h5.create_dataset('header_json', data=json.dumps(header))
    bank = dict(identity=identity, completion=True, coulomb_identity=dict(sha256=digest))
    moments = dict(bank, bank_complete=True)
    (root/'bank_receipt.json').write_text(json.dumps(bank))
    (root/'moments_receipt.json').write_text(json.dumps(moments))
    before = model.read_bytes()
    prepare = namespace['prepare_output']
    with pytest.raises(ValueError, match='GATE tangential_output: complete model'):
        prepare()
    namespace['config'].restart = True
    assert prepare() == 'committed'
    for changed in (dict(recipe, eta_ev=.5), dict(recipe, z_ry=np.array([.11+.2j]))):
        namespace['recipe'] = changed
        with pytest.raises(ValueError, match='GATE tangential_committed_resume: current identity or resolved recipe'):
            prepare()
    namespace['recipe'] = recipe
    namespace['coulomb'] = dict(sha256='different-bare-V')
    with pytest.raises(ValueError, match='GATE tangential_committed_resume: producer receipts'):
        prepare()
    namespace['coulomb'] = dict(sha256=digest)
    moments['coulomb_identity'] = dict(sha256='other-body')
    (root/'moments_receipt.json').write_text(json.dumps(moments))
    with pytest.raises(ValueError, match='GATE tangential_committed_resume: producer receipts'):
        prepare()
    (root/'moments_receipt.json').unlink()
    with pytest.raises(ValueError, match='GATE tangential_committed_resume: producer provenance'):
        prepare()
    (root/'moments_receipt.json').write_text(json.dumps(dict(bank, bank_complete=True)))
    namespace['identity'] = dict(identity, hamiltonian='different-H')
    with pytest.raises(ValueError, match='GATE tangential_output: complete model'):
        prepare()
    namespace.update(identity=identity, sc_scratch=True, config=SimpleNamespace(restart=False))
    assert prepare() == 'committed'
    namespace.update(photon=True, config=SimpleNamespace(restart=True))
    with pytest.raises(ValueError, match='GATE tangential_output: complete model'):
        prepare()
    # The producer's adjacent JSON cannot replace authenticated header binding.
    header['construction_receipts'][0]['receipt']['q_receipts'][0]['constructor']['coulomb']['sha256'] = 'old-V'
    with pytest.raises(ValueError, match='GATE tangential_committed_resume: committed parent receipts'):
        namespace['_authenticate_committed_body'](header, root, identity, recipe, digest)
    assert model.read_bytes() == before
