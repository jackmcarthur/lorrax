"""Actual-WFN link batching and unchanged authenticated artifact contract."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
from tests.hsuite import rank_session

REPO = Path(__file__).resolve().parents[3]
NA = REPO/'tests'/'hsuite'/'fixture_na'
H2 = REPO/'tests'/'hsuite'/'fixture'/'WFN.h5'
CHECK = Path(__file__).with_name('parallel_transport_batch_check.py')


def test_parallel_transport_batches_preserve_links_and_artifact():
    def prepare(source,target):
        target.mkdir(parents=True)
        for p in source.iterdir():
            shutil.copy2(p,target/p.name)
        (target/'pt.in').write_text('''[cohsex]
centroids_file = centroids_frac_56.txt
number_bands_protected = 8
number_bands = 13
sys_dim = 3
bispinor = false
wfn_file = WFN.h5
occ_smearing_width_ry = 0.01
fermi_reference = mp1_fixed_n
''')
        return target
    out = rank_session.stage(NA,prepare)
    result = subprocess.run([sys.executable,str(CHECK),'--out',str(out),
                             '--spinor-wfn',str(H2)],
                            text=True,capture_output=True,check=False)
    rank_session.completed(result)
    for signature in ('Traceback (most recent call last)','RESOURCE_EXHAUSTED',
                      'CUDA_ERROR','MPI_Abort'):
        assert signature not in result.stdout+result.stderr
    rows = json.loads((out/'parity.json').read_text())
    assert len(rows) == 4
    assert [r['nspinor'] for r in rows if r['kind']=='actual_spinor_wfn'] == [2,4]
    artifact = next(r for r in rows if r['kind']=='bcc_artifact')
    assert artifact['nk'] == 27 and artifact['authenticated']
    assert artifact['schema_version'] == 4 and artifact['link_directions'] == 6
    assert [r['link_directions'] for r in artifact['connection_placement']] == [3,2,6]
    assert artifact['velocity_max_error'] < 5e-12
    reduction = artifact['validation_reductions']
    assert reduction['max_absolute_error'] < 2e-15
    assert reduction['tile_one_exact'] and reduction['ragged_last_panel_detected']
    assert reduction['failed_controls'] == [False, False, True, True]
    for row in rows:
        assert row['relative_link_error'] < 5e-10
        assert row['max_singular_value_error'] < 5e-12
