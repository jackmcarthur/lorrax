"""Exercise the existing parent Sigma, vertex and Hall discrimination gates on P4."""
from pathlib import Path
import json
import sys
from runtime import initialize_communicator_stack, run_main_and_finalize

runtime = initialize_communicator_stack(platform='gpu')
worktree = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(worktree/'tests/multi_device'))
import bispinor_transverse_vertex_face_gate as vertex
import full_photon_head_sigma_gate as head
import jax


def main():
    results = {}
    for name, function in vertex._CLI_CELLS:
        results[name] = function(runtime.mesh, 'complex128')
        if jax.process_index() == 0:
            print(f'PASS {name}: {results[name]}', flush=True)
    wfn = '/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/41_bisp_parent_route_2026-09-05/02_preprocess/WFN.h5'
    results['head_hall_twins'] = head.run_gate(runtime.mesh, wfn, str(Path.cwd()))
    if jax.process_index() == 0:
        Path('combined_gate.json').write_text(json.dumps(results, indent=2)+'\n')
        print('COMBINED_SIGMA_GATE_PASS', flush=True)
    return 0


run_main_and_finalize(main)
