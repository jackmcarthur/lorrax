"""Compile-only full-k or raw-parent producer lower bound at production Na shape.

Only the raw-parent option reads HDF5 metadata on the compute node.
No large matrices are loaded. A >3U result already refuses this
geometry; a <=3U result is not admission because native workspace and the
rest of the live application are deliberately absent from this lower bound.
"""
import json
import os
from pathlib import Path
import sys

from runtime import initialize_communicator_stack, finalize_process
initialize_communicator_stack()

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import resolve_mesh
from gw.w_isdf import _get_chi_fractional_contour_kernel_face
from gw.wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC


def main():
    mesh = resolve_mesh()
    # Packed size is the campaign's P4 fixture, not production padding policy.
    nk, nb, n, logical = 512, 88, 912, 896
    parent = None
    input_k = nk
    if "--parent" in sys.argv:
        from wfn_loader import WfnLoader
        from file_io.centroids import load_centroids
        from gw.centroid_k_unfold import build_centroid_k_unfold_plan
        data = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/11_metal86_p4_plus18_production_2026-09-04/qe')
        wfn = WfnLoader(str(data/'WFN.h5'), mesh=mesh)
        sym = wfn.symmetry()
        _, indices, logical = load_centroids(
            str(data/'centroids_frac_896_metal4_b86_fit86_req896.txt'), wfn.fft_grid)
        parent = build_centroid_k_unfold_plan(sym, indices, wfn.fft_grid, mesh,
                                              nspinor=1, parent_k_frac=wfn.kvecs(k='ibz'))
        input_k, n = parent.n_parent, parent.n_centroid_packed
    def shape(dimensions, dtype, spec):
        return jax.ShapeDtypeStruct(dimensions, dtype,
            sharding=NamedSharding(mesh, spec))
    args = (shape((1000,), np.float64, P()),
        shape((2, 1000), np.complex128, P()),
        shape((input_k, 1, n, nb), np.complex128, PSI_MUN_SPEC),
        shape((input_k, nb, 1, n), np.complex128, PSI_NMU_SPEC),
        shape((input_k, nb), np.float64, P()),
        shape((input_k, nb), np.float64, P()),
        shape((input_k, nb), np.float64, P()), shape((), np.float64, P()))
    kernel = _get_chi_fractional_contour_kernel_face(mesh, (8, 8, 8), 2,
        (input_k, nb, n, 1), selected_q=(0,), k_unfold_plan=parent)
    compiled = kernel.lower(*args).compile()
    memory = compiled.memory_analysis()
    unit = 16 * nk * logical**2 / mesh.size
    bound = (memory.argument_size_in_bytes + memory.output_size_in_bytes
             + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    result = dict(scope="compile-only producer, one q, value+derivative",
        carrier="raw-parent" if parent is not None else "full-k",
        input_k=input_k, packed=n, logical=logical,
        U_bytes=unit, compiler_lower_bound_bytes=bound, compiler_lower_bound_U=bound/unit,
        memory=str(memory),
        status="FAIL" if bound > 3*unit else "NOT_MEASURED",
        omitted="native workspace and application resident buffers",
        job=os.getenv("SLURM_JOB_ID"), step=os.getenv("SLURM_STEP_ID"))
    if "--compare-incumbent" in sys.argv:
        # Incumbent fractional MPA screening owner, identical carriers and
        # node/output counts; only the bank's selected-q route is absent.
        incumbent = _get_chi_fractional_contour_kernel_face(mesh, (8, 8, 8), 2,
            (input_k, nb, n, 1), k_unfold_plan=parent)
        inc_compiled = incumbent.lower(*args).compile()
        inc_memory = inc_compiled.memory_analysis()
        inc_bound = (inc_memory.argument_size_in_bytes
            + inc_memory.output_size_in_bytes + inc_memory.temp_size_in_bytes
            - inc_memory.alias_size_in_bytes)
        result["incumbent"] = dict(memory=str(inc_memory),
            compiler_lower_bound_bytes=inc_bound, compiler_lower_bound_U=inc_bound/unit,
            scope="existing fractional screening kernel, matched two outputs and1000 nodes")
        result["stream_peak"] = dict(status="PASS" if bound <= 1.05*inc_bound else "FAIL",
            ratio=bound/inc_bound, limit=1.05, inherited=True,
            scope="matched compiled lower bounds, includes output carriers")
        result["status"] = result["stream_peak"]["status"]
        result["bank_outputs"] = dict(bytes_per_rank=memory.output_size_in_bytes,
            U=memory.output_size_in_bytes/unit,
            status="NOT_MEASURED", reason="output-only; full new-stage ledger still owed")
        if jax.process_index() == 0:
            (Path(sys.argv[1])/'incumbent.hlo').write_text(inc_compiled.as_text())
    if jax.process_index() == 0:
        root = Path(sys.argv[1])
        (root/'receipt.json').write_text(json.dumps(result, indent=2))
        (root/'producer.hlo').write_text(compiled.as_text())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
    finalize_process()
