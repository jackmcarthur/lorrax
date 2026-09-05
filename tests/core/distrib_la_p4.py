"""Run the exact core CUDA matrix using the service's unchanged check bodies.

The complete provider matrix, including SLATE and ScaLAPACK, belongs to the
nightly service stage. Core requires all ten cells below to execute; a
provider refusal cannot satisfy its parent's count assertion.
"""
from pathlib import Path
import runpy


CORE_CELLS = {
    "resolution_promise", "factor_refusals", "cusolvermp_factor_solve",
    "cusolvermp_lu_factor_solve", "cusolvermp_hostile_extents",
    "batch_reshard_local_ops", "matmul_batch_reshard", "matmul_cublasmp",
    "gemm_plan_cublasmp", "gemm_plan_manual_shard_map",
}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(
        root / "services/distrib_la/tests/test_distrib_la_multiproc.py"))
    main = namespace["_cli_main"]
    cells = [row for row in namespace["_CLI_CELLS"] if row[0] in CORE_CELLS]
    assert {row[0] for row in cells} == CORE_CELLS, "stale core matrix roster"
    main.__globals__["_CLI_CELLS"] = cells
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
