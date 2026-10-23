"""CPU MPI WFN read uses the host library and matches the eager read exactly.

Run as a compute-node script at P4, after the CPU MPI prelude:
``python tests/test_wfn_cpu_libraries.py --wfn /path/to/WFN.h5``.
The accelerator-probe observer calls the real availability probe and fails
before an accidental CUDA/ROCm probe can load a library.
"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wfn", required=True)
    args = p.parse_args()
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from file_io import slab_io
    from ffi.common import ffi_loader
    from wfn_loader import IBZRows, WfnLoader
    assert jax.default_backend() == "cpu" and jax.process_count() > 1
    probes = []
    probe = slab_io.probe_read_availability

    def host_only(platform):
        probes.append(platform)
        assert platform == "cpu", f"CPU WFN selection probed accelerator leg {platform}"
        return probe(platform)

    slab_io.probe_read_availability = host_only
    with WfnLoader(args.wfn, mesh=stack.mesh) as auto:
        assert auto.backend == "phdf5"
        actual = auto.load(bands=(0, 4), k=IBZRows((0,)))
        actual.block_until_ready()
        with WfnLoader(args.wfn, mesh=stack.mesh, backend="eager") as eager:
            expected = eager.load(bands=(0, 4), k=IBZRows((0,)))
            expected.block_until_ready()
            assert np.array_equal(np.asarray(actual.addressable_shards[0].data),
                                  np.asarray(expected.addressable_shards[0].data))
    assert probes == ["cpu"], probes
    assert ffi_loader.loaded_platforms_in_order() == ["cpu"]
    # GW's staging receipt has no mesh argument. After CPU initialization,
    # it must follow the live backend even without the prelude's env pin.
    from ffi.io import staging_totals
    get_lib, library_requests = ffi_loader.get_lib, []

    def host_library(platform=None):
        assert platform in (None, "cpu"), f"CPU staging receipt requested {platform}"
        library_requests.append(platform)
        return get_lib(platform)

    ffi_loader.get_lib = host_library
    from_env = ffi_loader.platform_from_env
    # Model an unset environment in its reader; mutating os.environ after
    # MPI/XLA started native threads would violate runtime.env_guard.
    ffi_loader.platform_from_env = lambda default=None: default
    try:
        totals = staging_totals()
        assert totals is not None and min(totals) >= 0, totals
        assert library_requests
        assert ffi_loader.loaded_platforms_in_order() == ["cpu"]
    finally:
        ffi_loader.get_lib = get_lib
        ffi_loader.platform_from_env = from_env
    if jax.process_index() == 0:
        print("WFN_CPU_LIBRARIES PASS " + json.dumps(dict(probes=probes,
              platforms=ffi_loader.loaded_platforms_in_order(), shape=actual.shape,
              staging_totals_without_platform_env=totals)), flush=True)
    runtime.finalize_process(0)
    return 0


def test_wfn_cpu_libraries_needs_compute_node():
    import pytest
    pytest.skip("run on P4 CPU compute nodes with --wfn /path/to/WFN.h5")


if __name__ == "__main__":
    sys.exit(main())
