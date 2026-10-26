"""The compile census refuses, by seam, a compile only some ranks make.

Every compile request is numbered and agreed across ranks (the compile
agreement), so ranks must request the same programs in the same order.  A
rank-local compile between collectives (an eager jnp op or a fresh jit on a
rank-varying shape) breaks that: the WS geometry upload (5a8abc87f) compiled one
``jnp.asarray`` per local displacement width, ranks visited different widths,
and CPU64/GPU4 hung in the next metadata gather; ``sweep_local_k`` stacked
each rank's k blocks eagerly (one program per block count) and an idle rank at
P > nk compiled nothing.  Two CPU processes with unequal rank-local work run
under ``LORRAX_COMPILE_CENSUS=1`` and a hard timeout: the WS codes and the k
sweep must pass every seam (before either fix they are refused), and a
deliberate rank-local eager upload must be refused with the seam's name.
Without the census the pre-fix WS code hangs here.
"""
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

pytest.importorskip("jax")

_RANK = textwrap.dedent("""
    import os
    import sys
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    rank, port, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=2, process_id=rank,
                               initialization_timeout=60)
    from common.jax_compile_cache import CompileAgreementError, install_compile_agreement
    install_compile_agreement()
    import jax.numpy as jnp
    from common.collectives import all_gather_processes, gather_k_blocks
    from common.ws_images import pair_ws_images, pair_ws_local_codes
    from runtime import tiles

    tiles.TILE_BYTES = 1 << 30
    images = pair_ws_images((1, 1, 1), np.eye(3), np.zeros((64 + 13, 3)), 3)
    # Unequal rank-local blocks: rank 0 visits width 64, rank 1 widths 64 and 8,
    # so before 5a8abc87f rank 1 alone compiled a second geometry upload.
    blocks = [(np.arange(8), np.arange(8))] + ([(np.array([-1]), np.arange(8))] if rank else [])
    try:
        pair_ws_local_codes(images, blocks)
        print("stage: ws codes", flush=True)
        # A round-robin k sweep: 3 k give the ranks 2 and 1 blocks (an eager
        # stack of each count), 1 k leaves rank 1 idle (no per-k compile at all).
        block = jax.jit(lambda k: jnp.full((2, 2), k, jnp.complex128))
        for n_k in (3, 1):
            table = gather_k_blocks(n_k, block, item_shape=(2, 2), label="census",
                                    print_fn=lambda *a: None)
            assert np.array_equal(table[:, 0, 0].real, np.arange(n_k)), table
        print("stage: k sweep", flush=True)
        if mode == "eager" and rank == 1:
            jnp.asarray(np.ones(5), dtype=jnp.float32)   # a compile only rank 1 makes
        all_gather_processes(np.array([rank]))
        jnp.zeros(4).block_until_ready()                           # the next shared compile
        all_gather_processes(np.array([rank]))
        print("stage: done", flush=True)
    except CompileAgreementError as exc:
        print("REFUSED", str(exc).splitlines()[0], flush=True)
        # The refused rank's peers wait in the seam; a launcher's abort ends
        # them (here, the test kills them), so skip the distributed shutdown.
        os._exit(3)
""")


def _run(mode, timeout=180):
    """Both ranks' ``(returncodes, outputs)``; the peers of a refused rank are killed."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    env = dict(os.environ, JAX_PLATFORMS="cpu", LORRAX_COMPILE_CENSUS="1",
               ISDF_JAX_CACHE_DIR="")
    logs = [tempfile.TemporaryFile("w+") for _ in range(2)]
    procs = [subprocess.Popen([sys.executable, "-c", _RANK, str(r), port, mode],
                              env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
             for r, log in enumerate(logs)]
    deadline = time.monotonic() + timeout
    while any(p.poll() is None for p in procs):
        if any(p.poll() == 3 for p in procs) or time.monotonic() > deadline:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            break
        time.sleep(0.2)
    outs = []
    for log in logs:
        log.seek(0)
        outs.append(log.read())
    if not any(p.returncode in (0, 3) for p in procs):
        raise AssertionError(f"{mode}: the ranks hung instead of refusing:\n" + "\n".join(outs))
    return [p.returncode for p in procs], outs


def test_census_passes_ws_codes_and_refuses_a_rank_local_compile():
    codes, outs = _run("ws")
    assert codes == [0, 0] and all("stage: done" in o for o in outs), outs
    codes, outs = _run("eager")
    assert codes[1] == 3, outs
    assert "REFUSED compile census: rank 1 requests compile" in outs[1], outs[1]
    assert "before host seam" in outs[1] and "jit_convert_element_type" in outs[1], outs[1]
