"""CPU pin of the CD checkpoint's rank-0 writer (``gw.cd.sigma.CdCheckpoint.write`` ->
``file_io.sigma_checkpoint.write_state_checkpoint``).

Unit cell: exempt from the four-GPU rule (an emulated four-device CPU mesh, like tests/test_cd_resume_window.py).
The accumulators are replicated, so rank 0 writes its own replica with serial h5py and publishes it by os.replace:
no SlabIO / parallel-HDF5 open (a collective open hung ~15 min on Lustre in the ZnO CD J3 leg).  The unchanged
``read_sigma_checkpoint`` authenticates the file and returns the accumulators bitwise, across a commit change with
the same accumulator code; other accumulator code is refused as foreign and the file is left in place.
"""
import os
import subprocess
import sys
import textwrap

ENV = dict(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", XLA_FLAGS="--xla_force_host_platform_device_count=4")

SCRIPT = textwrap.dedent(r'''
    import os, numpy as np, jax
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import gw.cd.sigma as cs
    import gw.kin_ion_io as kio
    import file_io.slab_io as slab_io
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rep = NamedSharding(mesh, P())
    path = os.path.join(os.environ["CKPT_DIR"], "tmp", "cd_checkpoint.h5")
    rng = np.random.default_rng(3)
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    accs = {k: jax.device_put(c(4, 2, 3, 3), rep) for k in ("acc_imag", "acc_real", "acc_anc")}
    accs.update({k: jax.device_put(c(2, 3), rep) for k in ("own_imag", "own_real", "own_anc")})

    parts = dict(deck_sha256="d" * 64, J=7, chunk_real=3, rows=[0, 1])
    kio._generator_commit = lambda: "a" * 40
    id_a = cs.cd_checkpoint_identity(**parts)
    kio._generator_commit = lambda: "b" * 40
    id_b = cs.cd_checkpoint_identity(**parts)

    # 2. the rank-0 writer: no SlabIO, the unchanged reader authenticates it, bitwise, across commits
    class NoSlabIO:
        def __init__(self, *a, **k):
            raise AssertionError("a state-only checkpoint opened SlabIO (collective parallel-HDF5)")
    slab_io.SlabIO = NoSlabIO
    ck = cs.CdCheckpoint(path, id_a, mesh, print_fn=print)
    ck.write("real", 5, accs, timings=dict(chi0=1.5), host=dict(n_real_chunks=9, tree="a" * 40))
    assert ck.writes == 1 and ck.nbytes > 0
    got = cs.CdCheckpoint(path, id_b, mesh, print_fn=print).read()
    assert got is not None and got["axis"] == "real" and got["chunk"] == 5, got
    for k, v in accs.items():
        assert np.array_equal(np.asarray(got["accs"][k]), np.asarray(v)), k
    assert got["host"]["tree"] == "a" * 40 and got["timings"] == dict(chi0=1.5)
    ck.write("real", 6, accs, host=dict(n_real_chunks=9))                 # rewrite in place (os.replace)
    assert cs.CdCheckpoint(path, id_b, mesh, print_fn=print).read()["chunk"] == 6
    assert not [p for p in os.listdir(os.path.dirname(path)) if p.endswith(".tmp")]

    # 3. other accumulator code: foreign, refused, left in place
    assert cs.CdCheckpoint(path, dict(id_a, accumulator_code="0" * 64), mesh, print_fn=print).read() is None
    assert os.path.exists(path)
    print("CKPT_WRITER_OK")
''')


def test_cd_checkpoint_rank0_writer(tmp_path):
    env = dict(os.environ, **ENV, CKPT_DIR=str(tmp_path))
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "CKPT_WRITER_OK" in run.stdout, run.stdout[-3000:] + run.stderr[-6000:]
