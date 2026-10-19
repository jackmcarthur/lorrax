"""CPU pins of the CD checkpoint identity (``gw.cd.sigma.cd_checkpoint_identity``).

Unit cell: exempt from the four-GPU rule (an emulated four-device CPU mesh, like tests/test_cd_resume_window.py).

1. Identity.  ``cd_checkpoint_identity`` holds the run's inputs and ``cd_accumulator_code_digest`` (the
   accumulator functions, the Sigma consumer up to ``CD_ASSEMBLE_MARKER``, the chi0/law/rule modules and
   ``gw.response_bank``), not the repository commit: a checkpoint written on one commit resumes on another whose
   accumulator code is the same (the ZnO CD finalize fix touched post-processing only), and a change to the
   accumulator code is refused as foreign and left in place.
"""
import os
import subprocess
import sys
import textwrap

ENV = dict(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", XLA_FLAGS="--xla_force_host_platform_device_count=4")

SCRIPT = textwrap.dedent(r'''
    import inspect, os, numpy as np, jax
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

    # 1. the digest: post-processing edits and the repository commit leave it; accumulator edits move it
    d0 = cs.cd_accumulator_code_digest()
    head, mark, tail = inspect.getsource(cs.compute_sigma_c_cd_omega_grid).partition(cs.CD_ASSEMBLE_MARKER)
    assert mark, "assemble marker missing"
    assert cs.cd_accumulator_code_digest({"compute_sigma_c_cd_omega_grid": head + mark + tail + "    # edit\n"}) == d0
    assert "cd_on_shell_body" not in cs.CD_ACCUMULATOR_FUNCTIONS
    assert cs.cd_accumulator_code_digest(
        {"compute_sigma_c_cd_omega_grid": head.replace("acc_imag", "acc_imag_", 1) + mark + tail}) != d0
    assert cs.cd_accumulator_code_digest({"stream_cd_chunks": inspect.getsource(cs.stream_cd_chunks) + "#"}) != d0
    assert cs.cd_accumulator_code_digest({"gw.response_bank": "edited"}) != d0
    try:
        cs.cd_accumulator_code_digest({"compute_sigma_c_cd_omega_grid": head})
        raise AssertionError("a consumer without the marker must refuse")
    except ValueError as exc:
        assert "GATE cd_accumulator_digest" in str(exc)
    parts = dict(deck_sha256="d" * 64, J=7, chunk_real=3, rows=[0, 1])
    kio._generator_commit = lambda: "a" * 40
    id_a = cs.cd_checkpoint_identity(**parts)
    kio._generator_commit = lambda: "b" * 40
    id_b = cs.cd_checkpoint_identity(**parts)
    assert id_a == id_b and "tree" not in id_a and id_a["accumulator_code"] == d0, id_a

    print("CKPT_IDENTITY_OK")
''')


def test_cd_checkpoint_identity_is_inputs_and_accumulator_code(tmp_path):
    env = dict(os.environ, **ENV, CKPT_DIR=str(tmp_path))
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "CKPT_IDENTITY_OK" in run.stdout, run.stdout[-3000:] + run.stderr[-6000:]
