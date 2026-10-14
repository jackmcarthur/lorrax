"""CPU pin of the exact Lehmann chi0 face kernel (w_isdf._get_chi_fractional_q_kernel_face, ordered) against
a dense numpy evaluation, on an emulated four-device 2x2 mesh, with occupations that leave a tile pair whose
f_a - f_b vanishes at every k (skipped by the kernel) and a metal-like table with no dead tile pair.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; device-count logic only).
"""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r'''
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw.w_isdf import _get_chi_fractional_q_kernel_face
    assert jax.device_count() == 4
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(5)
    nk, nb, nbl, s, n, nz, tile = 4, 64, 60, 2, 8, 3, 32
    psi = (rng.standard_normal((nk, nb, s, n)) + 1j * rng.standard_normal((nk, nb, s, n))) / 3
    e = np.sort(rng.uniform(-1, 1, (nk, nb)), axis=1)
    row = np.array([(k + 1) % nk for k in range(nk)], np.int32)
    z = np.array([0.01 + 0.02j, -0.3 + 0.05j, 0.7 + 0.1j])
    def dense(f):
        eb, fb, pb = e[row], f[row], psi[row]
        w = (f[None, :, :, None] - fb[None, :, None, :]) / (e[None, :, :, None] - eb[None, :, None, :] + z[:, None, None, None])
        w[:, :, nbl:, :] = 0; w[:, :, :, nbl:] = 0
        d = np.einsum("kasm,kbsm->kmab", psi, pb.conj())
        return np.einsum("zkab,kmab,knab->zmn", w, d.conj(), d) / np.sqrt(nk)
    kern = _get_chi_fractional_q_kernel_face(mesh, nb_full=nb, nb_logical=nbl, pair_tile=tile, n_z=nz,
                                             k_unfold_plan=None, layout="face", ordered=True)
    out = {}
    for name in ("insulator", "metal"):
        f = np.zeros((nk, nb)); f[:, :6] = 1.0
        if name == "metal":
            f[:, 6:9] = rng.uniform(0.1, 0.9, (nk, 3)); f[2, 40] = 0.3      # partial occupation in tile 1 too
        put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
        got = kern(put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC), put(psi, PSI_NMU_SPEC), jnp.asarray(row),
                   put(e, P(None, None)), put(f, P(None, None)), jnp.asarray(z))
        ref = dense(f)
        out[name] = float(np.abs(np.asarray(got) - ref).max() / np.abs(ref).max())
    print("CHI0TILES", out)
    assert max(out.values()) < 1e-12, out
''')


def test_chi0_face_kernel_matches_dense_lehmann_with_dead_tiles():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "CHI0TILES" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
