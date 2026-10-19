"""CPU pin of the CD consumer (gw.cd.sigma.compute_sigma_c_cd_omega_grid) across band tiles and knot chunks.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated CPU mesh checks
device-count logic).  A full-BZ toy (3x2x1 grid, nk = 6 on P = 4: q padded to 8; 14 carrier bands, Sigma
window 2..6, l sum 0..12, spinors) with chi0 replaced at its seam (``gw.cd.sigma.bank_chi0``) by a fixed analytic function of z
(this pin is about the layout, not chi0).  Pinned: the Sigma_c cube and the own-energy diagonal are the same, to
<= 1e-13 relative, for band tiles 1, 3, 5 and knot chunks 1 (below P / gcd(nq_irr, P) = 2: the (q, z) flat axis
takes zero rows), 2 and 3 (rounded up to 4), as for the planner's untiled one-chunk run; the receipt records
the tile, the slab widths and the padded flat rows.
"""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r'''
    import json, os, tempfile
    from types import SimpleNamespace
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.cd import sigma as cd
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(31)
    grid, nk, nb, s, n = (3, 2, 1), 6, 14, 2, 8
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    enk = np.sort(rng.uniform(-0.5, 0.6, (nk, nb)), axis=1)
    enk[:, 5:] += 0.15                                                 # a gap above band 5
    occ = (np.arange(nb)[None, :] < 5).astype(float) * np.ones((nk, 1))
    efermi = 0.5 * (enk[:, 4].max() + enk[:, 5].min())
    psi = c(nk, nb, s, n) / 4
    face4 = NamedSharding(mesh, P(None, "x", None, "y")); face3 = NamedSharding(mesh, P(None, "x", "y"))
    A = c(nk, n, n) / 6
    Vq = np.einsum("qij,qkj->qik", A, A.conj()) + 0.3 * np.eye(n)[None]
    kq = np.array([[(np.ravel_multi_index(tuple((np.array(np.unravel_index(k, grid)) - np.array(np.unravel_index(q, grid))) % grid), grid))
                    for k in range(nk)] for q in range(nk)])
    sym = SimpleNamespace(q_irr_full_idx=np.arange(nk), kq_map=kq.T, irr_idx_k=np.arange(nk), sym_idx_k=np.zeros(nk, int),
                          sym_mats_k=np.zeros((2, 3, 3)), nk_red=nk)
    meta = SimpleNamespace(kgrid=grid, nkx=3, nky=2, nkz=1, nk_tot=nk, nspin=1, nspinor_wfnfile=2, b_id_4_chi_user=12,
                           mu_basis=SimpleNamespace(n_packed=n, n_logical=n, pack_operator=None))
    slices = SimpleNamespace(sigma=slice(2, 6), sigma_sum=slice(0, 12), full=slice(0, nb), b0=0, b4_logical=12)
    wfns = SimpleNamespace(enk=jnp.asarray(enk), occ=jnp.asarray(occ), slices=slices,
                           psi_nmu=jax.device_put(psi, face4), green_parent=None)
    B = c(nk, n, n) / 40

    def fake_bank_chi0(wfns_, meta_, mesh_, z_ry, *, q_ids, mu_ry, rule_tolerance, slots, windows=None,
                       rule_capacity=None, rules_file=None, time_reversal=False, save_path=None, allow_partial=False,
                       band_windows=False, remote_margin_ry=None, print_fn=print):
        face4 = NamedSharding(mesh_, P(None, None, "x", "y"))

        def chi(points):
            z = np.asarray(points, np.complex128).ravel()[:, None, None, None]
            return jax.device_put(B[None] / (z - 0.9) - np.conj(np.swapaxes(B, 1, 2))[None] / (z + 0.9), face4)
        return chi, dict(source="analytic test chi0", points=int(np.asarray(z_ry).size), windows=len(windows or []),
                         rule_groups=[dict(sampled_error=0.0)], rule_nodes=0, rule_tolerance=float(rule_tolerance),
                         band_windows=None)
    cd.bank_chi0 = fake_bank_chi0
    omega = np.linspace(-0.25, 0.3, 5)

    def run(band_tile, knot_chunk):
        d = tempfile.mkdtemp()
        handle = dict(V_q=jax.device_put(Vq, face3), imaginary_nodes=6, knot_step_ev=0.5, residue_reach_ev="auto",
                      knot_chunk=knot_chunk, band_tile=band_tile, run_dir=d)
        res = cd.compute_sigma_c_cd_omega_grid(wfns, handle, meta, mesh, sym=sym, config=None, omega_grid_ry=omega,
                                               efermi_ry=efermi, regularization_width_ry=0.02, print_fn=lambda *a: None,
                                               run_dir=d)
        rec = json.load(open(os.path.join(d, "cd_sigma_receipt.json")))
        own = np.load(os.path.join(d, "cd_sigma_own_energy.npz"))
        return np.asarray(res.sigma_c_kij), np.asarray(own["sigma_c_own_ry"]), np.asarray(own["sigma_x_kmn_ry"]), rec
    base = run(0, 0)
    out = dict(J=base[3]["J"], chunks_planner=[base[3]["chunk_imag"], base[3]["chunk_real"]], L_neg=base[3]["psi_slab_bands"][1])
    err = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    for bt, kc in ((1, 1), (3, 1), (5, 3), (3, 2), (0, 1)):
        cube, own, sx, rec = run(bt, kc)
        out[f"tile{bt}_chunk{kc}"] = dict(cube=err(cube, base[0]), own=err(own, base[1]), sigma_x=err(sx, base[2]),
                                         chunks=[rec["chunk_imag"], rec["chunk_real"]], flat=rec["qz_flat_rows"],
                                         band_tile=rec["band_tile"])
        assert max(err(cube, base[0]), err(own, base[1]), err(sx, base[2])) <= 1e-13, (bt, kc, out)
        assert rec["band_tile"] == (bt or 12) and rec["psi_slab_bands"][0] == 12
    assert out["tile1_chunk1"]["chunks"] == [1, 1] and out["tile1_chunk1"]["flat"] == [8, 8]     # 6 rows + 2 zero rows
    assert out["tile5_chunk3"]["chunks"] == [4, 4] and out["tile5_chunk3"]["flat"] == [24, 24]
    assert float(np.abs(base[1]).max()) > 0 and np.isfinite(base[0]).all()
    print("CDCONSUMER", json.dumps(out))
''')


def test_cd_consumer_tiles_and_chunks_on_four_cpu_devices():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=1500)
    assert run.returncode == 0 and "CDCONSUMER" in run.stdout, run.stdout[-4000:] + run.stderr[-4000:]
