"""CPU pin of the CD on-shell derivative: the dacc accumulators against a central difference of the CD Sigma itself.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated CPU mesh checks
device-count logic).  The full-BZ toy of tests/test_cd_consumer_tiles.py (3x2x1 grid, nk = 6 on P = 4, 14 bands,
Sigma window 2..6, l sum 0..12, spinors, analytic chi0 at the ``bank_chi0`` seam), with energies on a 2^-24 Ry grid
so E_F - E_F round trips exactly.  The on-shell request's ``pair`` table is each window state's own energy, so
every state has an x = E - eps_l(k - q) = 0 term (q = 0, l = m), where the residue sign and the anchor log branch
jump and only the sum is smooth; the request also carries E +- dE for dE = 1e-4, 1e-6 and 1e-8 Ry.  Pinned:

1. the derivative slot of :func:`gw.cd.sigma.cd_on_shell_body` (the dacc accumulators, closed form) equals the
   central difference of the value slots at E +- dE, the same CD Sigma_c, on every state; Sigma_CD is C1 (the
   Hermite knots and the x = 0 branch switch), so the difference converges as O(dE) (relative 3e-4, 3e-6, ~1e-7);
2. the derivative is the same across band tiles and knot chunks (<= 1e-13 relative), as the cube is.
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
    from gw.ppm_sigma import OnShellRequest
    import runtime.xla_memory as xm
    if not hasattr(xm, "pool_high_water"):          # removed by aab565c0a; gw.cd.sigma.ledger_sample still imports it
        xm.pool_high_water = lambda reset=False: None   # (separate fix pending); the ledger column reads -1
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(31)
    grid, nk, nb, s, n = (3, 2, 1), 6, 14, 2, 8
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    q24 = lambda v: np.round(np.asarray(v) * 2.0 ** 24) / 2.0 ** 24
    enk = np.sort(rng.uniform(-0.5, 0.6, (nk, nb)), axis=1)
    enk[:, 5:] += 0.15                                                 # a gap above band 5
    enk = q24(enk)
    occ = (np.arange(nb)[None, :] < 5).astype(float) * np.ones((nk, 1))
    efermi = float(q24(0.5 * (enk[:, 4].max() + enk[:, 5].min())))
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
    m0, m1 = 2, 6
    slices = SimpleNamespace(sigma=slice(m0, m1), sigma_sum=slice(0, 12), full=slice(0, nb), b0=0, b4_logical=12)
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

    # The request: pair = each window state's own energy (from E_F); E +- dE for two dE as further tables.
    pair = enk[:, m0:m1] - efermi
    assert np.array_equal((pair + efermi), enk[:, m0:m1]), "E_F round trip must be exact (x = 0 terms)"
    steps = (1e-4, 1e-6, 1e-8)
    tables = [pair] + [pair + sgn * d for d in steps for sgn in (1.0, -1.0)]
    names = ("pair", "p4", "m4", "p6", "m6", "p8", "m8")
    freqs, index = np.unique(np.stack(tables), return_inverse=True)
    request = OnShellRequest(tables_ry=np.stack(tables), names=names, aliases={k: i for i, k in enumerate(names)},
                             parent_ry=np.stack(tables), frequencies_ry=freqs,
                             parent_index=np.asarray(index, np.int32).reshape(np.stack(tables).shape),
                             covered_kn=np.ones(pair.shape, bool), omega_range_ev=(-10.0, 10.0))

    def run(band_tile, knot_chunk):
        d = tempfile.mkdtemp()
        handle = dict(V_q=jax.device_put(Vq, face3), imaginary_nodes=6, knot_step_ev=0.5, residue_reach_ev="auto",
                      knot_chunk=knot_chunk, band_tile=band_tile, run_dir=d)
        res = cd.compute_sigma_c_cd_omega_grid(wfns, handle, meta, mesh, sym=sym, config=None,
                                               omega_grid_ry=request.frequencies_ry, efermi_ry=efermi,
                                               regularization_width_ry=0.02, print_fn=lambda *a: None, run_dir=d,
                                               on_shell_request=request)
        slots = np.asarray(res.sigma_c_kij.slots).sum(axis=-1)[:, :, :m1 - m0]      # [n_slot, nk, nb]
        npz = np.load(os.path.join(d, "cd_sigma_own_energy.npz"))
        return slots, npz
    slots, npz = run(0, 0)
    deriv = slots[request.slot("derivative")]
    np.testing.assert_array_equal(npz["dsigma_c_domega_pair"], deriv)
    np.testing.assert_allclose(npz["omega_pair_ry"] - npz["mu_ry"], pair, rtol=0, atol=0)
    out = dict(max_abs_derivative=float(np.abs(deriv).max()))
    for d_, p, m in ((1e-4, "p4", "m4"), (1e-6, "p6", "m6"), (1e-8, "p8", "m8")):
        fd = (slots[request.slot(p)] - slots[request.slot(m)]) / (2.0 * d_)
        out[f"max_abs_vs_fd_{d_:g}"] = float(np.abs(deriv - fd).max())
        out[f"max_rel_vs_fd_{d_:g}"] = float(np.abs(deriv - fd).max() / np.abs(deriv).max())
    tiled, _ = run(3, 1)
    out["tiles_rel"] = float(np.abs(tiled[-1] - deriv).max() / np.abs(deriv).max())
    print("CDDERIV", json.dumps(out))
    # Sigma_CD is C1 (Hermite knots, the x = 0 branch switch), so the central difference converges as O(dE)
    for d_, tol in ((1e-4, 1e-3), (1e-6, 1e-5), (1e-8, 1e-6)):
        assert out[f"max_rel_vs_fd_{d_:g}"] < tol, (d_, out)
    assert out["tiles_rel"] <= 1e-13, out
''')


def test_cd_on_shell_derivative_is_the_derivative_of_the_cd_sigma():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=1500)
    print(run.stdout[-1500:])
    assert run.returncode == 0 and "CDDERIV" in run.stdout, run.stdout[-4000:] + run.stderr[-4000:]
