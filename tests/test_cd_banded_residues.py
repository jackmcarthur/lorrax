"""CPU pin of the banded real-axis residues (gw.cd.sigma.residue_band_tables + the banded ``real`` kernel)
against the dense contraction, on an emulated four-device 2x2 mesh.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated CPU mesh checks
device-count logic).  Pinned, on a synthetic metal-like system (occupied, fractional and empty bands; a
row subset of the k grid; q padded from 6 to 8 with garbage W rows):

* the chunk loop exactly as the driver runs it (tables per chunk, chunks no residue reads skipped) equals
  (a) the pinned laws summed densely over every knot, q and l, and (b) the same kernel fed full tables
  (every l on both branches), to <= 1e-13 relative, on the omega-grid cube and the own-energy diagonal;
* the tables are a superset of the rows the dense weights read (every chunk, q, branch), including
  residues placed exactly on a knot, at x = 0, and straddling a chunk edge;
* the case exercises chunks with both branches, one branch only, and no branch, and widths below L.
"""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r'''
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.cd import laws
    from gw.cd.sigma import build_contraction_kernels, contract, residue_band_tables, BAND_RUNG
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(11)
    nk, n, L, m, s = 6, 8, 20, 3, 2
    nq_pad, nq_loc = 8, 2
    eta, h = 0.25 / 13.6, 0.0625 / 13.6
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    rows = np.array([0, 2, 3, 5])                                     # file-wedge-like row subset
    kw = rows.size
    psi = c(nk, L, s, n) / 3
    # bands: 6 occupied (deep pair + shallow), 2 fractional near mu = 0, 12 empty
    enk = np.concatenate([rng.uniform(-0.215, -0.205, (nk, 2)), rng.uniform(-0.03, -0.008, (nk, 4)),
                          rng.uniform(-0.002, 0.003, (nk, 2)), rng.uniform(0.006, 0.05, (nk, 12))], axis=1)
    enk = np.sort(enk, axis=1)
    f = np.where(enk < -0.005, 1.0, 0.0)
    f[:, 6:8] = rng.uniform(0.2, 0.8, (nk, 2))
    f[1, 9] = 3e-9                                                    # a tiny fractional tail on an empty band
    omega = np.array([-0.004, 0.0, 0.004, 0.012])                     # absolute Ry, grid near mu only
    enk[2, 8] = 0.004                                                 # x = 0 exactly (right branch, knots 0, 1)
    enk[3, 8] = -5 * h                                                # x = 5 h exactly at omega = 0 (on a knot)
    kmq = np.array([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    qneg = np.array([(-q) % nk for q in range(nk)])
    eps_own = enk[rows, :m]
    chunk = 4
    # straddle a chunk edge: |x| = (chunk - 0.5) h for one own-energy residue
    eps_own = eps_own.copy(); eps_own[1, 2] = enk[kmq[0][rows[1]], 12] + (chunk - 0.5) * h
    E_all = np.concatenate([omega, eps_own.ravel()])
    rr = kmq[:, rows]                                                 # every pair the rows meet
    el, fl = enk[rr], f[rr]
    need = max(np.where(fl > 0, np.maximum(0.0, el - E_all.min()), 0.0).max(),
               np.where(1.0 - fl > 0, np.maximum(0.0, E_all.max() - el), 0.0).max())
    J = int(np.ceil(need / h)) + 2
    J_pad = -(-J // chunk) * chunk
    n_chunks = J_pad // chunk
    KMQ_pad = np.concatenate([kmq, kmq[:2] * 0]); qneg_pad = np.concatenate([qneg, [0, 0]])
    qvalid = np.r_[np.ones(nk), np.zeros(nq_pad - nk)]
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a), rep)
    T = dict(psi_L=put(psi), psi_M=put(psi[rows, :m]), KMQ=put(KMQ_pad.astype(np.int32)), kw=put(rows.astype(np.int32)),
             enk=put(enk), f=put(f), omega=put(omega), eps_own=put(eps_own), qneg=put(qneg_pad.astype(np.int32)),
             qvalid=put(qvalid), u_all=put(np.array([0.01])), w_all=put(np.array([0.02])))
    beta = tuple(float(b) for b in laws.BETA_RY)
    K = build_contraction_kernels(mesh, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h, beta=beta)
    tables, summary = residue_band_tables(enk, f, KMQ_pad, qneg_pad, rows, omega, eps_own, h, chunk, n_chunks)
    W = c(J_pad, nq_pad, n, n) / 4; dW = c(J_pad, nq_pad, n, n) / 4    # pad q rows 6, 7 are garbage on purpose
    z_all = h * np.arange(J_pad) + 1j * eta
    qpar4 = NamedSharding(mesh, P(None, ("x", "y")))
    zeros = lambda *sh: put(np.zeros(sh, complex))
    full = put(np.broadcast_to(np.arange(L, dtype=np.int32), (nq_pad, kw, L)).copy())
    nE = omega.size
    acc_b, own_b = zeros(nE, kw, m, m), zeros(kw, m)
    acc_f, own_f = zeros(nE, kw, m, m), zeros(kw, m)
    kinds = set()
    for ic in range(n_chunks):
        j0, j1 = ic * chunk, (ic + 1) * chunk
        Wc, dWc = jax.device_put(W[j0:j1], qpar4), jax.device_put(dW[j0:j1], qpar4)
        bands, negative = tables(ic)
        kinds.add(negative)
        if bands:
            acc_b, own_b = K.real(Wc, dWc, jnp.asarray(z_all[j0:j1]), jnp.int32(j0), acc_b, own_b, T,
                                  tuple(put(b) for b in bands), negative)
        acc_f, own_f = K.real(Wc, dWc, jnp.asarray(z_all[j0:j1]), jnp.int32(j0), acc_f, own_f, T, (full, full),
                              (False, True))
    # dense laws reference, and the superset certificate of the tables
    con = lambda rho, op: np.asarray(contract(jnp.asarray(rho), jnp.asarray(op), nk))
    ref, ref_o = 0, 0
    missed = 0
    for q in range(nk):
        r = kmq[q][rows]
        rho = np.einsum("klsu,kmsu->kmlu", psi[r].conj(), psi[rows, :m])
        x, xo, fq = omega[None, :, None] - enk[r][:, None, :], eps_own[:, :, None] - enk[r][:, None, :], f[r]
        rn = kmq[qneg[q]][rows]
        rho_n = np.einsum("klsu,kmsu->kmlu", psi[rn].conj(), psi[rows, :m])
        xn, xon, fn = omega[None, :, None] - enk[rn][:, None, :], eps_own[:, :, None] - enk[rn][:, None, :], f[rn]
        for j in range(J_pad):
            a, d = con(rho, W[j, q]), con(rho, dW[j, q])
            at, dt = con(rho_n, W[j, q].T), con(rho_n, dW[j, q].T)
            op_p, w_p = laws.real_knot_terms(j, z_all[j], a, d, x, fq, h, negative=False)
            op_n, w_n = laws.real_knot_terms(j, z_all[j], at, dt, xn, fn, h, negative=True)
            _, wo_p = laws.real_knot_terms(j, z_all[j], a, d, xo, fq, h, negative=False)
            _, wo_n = laws.real_knot_terms(j, z_all[j], at, dt, xon, fn, h, negative=True)
            ref = ref + np.asarray(laws.apply(jnp.concatenate([op_p, op_n], -1), jnp.concatenate([w_p, w_n], -1)))
            ref_o = ref_o + np.asarray(laws.own_apply(op_p, wo_p) + laws.own_apply(op_n, wo_n))
            bands, negative = tables(j // chunk)
            for b, (w_e, w_o) in enumerate(((w_p, wo_p), (w_n, wo_n))):
                read = (np.abs(np.asarray(w_e)).max(axis=1) + np.abs(np.asarray(w_o)).max(axis=1))   # [kw, 2L]
                read = (read[:, :L] + read[:, L:]) > 0                                               # [kw, L]
                kept = np.zeros((kw, L), bool)
                if (b == 1) in negative:
                    tab = bands[negative.index(b == 1)][q]
                    np.put_along_axis(kept, tab, True, axis=1)
                missed += int((read & ~kept).sum())
    def err(mine, theirs):
        mine, theirs = np.asarray(mine), np.asarray(theirs)
        return float(np.abs(mine - theirs).max() / max(np.abs(theirs).max(), 1e-300))
    out = dict(banded_vs_laws=err(acc_b, ref), banded_vs_laws_own=err(own_b, ref_o),
               banded_vs_full=err(acc_b, acc_f), banded_vs_full_own=err(own_b, own_f),
               full_vs_laws=err(acc_f, ref), full_vs_laws_own=err(own_f, ref_o))
    # red twin: the same chunk loop with every table replaced by the top Lb bands (the live low bands cut
    # off) must NOT reproduce the dense result -- the comparison above can see a dropped row
    acc_r, own_r = zeros(nE, kw, m, m), zeros(kw, m)
    for ic in range(n_chunks):
        j0, j1 = ic * chunk, (ic + 1) * chunk
        bands, negative = tables(ic)
        if bands:
            red = tuple(put(np.broadcast_to(L - 1 - np.arange(b.shape[-1], dtype=np.int32), b.shape).copy())
                        for b in bands)
            acc_r, own_r = K.real(jax.device_put(W[j0:j1], qpar4), jax.device_put(dW[j0:j1], qpar4),
                                  jnp.asarray(z_all[j0:j1]), jnp.int32(j0), acc_r, own_r, T, red, negative)
    out_red = err(acc_r, ref)
    assert out_red > 1e-3, out_red
    widths = [int(v) for key in summary["widths_pos_neg"] for v in key.split("/")]
    print("CDBANDED", out, "red_twin", out_red, "missed", missed, "kinds", sorted(kinds), "summary", summary,
          "J", J, "J_pad", J_pad)
    assert missed == 0, missed
    assert max(out.values()) <= 1e-13, out
    assert {(), (True,), (False, True)} <= kinds, kinds                # no branch, negative only, both
    assert any(0 < v < L for v in widths) and all(v == 0 or v == L or v % BAND_RUNG == 0 for v in widths), summary
    assert float(np.abs(np.asarray(ref)).max()) > 0 and float(np.abs(np.asarray(ref_o)).max()) > 0
''')


def test_banded_residues_match_dense_on_four_cpu_devices():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0 and "CDBANDED" in run.stdout, run.stdout[-4000:] + run.stderr[-4000:]
