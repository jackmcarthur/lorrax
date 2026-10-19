"""CPU pins of the CD memory layout (gw.cd.sigma) on an emulated four-device 2x2 mesh.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated CPU mesh checks
device-count logic).  Pinned, on the metal-like fixture of tests/test_cd_banded_residues.py (L = 20 bands,
a row subset of the k grid, q padded from 6 to 8 with garbage W rows):
* band tiles (``band_tile``): the Sigma_x, imaginary-node and anchor kernels and the banded real-axis chunk
  loop at tiles 1, 3, 7, 8 and 19 equal the untiled kernels to <= 1e-14 relative (summation order only), a
  tile >= the band count equals them bit for bit, and one tiled case equals the dense laws to <= 1e-13;
* the pair slabs (``carrier_band_blocks`` + ``pair_psi_slabs``) from a face-sharded carrier with a band
  offset, a narrower x < 0 slab and a parent-unfold hook are the replicated-table rows bit for bit at
  every block width;
* the zero-row (q, z) flat (``qz_flat_programs``): live rows and roots are the unpadded layout bit for bit,
  the pad rows are zero and their Dyson rows exactly zero, and the live Dyson rows equal an unpadded chunk;
* ``knot_chunk`` keeps a chunk below P / gcd(nq_irr, P) and rounds the others up as before; every band-table
  entry lies below ``extent_pos_neg``.
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
    from gw.cd.sigma import (build_contraction_kernels, build_dyson_kernels, carrier_band_blocks, contract,
                             knot_chunk, pair_psi_slabs, qz_flat_programs, residue_band_tables)
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(23)
    nk, n, L, m, s = 6, 8, 20, 3, 2
    nq_pad, nq_loc = 8, 2
    eta, h = 0.25 / 13.6, 0.0625 / 13.6
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    rows = np.array([0, 2, 3, 5])
    kw = rows.size
    l0 = 2                                                             # the l sum starts at carrier band 2
    psi_all = c(nk, L + l0 + 2, s, n) / 3                              # carrier: 2 bands below, 2 above the sum
    psi = psi_all[:, l0:l0 + L]
    enk = np.concatenate([rng.uniform(-0.215, -0.205, (nk, 2)), rng.uniform(-0.03, -0.008, (nk, 4)),
                          rng.uniform(-0.002, 0.003, (nk, 2)), rng.uniform(0.006, 0.05, (nk, 12))], axis=1)
    enk = np.sort(enk, axis=1)
    f = np.where(enk < -0.005, 1.0, 0.0)
    f[:, 6:8] = rng.uniform(0.2, 0.8, (nk, 2))
    omega = np.array([-0.004, 0.0, 0.004, 0.012])
    kmq = np.array([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    qneg = np.array([(-q) % nk for q in range(nk)])
    eps_own = enk[rows, :m]
    KMQ_pad = np.concatenate([kmq, kmq[:2] * 0]); qneg_pad = np.concatenate([qneg, [0, 0]])
    qvalid = np.r_[np.ones(nk), np.zeros(nq_pad - nk)]
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a), rep)
    face = NamedSharding(mesh, P(None, "x", None, "y"))
    E_all = np.concatenate([omega, eps_own.ravel()])
    rr = kmq[:, rows]
    el, fl = enk[rr], f[rr]
    need = max(np.where(fl > 0, np.maximum(0.0, el - E_all.min()), 0.0).max(),
               np.where(1.0 - fl > 0, np.maximum(0.0, E_all.max() - el), 0.0).max())
    chunk = 4
    J = int(np.ceil(need / h)) + 2
    J_pad = -(-J // chunk) * chunk
    n_chunks = J_pad // chunk
    tables, summary = residue_band_tables(enk, f, KMQ_pad, qneg_pad, rows, omega, eps_own, h, chunk, n_chunks)
    ext = summary["extent_pos_neg"]
    out = {}

    # ---- band-table extent: every entry below it ------------------------------------------------------
    top = [-1, -1]
    for ic in range(n_chunks):
        bands, negative = tables(ic)
        for b, neg in zip(bands, negative):
            top[int(neg)] = max(top[int(neg)], int(b.max()))
    assert all(t < e for t, e in zip(top, ext)), (top, ext)
    L_neg = int(ext[1])
    assert 0 < L_neg < L, ext                                           # the x < 0 slab is narrower than L

    # ---- pair slabs: face carrier, band offset, narrow x < 0 slab, every block width ------------------
    carrier = jax.device_put(psi_all, face)
    ref_q, ref_n = psi[KMQ_pad[:, rows]], psi[KMQ_pad[qneg_pad][:, rows]][:, :, :L_neg]
    for bb in (1, 3, L):
        pq, pn = pair_psi_slabs(carrier_band_blocks(carrier, mesh), mesh, rows_q=KMQ_pad[:, rows],
                                rows_nq=KMQ_pad[qneg_pad][:, rows], l0=l0, L=L, L_neg=L_neg, block_bands=bb)
        assert pq.sharding.spec == P(("x", "y")), pq.sharding
        assert np.array_equal(np.asarray(pq), ref_q) and np.array_equal(np.asarray(pn), ref_n), bb
    # parent hook: 3 parents unfolded to 6 k by a row map and a phase (the role of unfold_wavefunction_local)
    par = c(3, L + l0, s, n)
    idx, ph = np.array([0, 1, 2, 2, 1, 0]), np.exp(1j * rng.uniform(0, 6, nk))
    unf = jax.jit(lambda a: a[idx] * ph[:, None, None, None], out_shardings=rep)
    full = np.asarray(unf(put(par)))                                    # the old route: unfold the replicated whole
    pq, pn = pair_psi_slabs(carrier_band_blocks(jax.device_put(par, face), mesh, unfold=unf), mesh,
                            rows_q=KMQ_pad[:, rows], rows_nq=KMQ_pad[qneg_pad][:, rows], l0=l0, L=L, L_neg=L_neg,
                            block_bands=3)
    assert np.array_equal(np.asarray(pq), full[:, l0:l0 + L][KMQ_pad[:, rows]])
    assert np.array_equal(np.asarray(pn), full[:, l0:l0 + L][KMQ_pad[qneg_pad][:, rows]][:, :, :L_neg])
    out["slabs"] = "bitwise"

    # ---- kernels: tiled vs untiled ---------------------------------------------------------------------
    psi_q, psi_nq = pair_psi_slabs(carrier_band_blocks(carrier, mesh), mesh, rows_q=KMQ_pad[:, rows],
                                   rows_nq=KMQ_pad[qneg_pad][:, rows], l0=l0, L=L, L_neg=L_neg, block_bands=5)
    _, psi_nq_full = pair_psi_slabs(carrier_band_blocks(carrier, mesh), mesh, rows_q=KMQ_pad[:, rows],
                                    rows_nq=KMQ_pad[qneg_pad][:, rows], l0=l0, L=L, L_neg=L, block_bands=5)
    u_all, w_all = np.array([0.01, 0.05, 0.3]), np.array([0.02, 0.06, 0.4])
    T = dict(psi_q=psi_q, psi_nq=psi_nq, psi_M=put(psi[rows, :m]), KMQ=put(KMQ_pad.astype(np.int32)),
             kw=put(rows.astype(np.int32)), enk=put(enk), f=put(f), omega=put(omega), eps_own=put(eps_own),
             qneg=put(qneg_pad.astype(np.int32)), qvalid=put(qvalid), u_all=put(u_all), w_all=put(w_all))
    beta = tuple(float(b) for b in laws.BETA_RY)
    qpar4 = NamedSharding(mesh, P(None, ("x", "y"))); qpar3 = NamedSharding(mesh, P(("x", "y")))
    zeros = lambda *sh: put(np.zeros(sh, complex))
    W = c(J_pad, nq_pad, n, n) / 4; dW = c(J_pad, nq_pad, n, n) / 4    # pad q rows 6, 7 are garbage on purpose
    z_all = h * np.arange(J_pad) + 1j * eta
    nE = omega.size
    full_t = put(np.broadcast_to(np.arange(L, dtype=np.int32), (nq_pad, kw, L)).copy())
    u, w = np.array([0.01, 0.05, 0.1, 0.4]), np.array([0.02, 0.04, 0.1, 0.2])

    def run(tile):
        K = build_contraction_kernels(mesh, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h, beta=beta, band_tile=tile)
        r = {}
        r["sigma_x"] = K.sigma_x(jax.device_put(W[1], qpar3), zeros(kw, m, m), T)
        acc, own = zeros(nE, kw, m, m), zeros(kw, m)
        for a0 in (0, 2):
            acc, own = K.imag(jax.device_put(W[a0:a0 + 2], qpar4), put(u[a0:a0 + 2]), put(w[a0:a0 + 2]), acc, own, T)
        r["imag"], r["imag_own"] = acc, own
        r["anchor"], r["anchor_own"] = K.anchor(jax.device_put(W[0], qpar3), jax.device_put(dW[0], qpar3),
                                                zeros(nE, kw, m, m), zeros(kw, m), T)
        acc, own = zeros(nE, kw, m, m), zeros(kw, m)
        accf, ownf = zeros(nE, kw, m, m), zeros(kw, m)
        for ic in range(n_chunks):
            j0, j1 = ic * chunk, (ic + 1) * chunk
            Wc, dWc = jax.device_put(W[j0:j1], qpar4), jax.device_put(dW[j0:j1], qpar4)
            bands, negative = tables(ic)
            if bands:
                acc, own = K.real(Wc, dWc, jnp.asarray(z_all[j0:j1]), jnp.int32(j0), acc, own, T,
                                  tuple(put(b) for b in bands), negative)
            if ic < 3:                                                  # full tables: the dense real axis
                accf, ownf = K.real(Wc, dWc, jnp.asarray(z_all[j0:j1]), jnp.int32(j0), accf, ownf,
                                    dict(T, psi_nq=psi_nq_full), (full_t, full_t), (False, True))
        r["real_banded"], r["real_banded_own"], r["real_full"], r["real_full_own"] = acc, own, accf, ownf
        return {k: np.asarray(v) for k, v in r.items()}

    def err(mine, theirs):
        return float(np.abs(mine - theirs).max() / max(np.abs(theirs).max(), 1e-300))
    base = run(None)
    widths = sorted({int(b.shape[-1]) for ic in range(n_chunks) for b in tables(ic)[0]})
    table = {}
    for tile in (1, 3, 7, 8, 19, L, 64):
        got = run(tile)
        table[tile] = {k: err(got[k], base[k]) for k in base}
        bit = {k: bool(np.array_equal(got[k], base[k])) for k in base}
        if tile >= L:
            assert all(bit.values()), (tile, bit)
        else:
            assert max(table[tile].values()) <= 1e-14, (tile, table[tile])
        for k in ("sigma_x", "imag", "imag_own", "anchor", "anchor_own"):
            if tile >= L:
                assert bit[k], (tile, k)
    # one tiled case against the dense laws (imaginary nodes, all q)
    tiled = run(3)
    ref, ref_o = 0, 0
    for q in range(nk):
        r = kmq[q][rows]
        rho = np.einsum("klsu,kmsu->kmlu", psi[r].conj(), psi[rows, :m])
        x, xo, fq = omega[None, :, None] - enk[r][:, None, :], eps_own[:, :, None] - enk[r][:, None, :], f[r]
        for iz in range(4):
            a = np.asarray(contract(jnp.asarray(rho), jnp.asarray(W[iz, q]), nk))
            ref = ref + np.asarray(laws.imag_raw_node(a, u[iz], w[iz], x, fq, eta))
            ref_o = ref_o + np.asarray(laws.imag_raw_node(a, u[iz], w[iz], xo, fq, eta, apply_fn=laws.own_apply))
    out["tiled3_vs_laws"] = max(err(tiled["imag"], ref), err(tiled["imag_own"], ref_o))
    assert out["tiled3_vs_laws"] <= 1e-13, out
    assert float(np.abs(base["real_banded"]).max()) > 0 and float(np.abs(base["anchor"]).max()) > 0

    # ---- (q, z) flat with zero rows, roots on the face, Dyson ------------------------------------------
    nq_irr = 2
    QZ = qz_flat_programs(mesh, nq_irr=nq_irr, n=n)
    face3, face4 = NamedSharding(mesh, P(None, "x", "y")), NamedSharding(mesh, P(None, None, "x", "y"))
    chi2 = c(2, nq_irr, n, n) / 9
    hh = c(nq_irr, n, n) / 3
    hh = hh + np.conj(np.swapaxes(hh, 1, 2))
    assert QZ.flat_rows(1) == 4 and QZ.flat_rows(2) == 4
    for nz in (1, 2):
        flat = QZ.to_qz(jax.device_put(chi2[:nz], face4))
        roots = QZ.roots_qz(jax.device_put(hh, face3), nz)
        live = np.swapaxes(chi2[:nz], 0, 1).reshape(nq_irr * nz, n, n)
        assert flat.shape[0] == 4 and roots.shape[0] == 4
        assert np.array_equal(np.asarray(QZ.rows(flat, nz)), live)
        assert np.array_equal(np.asarray(QZ.rows(roots, nz)), np.repeat(hh, nz, axis=0))
        assert not np.asarray(flat)[nq_irr * nz:].any() and not np.asarray(roots)[nq_irr * nz:].any()
    pair, value = build_dyson_kernels(mesh, n=n, pref=-0.7)
    wc1 = np.asarray(value(QZ.roots_qz(jax.device_put(hh, face3), 1), QZ.to_qz(jax.device_put(chi2[:1], face4))))
    wc2 = np.asarray(value(QZ.roots_qz(jax.device_put(hh, face3), 2), QZ.to_qz(jax.device_put(chi2, face4))))
    assert not wc1[nq_irr:].any()                                       # zero rows -> W_c exactly zero
    out["dyson_padded_vs_unpadded"] = err(wc1[:nq_irr], wc2[0::2])      # q-major rows: z = 0 of each q
    pw, pd = pair(QZ.roots_qz(jax.device_put(hh, face3), 1), QZ.to_qz(jax.device_put(chi2[:1], face4)),
                  QZ.to_qz(jax.device_put(chi2[1:], face4)))
    assert not np.asarray(pw)[nq_irr:].any() and not np.asarray(pd)[nq_irr:].any()
    assert out["dyson_padded_vs_unpadded"] <= 1e-14, out

    # ---- knot_chunk ---------------------------------------------------------------------------------
    assert [knot_chunk(v, 96, ndev=64, nq_irr=8) for v in (0, 1, 5, 7, 8, 9, 200)] == [1, 1, 5, 7, 8, 16, 96]
    assert [knot_chunk(v, 548, ndev=36, nq_irr=8) for v in (1, 8, 9, 10)] == [1, 8, 9, 18]
    assert [knot_chunk(v, 759, ndev=4, nq_irr=8) for v in (1, 6, 759, 900)] == [1, 6, 759, 759]
    print("CDTILES", out, "widths", widths, "extent", ext, "max_rel_vs_untiled",
          {t: max(v.values()) for t, v in table.items()})
''')


def test_cd_band_tiles_slabs_and_flat_padding_on_four_cpu_devices():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=1200)
    assert run.returncode == 0 and "CDTILES" in run.stdout, run.stdout[-4000:] + run.stderr[-4000:]
