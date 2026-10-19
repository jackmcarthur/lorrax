"""CPU tests of the bsemat writer's driver: one load, several files, stage timers.

``run_bsemat_exports`` builds the shared stages once (``_export_context``) and
writes each file from them (``_write_output``). Here the shared stages are a
synthetic context (random Hermitian W/V, ψ, head columns and split vectors on
the emulated 2x2 CPU mesh), so the test exercises the per-file loop itself:

* every file of a multi-file export equals its one-file export bit for bit
  (TDA, extended, the identity-gauge red twin, the dipole q0 body, a downfold);
* a non-final file leaves the shared context untouched;
* the stage timers report every stage once per file and close on an exception;
* the CLI refuses contradictory flags before any runtime and names the files.

Unit cells: exempt from the four-GPU rule (emulated CPU mesh, no driver leg).
"""
import os
import subprocess
import sys
import time
import types

import h5py
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import common.timing as timing
from bse import bsemat_algebra as A
from bse import bsemat_export as X

OMEGA = 270.0
GRID = (2, 2, 1)
N_MU, N_MU_PAD = 22, 24
N_VAL, N_COND, NS = 2, 3, 2


def _mesh():
    from bse.bse_ring_comm import create_mesh_2d
    from common.collectives import mesh_is_emulated
    if jax.device_count() != 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    mesh = create_mesh_2d()
    if not mesh_is_emulated(mesh):
        pytest.skip("needs the emulated single-process mesh (serial SlabIO)")
    return mesh


def _herm_pd(rng, n, n_pad, nq):
    out = np.zeros((n_pad, n_pad, nq), complex)
    for q in range(nq):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        out[:n, :n, q] = a @ a.conj().T / n + 0.1 * np.eye(n)
    return out


def _unitary_blocks(rng, nk, nv, nb):
    U = np.zeros((nk, nb, nb), complex)
    for k in range(nk):
        for sl in (slice(0, nv), slice(nv, nb)):
            m = sl.stop - sl.start
            q, _ = np.linalg.qr(rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m)))
            U[k][sl, sl] = q
    return U


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return synthetic_arrays(tmp_path_factory.mktemp("bsemat_multi"))


def synthetic_arrays(d):
    """Host arrays of one synthetic export and the two small files it reads (written under ``d``)."""
    rng = np.random.default_rng(7)
    nk = int(np.prod(GRID))
    nb = N_VAL + N_COND
    k_int = np.stack(np.meshgrid(*[np.arange(n) for n in GRID], indexing="ij"), -1).reshape(-1, 3)
    iq_pair = np.array([[int(np.ravel_multi_index(tuple(np.mod(k_int[i] - k_int[j], GRID)), GRID))
                         for j in range(nk)] for i in range(nk)])
    keys = [(iq, (0, 0, 0)) for iq in range(nk)] + [(1, (-1, 0, 0))]       # one zone-boundary tie
    key_pair = iq_pair.copy()
    key_pair[0, 1] = len(keys) - 1
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    pad = lambda a: np.concatenate([a, np.zeros(a.shape[:-1] + (N_MU_PAD - N_MU,), complex)], -1)
    z_keys = pad(cplx(len(keys), N_MU))
    y_keys = pad(cplx(len(keys), N_MU))
    W11 = rng.uniform(0.2, 1.0, len(keys))
    for kk, (iq, _) in enumerate(keys):
        if iq == 0:                                     # the split leaves q = 0 untouched
            y_keys[kk] = 0.0
            W11[kk] = 0.0
    lam = rng.uniform(0.3, 1.0, (nk, nk))
    lam[iq_pair == 0] = 0.0
    W = _herm_pd(rng, N_MU, N_MU_PAD, nk)
    V = _herm_pd(rng, N_MU, N_MU_PAD, nk) + W
    Vbar = V[:, :, 0].real.astype(complex)
    psi_win = pad(cplx(nk, nb, NS, N_MU)) / np.sqrt(N_MU)
    # q -> 0 wings whose fold matches this W's Γ tile (bsemat_algebra.q0_dipole_body_rows).
    L = 0.05 * cplx(3, N_MU)
    Y = rng.standard_normal((3, 3))
    S_d = -0.002 * (Y @ Y.T + np.eye(3))
    S_eff = S_d + L @ W[:N_MU, :N_MU, 0] @ L.conj().T / OMEGA
    restart = str(d / "isdf_tensors_synthetic.h5")
    with h5py.File(restart, "w") as f:
        f["head_wing_L"], f["head_wing_R"] = L, L.conj().T
        f["S_cart_head"], f["S_cart_head_direct"] = S_eff, S_d
    wfn_co = str(d / "WFN_co.h5")
    with h5py.File(wfn_co, "w") as f:
        f["mf_header/versionnumber"] = np.int32(1)
        f["mf_header/kpoints/nrk"] = np.int32(nk)
    return dict(dir=d, restart=restart, wfn_co=wfn_co, nk=nk, keys=keys, iq_pair=iq_pair,
                key_pair=key_pair, z_keys=z_keys, y_keys=y_keys, W11=W11, lam=lam, W=W, V=V,
                Vbar=Vbar, psi_win=psi_win, Ug=_unitary_blocks(rng, nk, N_VAL, nb),
                b2l=np.array([2, 0, 3, 1]), kpts=k_int / np.asarray(GRID, float))


def _context(s, mesh, *, use_gauge_map=True):
    """The dict ``_export_context`` returns, from the synthetic host arrays (fresh device copies)."""
    from common.collectives import device_put_process_local
    put = lambda a, *spec: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, P(*spec)))
    W_flat = put(s["W"], "x", "y", None)
    nb = N_VAL + N_COND
    Ug = s["Ug"] if use_gauge_map else np.broadcast_to(np.eye(nb), (s["nk"], nb, nb)).copy()
    return dict(
        restart=s["restart"], sys_dim=3, bgw_eps=None, wfn=types.SimpleNamespace(close=lambda: None),
        wfn_co=s["wfn_co"], mesh_xy=mesh, rep=jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, P())),
        n_val=N_VAL, n_cond=N_COND, nk=s["nk"], grid=np.asarray(GRID), n_rmu=N_MU, n_rmu_pad=N_MU_PAD,
        nspinor=NS, omega=OMEGA, bvec=np.eye(3) * 2 * np.pi / 10.26, kpts=s["kpts"],
        kgrid_w=np.asarray(GRID), b2l=s["b2l"], ecutwfc=25.0, efermi=6.1,
        keys=list(s["keys"]), n_ties=1, iq_pair=s["iq_pair"].copy(), key_pair=s["key_pair"].copy(),
        lam_pair=s["lam"].copy(), z_keys=s["z_keys"].copy(), y_keys=s["y_keys"].copy(),
        W11=s["W11"].copy(), figs=[{"iq": 0, "q0": True}], split_stats=None,
        v_head_q=np.full(s["nk"], 3.0),
        W_flat=W_flat, V_flat=put(s["V"], "x", "y", None), Vbar=put(s["Vbar"], "x", "y"),
        psi_win=s["psi_win"].copy(), use_gauge_map=use_gauge_map, Ug=Ug,
        gfig={"gauge_map": "synthetic"} if use_gauge_map else {"gauge_map": "identity (red twin)"},
        q0_dipole={}, q0_update=jax.jit(lambda w, a, d: w.at[:, :, 0].set(
            A.add_q0_body_update(w[:, :, 0], a, d, xp=jnp)), out_shardings=W_flat.sharding))


def _datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
    return out


def _bitwise_equal(p, q):
    a, b = _datasets(p), _datasets(q)
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    for name in a:
        x, y = np.asarray(a[name]), np.asarray(b[name])
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert x.tobytes() == y.tobytes(), f"{name} differs: max |d| = {np.max(np.abs(x - y))}"


def _specs(d, tag):
    q0 = [1.0e-3, 2.0e-4, 0.0]
    return [X.output_spec(str(d / f"{tag}_tda.h5")),
            X.output_spec(str(d / f"{tag}_ext.h5"), extended=True),
            X.output_spec(str(d / f"{tag}_nomap.h5"), use_gauge_map=False),
            X.output_spec(str(d / f"{tag}_dip.h5"), q0_body="dipole", q0_vector=q0),
            X.output_spec(str(d / f"{tag}_df.h5"), mu_small=12),
            X.output_spec(str(d / f"{tag}_extdip.h5"), extended=True, q0_body="dipole", q0_vector=q0)]


def _export(monkeypatch, s, mesh, outputs):
    monkeypatch.setattr(X, "_export_context", lambda **kw: _context(s, mesh, use_gauge_map=kw["use_gauge_map"]))
    return X.run_bsemat_exports(input_file="unused.in", wfn_co=s["wfn_co"], n_val=N_VAL, n_cond=N_COND,
                                outputs=outputs, mesh_xy=mesh)


def test_multi_output_equals_separate_exports_bitwise(synthetic, monkeypatch):
    mesh = _mesh()
    d = synthetic["dir"]
    multi = _export(monkeypatch, synthetic, mesh, _specs(d, "multi"))
    single = [_export(monkeypatch, synthetic, mesh, [o])[0] for o in _specs(d, "single")]
    assert len(multi) == len(single) == 6
    for m, s in zip(multi, single):
        _bitwise_equal(m["out_path"], s["out_path"])
        assert m["frobenius"] == s["frobenius"], (m["out_path"], m["frobenius"], s["frobenius"])
        assert (m["n1b"], m["n2b"]) == (s["n1b"], s["n2b"])
        assert m["q0_dipole"] == s["q0_dipole"]
    # the files are distinct physics, not one file six times
    fro = [tuple(m["frobenius"].values()) for m in multi]
    assert len(set(fro)) == 6, fro
    assert multi[4]["downfold"]["mu_S"] <= 12 and multi[0]["downfold"] is None
    assert multi[3]["q0_dipole"]["fold_gate"] < 1e-10


def test_non_final_output_leaves_context_untouched(synthetic):
    mesh = _mesh()
    ctx = _context(synthetic, mesh)
    host = {k: v.copy() for k, v in ctx.items() if isinstance(v, np.ndarray)}
    dev = {k: np.asarray(ctx[k]) for k in ("W_flat", "V_flat", "Vbar")}
    ids = {k: ctx[k] for k in ("W_flat", "V_flat", "Vbar")}
    for o in _specs(synthetic["dir"], "ctx")[:5]:
        X._write_output(ctx, o, release=False)
    for k, v in host.items():
        assert ctx[k].tobytes() == v.tobytes(), k
    for k, v in dev.items():
        assert ctx[k] is ids[k] and np.asarray(ctx[k]).tobytes() == v.tobytes(), k
    X._write_output(ctx, _specs(synthetic["dir"], "ctx")[5], release=True)
    assert ctx["W_flat"] is None and ctx["V_flat"] is None
    assert len(ctx["q0_dipole"]) == 1                     # one q0 direction, computed once


def test_stage_timers_smoke(synthetic, monkeypatch):
    mesh = _mesh()

    def fake_context(**kw):
        for name in X.SHARED_STAGES:
            X._stage(name)
            time.sleep(0.01)
        X._stage(None)
        return _context(synthetic, mesh, use_gauge_map=kw["use_gauge_map"])

    monkeypatch.setattr(X, "_export_context", fake_context)
    d = synthetic["dir"]
    timing.reset()
    outs = [X.output_spec(str(d / "t_tda.h5")), X.output_spec(str(d / "t_df.h5"), mu_small=12)]
    ss = X.run_bsemat_exports(input_file="unused.in", wfn_co=synthetic["wfn_co"], n_val=N_VAL,
                              n_cond=N_COND, outputs=outs, mesh_xy=mesh)
    nk = synthetic["nk"]
    for s, o in zip(ss, outs):
        st = s["stages"]
        assert set(st["shared"]) == set(X.SHARED_STAGES)
        assert all(v >= 0.009 for v in st["shared"].values()), st["shared"]
        want = set(X.OUTPUT_STAGES) - (set() if o["mu_small"] else {"downfold"})
        assert set(st["output"]) >= want and all(v >= 0.0 for v in st["output"].values()), st["output"]
        assert 0.0 < sum(st["output"].values()) <= s["wall_total_s"] - s["wall_shared_s"] + 1e-3
        assert s["wall_shared_s"] <= s["wall_total_s"]
    rows = {r["path"]: r for r in timing.records()}
    assert rows[("bsemat.output",)]["count"] == 2
    assert rows[("bsemat.output", "rows")]["count"] == 2 * (1 + 4 * nk)    # setup + 4 per row
    assert rows[("bsemat.output", "exchange")]["count"] == 2 * (1 + nk)
    assert rows[("bsemat.output", "downfold")]["count"] == 1
    assert timing.current_path() == ()

    def failing_context(**kw):
        X._stage("restart_load")
        raise ValueError("GATE synthetic_refusal")

    monkeypatch.setattr(X, "_export_context", failing_context)
    with pytest.raises(ValueError, match="synthetic_refusal"):
        X.run_bsemat_exports(input_file="unused.in", wfn_co=synthetic["wfn_co"], n_val=N_VAL,
                             n_cond=N_COND, outputs=outs, mesh_xy=mesh)
    assert timing.current_path() == () and not X._OPEN_STAGE
    with pytest.raises(ValueError, match="distinct output paths"):
        X.run_bsemat_exports(input_file="unused.in", wfn_co=synthetic["wfn_co"], n_val=N_VAL,
                             n_cond=N_COND, outputs=[outs[0], outs[0]], mesh_xy=mesh)


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "bse.bsemat_cli", "-i", "x.in", "--wfn-co", "w.h5",
                           "--n-val", "8", "--n-cond", "8", *args],
                          capture_output=True, text=True, timeout=60, env=os.environ.copy())


def test_cli_refuses_contradictions_before_runtime():
    for args, msg in ((("--extended", "--kernel", "tda"), "--extended is --kernel extended"),
                      (("--no-gauge-map", "--gauge", "map"), "--no-gauge-map is --gauge identity"),
                      (("--mu-small", "off", "OFF"), "--mu-small repeats"),
                      (("--mu-small", "7y"), "--mu-small '7y'")):
        r = _cli(*args)
        assert r.returncode == 2 and msg in r.stderr, (args, r.returncode, r.stderr[-400:])
        assert "lorrax source closure" not in r.stdout          # no runtime was brought up


def test_cli_file_names():
    import ast
    import importlib.util
    src = open(importlib.util.find_spec("bse.bsemat_cli").origin).read()
    tree = ast.parse(src)
    # The CLI brings the runtime up at import; take only its pure argv functions.
    keep = [n for n in tree.body
            if (isinstance(n, ast.Import) and n.names[0].name in ("argparse", "itertools", "os"))
            or (isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["TAGS"])
            or (isinstance(n, ast.FunctionDef)
                and n.name in ("build_parser", "mu_small_of", "file_combinations"))]
    assert len(keep) == 7, [type(n).__name__ for n in keep]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "bsemat_cli_pure", "exec"), ns)
    ap = ns["build_parser"]()
    base = ["-i", "x", "--wfn-co", "w", "--n-val", "8", "--n-cond", "8", "-o", "out/bsemat.h5"]
    one = ns["file_combinations"](ap, ap.parse_args(base + ["--extended"]))
    assert [f["path"] for f in one] == ["out/bsemat.h5"] and one[0]["kernel"] == "extended"
    many = ns["file_combinations"](ap, ap.parse_args(base + ["--kernel", "tda", "extended",
                                                            "--mu-small", "off", "14x"]))
    assert [f["path"] for f in many] == ["out/bsemat_tda_muoff.h5", "out/bsemat_tda_mu14x.h5",
                                         "out/bsemat_ext_muoff.h5", "out/bsemat_ext_mu14x.h5"]
    assert ns["mu_small_of"]("14x", 16) == 224 and ns["mu_small_of"]("off", 16) is None


def test_red_twin_shared_state_leak_is_caught(synthetic, monkeypatch):
    """A file stage that edits the shared context in place (here the downfold scaling the
    parent's z columns) makes a later file differ from its one-file export: the comparison
    above can fail, and a file that differs in physics is not bitwise equal."""
    mesh = _mesh()
    d = synthetic["dir"]
    honest = X._downfold_stage

    def leaky(**kw):
        kw["z_keys"] *= 1.0 + 1e-9                    # in place: the shared z columns move
        return honest(**kw)

    monkeypatch.setattr(X, "_downfold_stage", leaky)
    outs = lambda tag: [X.output_spec(str(d / f"{tag}_df.h5"), mu_small=12),
                        X.output_spec(str(d / f"{tag}_tda.h5"))]
    multi = _export(monkeypatch, synthetic, mesh, outs("leak_multi"))
    single = _export(monkeypatch, synthetic, mesh, outs("leak_single")[1:])
    with pytest.raises(AssertionError, match="differs"):
        _bitwise_equal(multi[1]["out_path"], single[0]["out_path"])
    with pytest.raises(AssertionError, match="differs"):
        _bitwise_equal(multi[0]["out_path"], multi[1]["out_path"])     # other physics, same shape


def test_row_kernels_compile_once_per_shape(synthetic, monkeypatch):
    """Files of one shape (gauge twin, dipole body) reuse the compiled row kernels: the jit
    caches of ``_direct_row`` and ``_exchange_row`` grow by at most one entry for three files."""
    mesh = _mesh()
    if not hasattr(X._direct_row, "_cache_size"):
        pytest.skip("this jax has no jit _cache_size")
    d = synthetic["dir"]
    n0 = (X._direct_row._cache_size(), X._exchange_row._cache_size())
    _export(monkeypatch, synthetic, mesh,
            [X.output_spec(str(d / "j_tda.h5")), X.output_spec(str(d / "j_nomap.h5"), use_gauge_map=False),
             X.output_spec(str(d / "j_dip.h5"), q0_body="dipole", q0_vector=[1.0e-3, 0.0, 0.0])])
    n1 = (X._direct_row._cache_size(), X._exchange_row._cache_size())
    assert n1[0] - n0[0] <= 1 and n1[1] - n0[1] <= 1, (n0, n1)
