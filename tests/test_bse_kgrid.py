"""Gates for the ``bse_k_grid`` coarse→fine general-BSE-init densification.

The knob lives in the GENERAL init
(``bse.bse_io.load_bse_data_from_restart_sharded``): when set and DIFFERENT
from the coarse restart grid it interpolates the WHOLE bundle (ψ/ε via
htransform, V_Q via vq_interp, W via zero-pad in R) so every solver runs on the
fine grid.  Unset or == coarse → the coarse bundle is returned byte-identically.

  1. **Parser** (fixture-free, CPU): ``_parse_grid_spec`` accepts
     ``"nx ny nz"`` / ``"nx,ny,nz"`` / tuple / empty / None, rejects the rest.
  2. **On-grid identity** (fixture+GPU): ``bse_k_grid == coarse`` returns a
     byte-identical bundle vs the no-flag path (the fast-path guarantee).
  3. **Densify** (fixture+GPU): a small 3×3→6×6 run returns a fine-grid bundle
     with the right shapes (nk=36, band/μ dims unchanged) and a W_q on the fine
     k-lattice, and it is solvable (finite, ordered, positive eigenvalues).
  4. **The slab refuses** (fixture+GPU): the fixture IS a slab, and on the
     DEFAULT head arm the same densification must REFUSE rather than
     re-attach the bulk 3-D pole.  Cells 2 and 3 therefore name the
     documented ``legacy`` arm — see ``LEGACY_HEAD``.

All 1-GPU (no 16-GPU gating).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")

FXDIR = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
         "05_lorrax_cohsex_native")

# Modest htransform window (nband=40 <= 48) so the fine-grid conduction ENERGY
# recovery stays accurate (10_lorrax_exciton_bands nband<=48 finding).
_KG_INPUT = """[cohsex]
restart = false
centroids_file = centroids_frac_640.txt
nval = 26
ncond = 14
nband = 40
sys_dim = 2
do_screened = true
screening_method = minimax
fermi_reference = midgap
sigma_at_dft_energies = true
memory_per_device_gb = 28
wfn_file = WFN.h5
bare_coulomb_cutoff = 30.0
"""


def test_parse_grid_spec():
    from bse.bse_io import _parse_grid_spec
    assert _parse_grid_spec(None) is None
    assert _parse_grid_spec("") is None
    assert _parse_grid_spec("   ") is None
    assert _parse_grid_spec("12 12 1") == (12, 12, 1)
    assert _parse_grid_spec("12,12,1") == (12, 12, 1)
    assert _parse_grid_spec("6, 6, 1") == (6, 6, 1)
    assert _parse_grid_spec((8, 8, 2)) == (8, 8, 2)
    assert _parse_grid_spec([4, 4, 1]) == (4, 4, 1)
    for bad in ("12 12", "1 2 3 4", "12x12x1"):
        with pytest.raises((ValueError,)):
            _parse_grid_spec(bad)


def _make_run(tmp_path):
    run = tmp_path / "kg"
    run.mkdir()
    os.symlink(f"{FXDIR}/tmp", run / "tmp")
    os.symlink(f"{FXDIR}/WFN.h5", run / "WFN.h5")
    os.symlink(f"{FXDIR}/centroids_frac_640.txt", run / "centroids_frac_640.txt")
    (run / "cohsex_kg.in").write_text(_KG_INPUT)
    return run


# PER-CELL, not module-level.  This was a ``pytestmark``, and a module-level
# skipif keyed on a fixture path collapses the FIXTURE-FREE cells too — the
# parser gate and the two AST red twins have no fixture and no GPU, and they
# were silently not running anywhere the MoS2 deck is absent (which is every
# box but one).  A whole module reading SKIPPED is the collection failure
# AGENT_PREAMBLE measurement rule 10 names; the marker belongs on the cells
# that actually read the deck.
needs_mos2_fixture = pytest.mark.skipif(
    not os.path.exists(f"{FXDIR}/tmp/zeta_q.h5"),
    reason="MoS2 3x3 640-centroid fixture not present")


#: THE SLAB HEAD ARM, and why three cells below have to name it.
#:
#: This fixture is MoS2: ``sys_dim = 2``.  The default ``w_head_densify = c1``
#: re-attaches W's Γ head analytically per fine q from ``gw.head_densify``,
#: whose integrand is the BULK pole ``8π/|q|²`` — and since 2026-08-17 that
#: stage REFUSES a slab rather than running the 3-D expression on it (in 2D
#: the head is a ``|q|`` cusp, so the error grew without bound as the fine
#: grid densified).  The three densifying cells below gate ψ/ε/V_q0/W shapes
#: and the exchange tile, none of which is the head channel, so they take the
#: documented ``legacy`` arm and stay exactly as informative as they were.
#: The refusal is gated on its own by
#: ``test_the_slab_refuses_the_bulk_head_rather_than_running_it``.
#:
#: It is passed as an ARGUMENT, not written into the deck, and that is not a
#: style choice: ``gw_config.read_lorrax_input`` builds params only out of
#: ``_DEFAULTS``, ``w_head_densify`` is not a key there, and so a deck line
#: saying ``w_head_densify = legacy`` is dropped on the floor and the run
#: silently takes ``c1``.  Registered in the sandbox's KNOWN_LORRAX_ISSUES
#: (2026-08-17); out of scope here.
LEGACY_HEAD = "legacy"


def _load(restart, inp, mesh_xy, bse_k_grid, w_head_densify=None):
    from bse.bse_io import load_bse_data_from_restart_sharded
    return load_bse_data_from_restart_sharded(
        restart, n_val=2, n_cond=2, mesh_xy=mesh_xy, input_file=str(inp),
        inject_head=True, bse_k_grid=bse_k_grid,
        w_head_densify=w_head_densify)


@needs_mos2_fixture
@pytest.mark.gpu
def test_on_grid_identity_byte_equal(tmp_path):
    """bse_k_grid == coarse → the bundle is byte-identical to the no-flag path."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)
    cg = (int(d0["nkx"]), int(d0["nky"]), int(d0["nkz"]))
    d1 = _load(restart, inp, mesh_xy, cg)          # == coarse → fast path
    for k in ("psi_c_X", "psi_v_X", "M_X", "eps_c", "eps_v", "W_q", "V_q0"):
        a, b = np.asarray(jax.device_get(d0[k])), np.asarray(jax.device_get(d1[k]))
        assert np.array_equal(a, b), f"{k} not byte-identical on the fast path"


@needs_mos2_fixture
@pytest.mark.gpu
def test_densify_3to6_shapes_and_solvable(tmp_path):
    """3×3→6×6 densification: fine-grid bundle shapes + a solvable spectrum."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from solvers.lanczos import block_lanczos_eig_jit
    import jax.numpy as jnp

    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)
    df = _load(restart, inp, mesh_xy, (6, 6, 1), LEGACY_HEAD)   # slab arm

    assert (int(df["nkx"]), int(df["nky"]), int(df["nkz"])) == (6, 6, 1)
    nk_f = 36
    # band / μ dims are grid-invariant; only the k axis (and W's k-axes) grow.
    assert df["n_val"] == d0["n_val"] and df["n_cond"] == d0["n_cond"]
    assert df["n_rmu_pad"] == d0["n_rmu_pad"]
    assert df["psi_c_X"].shape[0] == nk_f
    assert df["psi_v_X"].shape[0] == nk_f
    assert df["eps_c"].shape[0] == nk_f
    assert tuple(df["W_q"].shape[-3:]) == (6, 6, 1)
    assert df["V_q0"].shape == d0["V_q0"].shape        # μ-basis tile unchanged shape
    assert bool(np.all(np.isfinite(np.asarray(jax.device_get(df["eps_c"])))))

    # solvable: lowest TDA excitons finite, ordered, positive
    nc_pad, nv_pad = int(df["n_cond_pad"]), int(df["n_val_pad"])
    n_flat = nc_pad * nv_pad * nk_f
    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_stack_matvec(mesh_xy, 6, 6, 1, kernel="bse")
    W_R = jnp.fft.ifftn(df["W_q"], axes=(2, 3, 4), norm="ortho")
    bs = 4

    def mvb(Vb):
        X = Vb.reshape(bs, nc_pad, nv_pad, nk_f)
        X = jax.lax.with_sharding_constraint(X, sh.X)
        HX = matvec(X, df["psi_c_X"], df["psi_c_Y"], df["psi_v_X"], df["psi_v_Y"],
                    df["eps_c"], df["eps_v"], W_R, df["V_q0"], df["M_X"], df["M_Y"])
        return HX.reshape(bs, -1)

    evs, _ = block_lanczos_eig_jit(mvb, n_flat, n_eig=4, block_size=bs,
                                   max_iter=20, n_reorth=20)
    evs = np.asarray(jax.device_get(evs))[:4].real
    assert np.all(np.isfinite(evs))
    assert np.all(np.diff(evs) >= -1e-6)               # ascending
    assert evs[0] > 0.0                                # bound state above zero


@needs_mos2_fixture
@pytest.mark.gpu
def test_the_slab_refuses_the_bulk_head_rather_than_running_it(tmp_path):
    """A ``sys_dim = 2`` deck + ``bse_k_grid`` + the DEFAULT head arm REFUSES.

    This is the shipping configuration the defect lived in, on a real slab
    fixture: MoS2 3×3×1 densified to 6×6×1 with ``w_head_densify`` unset, so
    it resolves to ``c1``.  Before 2026-08-17 this computed — it re-attached
    the untruncated 3D pole ``8π/|q|²`` at every fine q inside the coarse Γ
    cell, where the true 2D head goes as ``8π·z_c/|q_∥|``, and printed a
    provenance ratio saying the head matched.  Nothing raised, and the error
    grew as the fine grid densified.

    Two things had to be true for that: ``Meta`` has no ``sys_dim`` field and
    ``bandstructure.htransform.initialize_wfns`` did not stamp one, and
    ``gw.head_densify._refuse_non_bulk`` returned on ``None``.  Both are
    fixed; this cell is the end-to-end statement that the refusal now reaches
    the deck.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"                 # w_head_densify unset → c1
    assert "w_head_densify" not in inp.read_text()
    assert "sys_dim = 2" in inp.read_text()
    mesh_xy = _create_mesh_xy(1, 1)
    with pytest.raises(NotImplementedError, match="SLAB_2D"):
        _load(restart, inp, mesh_xy, (6, 6, 1))    # no arm named → the default


# ─────────────────────────────────────────────────────────────────────────────
# The ``head_minibz_average`` key on the coarse→fine path (2026-08-09).
#
# Until 2026-08-09 ``_interpolate_bse_data_to_grid`` passed
# ``head_minibz_average=True`` UNCONDITIONALLY to ``build_vq_evaluator`` and
# then REPLACED ``V_q0`` outright, so (i) the opt-in, documented-defective
# scalar-<v> head average (tests/KNOWN_FAILURES.md) reached every coarse→fine
# user who never set the key, and (ii) the deck's exact disk body and its
# loader-injected ``vhead`` rank-1 head were silently discarded.  The three
# gates below are the red twins: each one FAILS on the pre-fix tree.
# ─────────────────────────────────────────────────────────────────────────────

def test_kgrid_head_key_is_read_not_forced():
    """RED TWIN (fixture-free, login-safe): the coarse→fine V_q0 leg must READ
    the deck ``head_minibz_average`` key, and every ``build_vq_evaluator`` call
    on that path must sit under a conditional.  Pre-fix the function contained
    no key read at all and the call was unconditional."""
    import ast
    import inspect
    from bse import bse_io

    tree = ast.parse(inspect.getsource(bse_io._interpolate_bse_data_to_grid))
    fn = tree.body[0]

    # (1) the deck key is read on this path
    reads = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "head_minibz_average"
    ]
    assert reads, (
        "_interpolate_bse_data_to_grid never reads the deck "
        "``head_minibz_average`` key — the coarse→fine path is forcing it")

    # (2) no build_vq_evaluator call escapes a conditional
    def _calls_under_if(node, guarded):
        out = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                out += _calls_under_if(child, True)
            else:
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "build_vq_evaluator"):
                    out.append(guarded)
                out += _calls_under_if(child, guarded)
        return out

    guards = _calls_under_if(fn, False)
    assert guards, "no build_vq_evaluator call found on the coarse→fine path"
    assert all(guards), (
        "a build_vq_evaluator call on the coarse→fine path is UNCONDITIONAL — "
        "the deck key is being ignored")


@needs_mos2_fixture
@pytest.mark.gpu
def test_densify_default_preserves_deck_vq0_bitwise(tmp_path):
    """RED TWIN (a): with ``head_minibz_average`` unset (the default), a
    coarse→fine densification must carry the deck's ``V_q0`` — disk body plus
    the loader-injected ``vhead`` rank-1 head — through BIT-IDENTICALLY.

    Pre-fix this tile was replaced by a vq_interp model reconstruction carrying
    an LR-only fine mini-BZ head, so the assert below was provably false."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"
    assert "head_minibz_average" not in inp.read_text()
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)                        # coarse 3×3×1
    df = _load(restart, inp, mesh_xy, (6, 6, 1), LEGACY_HEAD)      # fine 6×6×1

    a = np.asarray(jax.device_get(d0["V_q0"]))
    b = np.asarray(jax.device_get(df["V_q0"]))
    assert np.array_equal(a, b), (
        "coarse→fine REPLACED the deck's V_q0 with the key unset; "
        f"max|Δ| = {np.max(np.abs(a - b)):.6e}")

    # and the rest of the bundle really did densify (the fix is scoped to V_q0)
    assert (int(df["nkx"]), int(df["nky"]), int(df["nkz"])) == (6, 6, 1)
    assert df["psi_c_X"].shape[0] == 36
    assert tuple(df["W_q"].shape[-3:]) == (6, 6, 1)


@needs_mos2_fixture
@pytest.mark.gpu
def test_densify_optin_matches_prefix_construction(tmp_path):
    """GATE (b): with ``head_minibz_average = true`` explicitly set, the
    coarse→fine V_q0 must be BIT-IDENTICAL to the pre-fix construction — the
    opt-in path stays reachable and stays exactly as defective as documented.

    The reference is built here by replaying the pre-fix statements verbatim."""
    harness.skip_unless_gpu(pytest)
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from bse.bse_w_exact import _create_mesh_xy
    from bse import vq_interp

    run = _make_run(tmp_path)
    inp = run / "cohsex_on.in"
    inp.write_text(_KG_INPUT + "head_minibz_average = true\n")
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)
    df = _load(restart, inp, mesh_xy, (6, 6, 1), LEGACY_HEAD)   # slab arm

    # ── the pre-fix statements, replayed ────────────────────────────────
    n_rmu_pad = int(d0["n_rmu_pad"])
    vqm = vq_interp.build_vq_evaluator(
        restart, mesh_xy, n_rmu_pad, head_minibz_average=True, log_fn=print)
    gstar, head_val = vq_interp.minibz_head_vlr(
        vqm.zx, vqm.prep, np.zeros(3), kgrid=(6, 6, 1))
    ref = vqm.eval_vq(jnp.zeros(3), vqm.prep["V_SRc"], vqm.pinvF,
                      vqm.coeffs_packed,
                      jnp.asarray(head_val, dtype=jnp.float64),
                      jnp.asarray(gstar, dtype=jnp.int32))
    ref = 0.5 * (ref + jnp.conj(ref).T)
    ref = jax.device_put(ref, NamedSharding(mesh_xy, P("x", "y")))

    got = np.asarray(jax.device_get(df["V_q0"]))
    ref = np.asarray(jax.device_get(ref))
    assert np.array_equal(got, ref), (
        "the head_minibz_average=true arm no longer reproduces the pre-fix "
        f"construction; max|Δ| = {np.max(np.abs(got - ref)):.6e}")

    # the opt-in arm really does move the tile off the deck's
    assert not np.array_equal(got, np.asarray(jax.device_get(d0["V_q0"])))


# ─────────────────────────────────────────────────────────────────────────────
# The finite-q EXCHANGE tiles across a densification (2026-08-10).
#
# ``_interpolate_bse_data_to_grid`` nulls ``V_q_full`` because the fine-grid
# resolvent is not a use of it — but ``exciton_bands --vq-mode ongrid`` reads
# exactly that array for every Q != Gamma, so a densified bundle met it as a
# ``None`` subscript with no explanation.  The tiles ARE still available and
# still exact: V_{mu nu}(q) is a zeta/G-sphere object that never sees the BSE
# k-grid, so the coarse tile IS the fine solve's answer at any Q on the coarse
# grid.  What the coarse restart cannot supply is a tile at a fine q that is
# not a coarse q, and that is a REFUSAL rather than an interpolation.
# ─────────────────────────────────────────────────────────────────────────────

def test_densifier_keeps_the_coarse_exchange_tiles_with_their_grid():
    """The densifier must hand the coarse V_qmunu forward under its OWN grid
    stamp, and must NOT leave it in ``V_q_full`` (whose consumers index it with
    the bundle's — now fine — grid, so they would read the wrong tile)."""
    import ast
    import inspect
    from bse import bse_io

    fn = ast.parse(inspect.getsource(bse_io._interpolate_bse_data_to_grid)).body[0]
    keys = {n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "V_q_coarse" in keys and "V_q_coarse_grid" in keys, (
        "the densified bundle carries no coarse-exchange-tile key: "
        "exciton_bands --vq-mode ongrid has nothing to read")

    # ``V_q_full`` must still be set to None on this path — the red twin is
    # leaving the coarse array there, which every other consumer would index
    # with the fine grid.
    src = inspect.getsource(bse_io._interpolate_bse_data_to_grid)
    assert '"V_q_full": None' in src, (
        "V_q_full is no longer None on the densified bundle — bse_w_exact "
        "would index a coarse array with fine (nkx,nky,nkz) and silently read "
        "the wrong exchange tile")


def test_ongrid_exchange_is_indexed_by_the_tile_grid_not_the_bse_grid():
    """RED TWIN (fixture-free): the ongrid exchange lookup in ``exciton_bands``
    must use the EXCHANGE-TILE grid.  Pre-fix it used ``kgrid_bse`` — the
    bundle's own grid — which on a densified bundle is the FINE one, so the
    lookup both indexed a nulled array and asked for Q the tiles do not have.

    Read as TEXT rather than imported: ``bse.exciton_bands`` pulls in the FFI
    gate at import, which is unavailable on a login node or a laptop, and this
    cell has no business needing a built ``.so`` to read a subscript."""
    import ast
    import os as _os
    from bse import bse_io

    src_path = _os.path.join(_os.path.dirname(_os.path.abspath(bse_io.__file__)),
                             "exciton_bands.py")
    src = open(src_path).read()
    mod = ast.parse(src)
    fns = [n for n in ast.walk(mod)
           if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert fns, "exciton_bands has no main() to gate"
    fn = fns[0]

    # the tile grid is resolved from the bundle, with the densifier's coarse
    # stamp winning when there is one
    assert "V_q_coarse_grid" in src, (
        "exciton_bands never reads the densifier's coarse-tile grid stamp, so "
        "--vq-mode ongrid on a bse_k_grid-densified bundle meets a None "
        "subscript instead of an explanation")

    subs = [n for n in ast.walk(fn)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == "V_ongrid"]
    assert subs, "the ongrid exchange lookup no longer reads V_ongrid"

    # and kgrid_bse must not be what rounds Q onto the tile lattice
    bad = [n for n in ast.walk(fn)
           if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr == "round"
           and n.args and isinstance(n.args[0], ast.BinOp)
           and isinstance(n.args[0].right, ast.Name)
           and n.args[0].right.id == "kgrid_bse"]
    assert not bad, (
        "a Q is still being rounded onto ``kgrid_bse`` for the exchange-tile "
        "lookup; on a densified bundle that is the FINE grid and the tiles are "
        "COARSE")
