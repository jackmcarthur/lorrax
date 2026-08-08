"""The restart CONSUMERS against the q_irr format: unfold, refuse, or nothing.

WHAT THIS MEASURES AND WHY IT IS SEPARATE FROM THE FORMAT SUITE.
``services/symmetry_maps/tests/test_symmetry_maps_qirr_store.py`` proves the
FORMAT round-trips.  This file proves the TREE'S READERS do, through the
functions ``bse_io`` and ``vq_interp`` actually call — a format that round
trips and a consumer that never asks it to are indistinguishable from the
format's own test suite.

THREE CONSUMER CLASSES, THREE DIFFERENT RIGHT ANSWERS, and separating them
is the whole design:

* ALL-AT-ONCE HOST READERS (``_load_ring_subset``, ``vq_interp``'s lazy
  handles, and the serial-h5py tile shim ``_resolve_munu_reader``) UNFOLD.
  That is the survey's §5 ruling — unfold once on load; per-q reconstruction
  is unavailable, and per the owner memory was never the goal.
* THE SHARDED SlabIO TRANSPORT (``_MunuSlabPlan``) REFUSES, and this is not
  a gap left open by accident: the unfold is a double-gather across the μ
  and ν axes that plan shards on, so a rank holding one (μ, ν) block does
  not hold the elements its own block's images come from.  The refusal
  already existed as a q-extent disagreement; what is new is that it NAMES
  the q wedge, because "the q extent is wrong" sends an operator looking for
  a truncated file.
* EVERY FILE ON DISK TODAY carries no ``qirr_*`` attrs and goes through the
  identical expression it went through before.  Not a tolerance — the same
  ``dset[()]``.

THE TRS ARM CARRIES THE CLAIM.  gnppm's q-stars begin on time-reversal rows,
so the antiunitary branch is live, and the comparison is ELEMENT-WISE ON THE
OFF-DIAGONALS against ``unfold_isdf_operator`` on the same wedge — the array
the uncompressed path itself held.  Both conjugation bugs this campaign has
shipped were diagonal-preserving and off-diagonal-destroying (183.61 eV with
the diagonal exactly zero), so no assertion here reduces to a norm.

THE Si PRODUCTION DECK IS THE UNTOUCHED CONTROL.  Its 960-centroid set is
NOT orbit-closed (47 of 48 ops violating), so it never reaches this format
at all; the cell that matters for it is that a file with no attrs comes back
byte-identical, asserted with ``array_equal`` on the raw bytes.
"""

from __future__ import annotations

import ast
import json
import pathlib

import h5py
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SVC_TESTS = _ROOT / "services" / "symmetry_maps" / "tests"
_STAR_TABLES = _SVC_TESTS / "data" / "star_tables_e9340d1.json"
_BSE_IO = _ROOT / "src" / "bse" / "bse_io.py"
_VQ_INTERP = _ROOT / "src" / "bse" / "vq_interp.py"

_FFT = np.array([12, 12, 12], dtype=np.int64)
_SEEDS_12 = ((0, 0, 0), (4, 5, 0), (2, 2, 6), (9, 7, 6),
             (1, 2, 3), (7, 3, 4), (1, 1, 1), (5, 8, 9))
_Q_IRR = np.array([[0.0, 0.0, 0.0],
                   [1 / 3, 0.0, 1 / 4],
                   [0.0, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 6],
                   [1 / 6, 2 / 3, 5 / 12]])


# ---------------------------------------------------------------------------
# Geometry — the deck's real star structure, a synthetic closed centroid set
# ---------------------------------------------------------------------------

def _service_on_path():
    from ffi import _services
    _services.ensure_on_path()


def _star_tables(deck):
    with open(_STAR_TABLES) as fh:
        d = json.load(fh)[deck]
    labels = d["star_first_occurrence_labels"]
    pos = {int(v): i for i, v in enumerate(labels)}
    irr = np.array([pos[int(v)] for v in d["irr_idx_k"]], dtype=np.int32)
    sym = np.array(d["sym_idx_k"], dtype=np.int32)
    return irr, sym, int(d["n_sym_spatial"]), len(labels)


#: Deck fixtures are READ-ONLY and are opened ``'r'`` here, always.  The
#: service suite reaches them through its own ``_deck_stub``; that module
#: is not on the main suite's path, and the only thing needed here is the
#: ``mf_header/symmetry`` group, so the path is spelled directly.
_REG = _ROOT / "tests" / "regression"


def _deck_syms(deck):
    with h5py.File(_REG / deck / "WFN.h5", "r") as f:
        g = f["mf_header"]["symmetry"]
        n = int(g["ntran"][()])
        return g["mtrx"][:n], g["tnp"][:n]


def _closed_centroid_set(sym_matrices, tnp, fft_grid, seeds):
    """A union of orbits: closed BY CONSTRUCTION, not by luck or by a file."""
    S = np.asarray(sym_matrices, dtype=np.float64)
    rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    tint = np.rint(np.asarray(tnp, dtype=np.float64) / (2.0 * np.pi)
                   * fft_grid).astype(np.int64)
    imgs = set()
    for r in np.asarray(seeds, dtype=np.int64):
        for s in range(S.shape[0]):
            imgs.add(tuple(((rinv[s] @ r + tint[s]) % fft_grid).tolist()))
    return np.array(sorted(imgs), dtype=np.int32)


def _hermitian_ibz(n_q, n_mu, seed=7):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n_q, n_mu, n_mu))
         + 1j * rng.standard_normal((n_q, n_mu, n_mu)))
    return 0.5 * (a + np.swapaxes(a.conj(), -1, -2))


class _Arm:
    """One deck's real stars + a closed set + a wedge + its own reference."""

    def __init__(self, deck, seed=7):
        _service_on_path()
        from symmetry_maps import (centroid_source_map_and_wrap,
                                   verify_centroid_orbit_closure)
        from symmetry_maps import qirr_store as QS

        S, tnp = _deck_syms(deck)
        irr, sym, n_spatial, n_ibz = _star_tables(deck)
        cent = _closed_centroid_set(S, tnp, _FFT, _SEEDS_12)
        self.deck = deck
        self.verdict = verify_centroid_orbit_closure(
            cent.astype(np.float64) / _FFT, S, tnp=tnp, fft_grid=_FFT)
        assert self.verdict.closed, "the arm's own set must be closed"
        perm, L = centroid_source_map_and_wrap(
            cent, S, tnp, _FFT, validate=True, extend_trs=True)
        self.n_mu = int(cent.shape[0])
        self.tables = QS.QirrTables(irr, sym, _Q_IRR[:n_ibz], perm, L,
                                    n_spatial)
        self.X_ibz = _hermitian_ibz(n_ibz, self.n_mu, seed=seed)
        self.n_trs_rows = int((np.asarray(sym) >= n_spatial).sum())
        self.n_q_full = int(len(irr))

    def kernel(self):
        """``unfold_isdf_operator`` on the wedge — the uncompressed array."""
        import jax.numpy as jnp
        from common.collectives import single_device_mesh
        from symmetry_maps import unfold_isdf_operator
        t = self.tables
        return np.asarray(unfold_isdf_operator(
            jnp.asarray(self.X_ibz), irr_idx=t.irr_idx_q,
            sym_idx=t.sym_idx_q, sym_perm=t.sym_perm, L_table=t.L_table,
            q_irr_frac=t.q_irr_frac, mesh_xy=single_device_mesh(),
            n_sym_spatial=int(t.n_sym_spatial)))

    def write_wedge(self, path, name="W0_qmunu"):
        from symmetry_maps import qirr_store as QS
        QS.write_qirr_tensor(path, name, self.X_ibz, tables=self.tables,
                             closure_verdict=self.verdict)
        return path

    def write_legacy_full(self, path, name="W0_qmunu"):
        """The SAME numbers, written the way every file on disk is written."""
        with h5py.File(path, "a") as f:
            if name in f:
                del f[name]
            f.create_dataset(name, data=self.kernel())
        return path


def _assert_offdiag_elementwise(got, ref, label):
    """Every μ != ν element, compared individually.  Never a norm.

    The failure shape this exists for is diagonal-preserving: 183.61 eV of
    error in the kin_ion unfold with the diagonal exactly zero, and a
    Frobenius norm or a trace could not see it.  The count of compared
    elements is returned so a cell can assert it compared something.
    """
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{label}: {got.shape} vs {ref.shape}"
    n_mu = got.shape[-1]
    off = ~np.eye(n_mu, dtype=bool)
    mask = np.broadcast_to(off, got.shape)
    n_off = int(mask.sum())
    assert n_off > 0, f"{label}: no off-diagonal elements to compare"
    bad = np.nonzero(mask & (got != ref))
    assert bad[0].size == 0, (
        f"{label}: {bad[0].size} of {n_off} off-diagonal elements differ; "
        f"worst |delta| {np.abs(got[bad] - ref[bad]).max():.3e}")
    return n_off


@pytest.fixture()
def trs_arm():
    return _Arm("gnppm_debug")


# ---------------------------------------------------------------------------
# 1. The TRS round trip, through the REAL consumer seam
# ---------------------------------------------------------------------------

def test_the_consumer_seam_round_trips_a_trs_deck_bit_identically(
        trs_arm, tmp_path):
    """THE CLAIM: ``bse_io.restart_munu_full_bz`` returns what the run held.

    Not "agrees to a tolerance" — the same function of the same inputs.  The
    format stores the PRE-UNFOLD block, so the reader's unfold and the
    producing run's unfold are one call with one argument list, and BIT
    equality is the only honest bar.  Compared element-wise on the
    off-diagonals, on a deck whose stars are TRS-active so the antiunitary
    branch is live.
    """
    from bse.bse_io import restart_munu_full_bz

    assert trs_arm.n_trs_rows >= 4, "the arm lost its TRS rows"
    path = str(tmp_path / "restart.h5")
    trs_arm.write_wedge(path)
    with h5py.File(path, "r") as f:
        got = restart_munu_full_bz(f["W0_qmunu"], "W0_qmunu", path)
    kern = trs_arm.kernel()
    assert got.shape == (trs_arm.n_q_full, trs_arm.n_mu, trs_arm.n_mu)
    n_off = _assert_offdiag_elementwise(got, kern, "gnppm consumer seam")
    assert np.array_equal(got, kern), "the round trip must be an identity"
    assert n_off > 100


def test_the_seam_is_a_no_op_on_a_legacy_file_byte_for_byte(trs_arm,
                                                            tmp_path):
    """EVERY RESTART FILE THAT EXISTS TODAY, unchanged.

    A no-attr dataset is what the Si production deck writes and what every
    archived run wrote.  ``array_equal`` against the raw ``dset[()]``, not a
    tolerance: the seam must be the same expression on this path.
    """
    from bse.bse_io import is_q_wedge, restart_munu_full_bz

    path = str(tmp_path / "legacy.h5")
    trs_arm.write_legacy_full(path)
    with h5py.File(path, "r") as f:
        assert is_q_wedge(f["W0_qmunu"]) is False
        raw = f["W0_qmunu"][()]
        got = restart_munu_full_bz(f["W0_qmunu"], "W0_qmunu", path)
    assert np.array_equal(got, raw)


def test_a_partially_stamped_file_refuses_at_the_probe(trs_arm, tmp_path):
    """RED TWIN: half a stamp is not "no attrs".

    The missing half is exactly the half that says whether the shape means
    what it looks like, so the cheap probe refuses rather than falling
    through to the legacy path — which would read a wedge as a full-BZ
    tensor of the wrong length and hand it to a consumer whose q indexing
    is silently out by a factor of the star size.
    """
    from bse.bse_io import is_q_wedge

    path = str(tmp_path / "partial.h5")
    trs_arm.write_wedge(path)
    with h5py.File(path, "a") as f:
        del f["W0_qmunu"].attrs["qirr_format_version"]
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError, match=r"PARTIAL stamp"):
            is_q_wedge(f["W0_qmunu"])


# ---------------------------------------------------------------------------
# 2. The serial tile shim both sharded readers fall back to
# ---------------------------------------------------------------------------

def test_the_tile_shim_serves_full_bz_tiles_from_a_wedge(trs_arm, tmp_path):
    """``_resolve_munu_reader`` on a wedge == the same reader on the full file.

    This is the reader ``load_bse_data_from_restart_sharded`` uses whenever
    SlabIO is unavailable, and it is where the (μ, ν, nkx, nky, nkz)
    transpose lives.  Both routes are driven here and every returned tile is
    compared element-wise: a shim that unfolded correctly and then handed
    back the axes in the other order would be exactly as wrong as one that
    did not unfold.
    """
    from bse.bse_io import _resolve_munu_reader

    kgrid = (trs_arm.n_q_full, 1, 1)
    wedge = str(tmp_path / "wedge.h5")
    full = str(tmp_path / "full.h5")
    trs_arm.write_wedge(wedge)
    trs_arm.write_legacy_full(full)
    with h5py.File(wedge, "r") as fw, h5py.File(full, "r") as ff:
        aw = _resolve_munu_reader(fw["W0_qmunu"], kgrid=kgrid)
        af = _resolve_munu_reader(ff["W0_qmunu"], kgrid=kgrid)
        assert aw[:5] == af[:5], "the resolved layout facts must agree"
        n_mu = aw[0]
        slab_w = aw[5](0, n_mu, 0, n_mu)
        slab_f = af[5](0, n_mu, 0, n_mu)
        _assert_offdiag_elementwise(
            slab_w.transpose(2, 3, 4, 0, 1).reshape(-1, n_mu, n_mu),
            slab_f.transpose(2, 3, 4, 0, 1).reshape(-1, n_mu, n_mu),
            "read_slab wedge vs full")
        assert np.array_equal(slab_w, slab_f)
        for q in range(trs_arm.n_q_full):
            assert np.array_equal(aw[6](q, 0, n_mu, 0, n_mu),
                                  af[6](q, 0, n_mu, 0, n_mu)), (
                f"read_q_slab disagrees at q={q}")


def test_the_tile_shim_still_reads_a_legacy_file_from_disk_lazily(
        trs_arm, tmp_path):
    """RED TWIN of the unfold branch: it must not fire on a legacy file.

    The shim's closures normally hold the h5py DATASET and slice it, which
    is what keeps the per-rank readers from materialising the whole tensor.
    A wedge is materialised because it must be; a full-BZ file must not be.
    Measured by asking whether the closure's captured object is still an
    h5py dataset.
    """
    from bse.bse_io import _resolve_munu_reader

    path = str(tmp_path / "legacy.h5")
    trs_arm.write_legacy_full(path)
    with h5py.File(path, "r") as f:
        arm = _resolve_munu_reader(f["W0_qmunu"], kgrid=(trs_arm.n_q_full,
                                                         1, 1))
        held = [c.cell_contents for c in (arm[6].__closure__ or ())]
        assert any(isinstance(o, h5py.Dataset) for o in held), (
            "the legacy path stopped slicing the dataset; the per-rank "
            "readers would now materialise the whole tensor")


# ---------------------------------------------------------------------------
# 3. The sharded transport refuses, and says why
# ---------------------------------------------------------------------------

def test_the_slabio_plan_refuses_a_wedge_and_names_it(trs_arm, tmp_path):
    """The refusal an operator has to be able to act on.

    ``_MunuSlabPlan`` describes a per-rank (μ, ν) hyperslab; the unfold
    gathers ACROSS μ and ν, so there is no offset it could ask for that
    would reconstruct a full-BZ q.  It refuses — and the message must name
    the q wedge and the deck key that avoids it, because the bare extent
    disagreement reads as a truncated file.
    """
    from bse.bse_io import _MunuSlabPlan

    with pytest.raises(ValueError) as exc:
        _MunuSlabPlan((5, trs_arm.n_mu, trs_arm.n_mu),
                      (trs_arm.n_q_full, 1, 1))
    msg = str(exc.value)
    assert "wedge" in msg.lower()
    assert "restart_q_storage=full" in msg


def test_the_slabio_plan_accepts_a_full_bz_dataset(trs_arm):
    """RED TWIN: the refusal above must not be firing on everything.

    Same class, same kgrid, the full-BZ q extent — accepted, with the μ/ν
    extents read off the shape.  Without this the cell above would pass on
    a plan that refused unconditionally.
    """
    from bse.bse_io import _MunuSlabPlan

    plan = _MunuSlabPlan((trs_arm.n_q_full, trs_arm.n_mu, trs_arm.n_mu),
                         (trs_arm.n_q_full, 1, 1))
    assert plan.n_rmu == trs_arm.n_mu and plan.nq == trs_arm.n_q_full


def test_an_oversized_q_extent_does_not_claim_to_be_a_wedge(trs_arm):
    """The wedge advice is attached to the arm it applies to.

    A dataset with MORE q rows than the k-grid is not a wedge and cannot be
    fixed by ``restart_q_storage=full``; telling an operator otherwise sends
    them to re-run a GW leg that was never the problem.
    """
    from bse.bse_io import _MunuSlabPlan

    with pytest.raises(ValueError) as exc:
        _MunuSlabPlan((trs_arm.n_q_full + 3, trs_arm.n_mu, trs_arm.n_mu),
                      (trs_arm.n_q_full, 1, 1))
    assert "restart_q_storage" not in str(exc.value)


# ---------------------------------------------------------------------------
# 4. The two-channel divergence vector (phase-1b carry-forward 2)
# ---------------------------------------------------------------------------

def test_two_channels_in_one_file_resolve_independently(tmp_path):
    """ONE FILE, ONE CHANNEL ON THE WEDGE AND ONE ON THE FULL BZ.

    ``bispinor_debug``'s charge set (256) is NOT orbit-closed while its
    transverse set (209_current) IS, so a bispinor run genuinely reaches a
    state where one channel resolves ``full`` and the other ``ibz``.  The
    format is per-DATASET — the tables live in a sibling group of each
    tensor, not once per file — and this is the cell that says the two
    datasets do not contaminate each other: both come back through the same
    consumer seam, each by its own route, bit-identically.

    (The two channels are stood up here from the two TRS decks' real star
    structures rather than from the bispinor deck's centroid files, because
    what is being measured is the READER's per-dataset independence, not
    those files' closure — which is measured where it belongs, in the
    service's closure suite.)
    """
    from bse.bse_io import is_q_wedge, restart_munu_full_bz

    charge = _Arm("gnppm_debug", seed=5)
    transverse = _Arm("bispinor_debug", seed=11)
    path = str(tmp_path / "twochannel.h5")
    # The transverse channel is the closed one and goes on the wedge; the
    # charge channel is the open one and is written the legacy way.
    transverse.write_wedge(path, name="W0_qmunu_transverse")
    charge.write_legacy_full(path, name="W0_qmunu")
    with h5py.File(path, "r") as f:
        assert is_q_wedge(f["W0_qmunu_transverse"]) is True
        assert is_q_wedge(f["W0_qmunu"]) is False
        got_t = restart_munu_full_bz(f["W0_qmunu_transverse"],
                                     "W0_qmunu_transverse", path)
        got_c = restart_munu_full_bz(f["W0_qmunu"], "W0_qmunu", path)
        raw_c = f["W0_qmunu"][()]
    _assert_offdiag_elementwise(got_t, transverse.kernel(), "transverse/ibz")
    assert np.array_equal(got_t, transverse.kernel())
    assert np.array_equal(got_c, raw_c), (
        "the full-BZ channel must be untouched by the wedge channel in the "
        "same file")


def test_the_placeholder_refusal_survives_the_consumer_seam(trs_arm,
                                                            tmp_path):
    """RED TWIN: a q_irr PLACEHOLDER must not read as data through a consumer.

    ``gw_init`` allocates a zero W0 before the screening that fills it
    exists.  In this format that file is eight times smaller and just as
    plausible — same shape, same tables, all zeros — which is the April
    all-zero-screening incident with a smaller footprint.  The seam inherits
    ``read_tensor``'s ``require_persisted`` refusal rather than routing
    around it.
    """
    from bse.bse_io import restart_munu_full_bz
    from symmetry_maps import qirr_store as QS

    path = str(tmp_path / "placeholder.h5")
    QS.allocate_qirr_placeholder(
        path, "W0_qmunu",
        (trs_arm.tables.n_q_ibz, trs_arm.n_mu, trs_arm.n_mu),
        tables=trs_arm.tables, closure_verdict=trs_arm.verdict)
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError, match=r"PLACEHOLDER"):
            restart_munu_full_bz(f["W0_qmunu"], "W0_qmunu", path)


# ---------------------------------------------------------------------------
# 5. Source ratchets: the seam is the only door
# ---------------------------------------------------------------------------

def _subscripted_datasets(src, names):
    """Every ``X[...]`` whose value is one of ``names`` — the bypass shape."""
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Subscript):
            continue
        v = node.value
        if isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Constant):
            if v.slice.value in names:
                found.append(ast.unparse(node))
    return found


def test_the_full_file_reader_goes_through_the_seam():
    """``_load_ring_subset`` must not subscript V_qmunu/W0_qmunu directly.

    The bypass is one expression wide — ``f["V_qmunu"][:]`` — and it would
    hand the BSE a wedge under a full-BZ q index, which every downstream
    shape check accepts because μ and ν are right and only the q axis is
    short.  That is precisely the ``build_hdir`` hazard dbe3b4ec closed for
    W0, arriving through the other door.
    """
    src = _BSE_IO.read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_load_ring_subset")
    leaks = _subscripted_datasets(ast.unparse(fn),
                                  {"V_qmunu", "W0_qmunu"})
    assert not leaks, (
        f"_load_ring_subset subscripts a restart tensor directly: {leaks}; "
        f"route it through restart_munu_full_bz")
    assert "restart_munu_full_bz" in ast.unparse(fn)


def test_vq_interp_binds_both_tensors_through_the_seam():
    """The lazy-handle reader asks the same question for V and for W0.

    A seam applied to one of two tensors is a seam the other silently
    routes around, and V is the one with no placeholder path and therefore
    the one nobody would think to check.
    """
    tree = ast.parse(_VQ_INTERP.read_text())
    seam = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "restart_munu_full_bz"]
    probe = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "is_q_wedge"]
    assert len(seam) == 2, (
        f"vq_interp must route BOTH V_qmunu and W0_qmunu through the seam; "
        f"found {len(seam)} call(s)")
    assert len(probe) == 2, f"found {len(probe)} probe call(s)"
    # ...and each seam call must name the tensor it is reading, so a future
    # edit cannot point both at V.
    named = {ast.unparse(c.args[1]) for c in seam}
    assert named == {"'V_qmunu'", "'W0_qmunu'"}, named


def test_the_w0_ready_guard_shape_is_untouched():
    """The q-storage branch must not have become a nested ``if``.

    ``test_bse_w0_ready_gate``'s ratchet requires every ``if`` binding
    ``zx["W0"]`` to test the persisted flag; adding a q-storage ``if``
    inside that body would either break it or teach it an exception it
    would then carry forever.  The branch is a conditional EXPRESSION, and
    this cell is what keeps it one.
    """
    tree = ast.parse(_VQ_INTERP.read_text())
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = "".join(ast.unparse(s) for s in node.body)
        if "zx['W0']" in body or 'zx["W0"]' in body:
            guards.append("W0_ready" in ast.unparse(node.test))
    assert guards and all(guards), (
        f"vq_interp binds zx['W0'] behind {len(guards)} guard(s) and not all "
        f"of them test W0_ready")


def test_the_probe_has_one_implementation():
    """``is_q_wedge`` is the only place the tree asks the q-storage question.

    Two implementations of "is this a q_irr file" is how a reader ends up
    disagreeing with the format about what it is holding — and the
    disagreement would be silent, because both answers are the string
    ``"full"`` most of the time.
    """
    calls = []
    for path in (_BSE_IO, _VQ_INTERP):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "dataset_q_storage"):
                calls.append(path.name)
    assert calls == [_BSE_IO.name], (
        f"dataset_q_storage must be called exactly once in the whole BSE "
        f"reader surface (inside is_q_wedge); found {calls}")
    fn = next(n for n in ast.walk(ast.parse(_BSE_IO.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "is_q_wedge")
    assert "dataset_q_storage" in ast.unparse(fn), (
        "the one call moved out of is_q_wedge")
