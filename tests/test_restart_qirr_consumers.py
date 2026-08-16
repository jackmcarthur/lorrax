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
_BSE_IO = _ROOT / "src" / "bse" / "bse_loading.py"
_VQ_INTERP = _ROOT / "src" / "bse" / "vq_interp.py"
_TAGGED_ARRAYS = _ROOT / "src" / "file_io" / "tagged_arrays.py"

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


def _q_irr_for(n_ibz):
    """``n_ibz`` IBZ q's in fractional coords, generic enough to phase.

    ``_Q_IRR`` is gnppm's real five; a deck with more stars (Si has eight)
    needs more, and what the unfold cares about is only that the q's are
    generic — the umklapp phase is ``exp(2πi q·(L_μ − L_ν))``, so a q whose
    dot with every lattice wrap is an integer makes the phase the identity
    and hides a dropped ``L_table``.  Thirds and twelfths are used for the
    same reason the deck's own q's are: they are the fractions a real
    Monkhorst-Pack grid produces, and they are not integers against L.
    """
    if n_ibz <= len(_Q_IRR):
        return _Q_IRR[:n_ibz]
    extra = np.array([[(3 + 2 * i) / 12.0, (1 + 5 * i) % 12 / 12.0,
                       (7 + i) % 12 / 12.0]
                      for i in range(n_ibz - len(_Q_IRR))])
    return np.concatenate([_Q_IRR, extra], axis=0)


class _Arm:
    """One deck's real stars + a closed set + a wedge + its own reference."""

    def __init__(self, deck, seed=7, seeds=_SEEDS_12):
        _service_on_path()
        from symmetry_maps import (centroid_source_map_and_wrap,
                                   verify_centroid_orbit_closure)
        from symmetry_maps import qirr_store as QS

        S, tnp = _deck_syms(deck)
        irr, sym, n_spatial, n_ibz = _star_tables(deck)
        cent = _closed_centroid_set(S, tnp, _FFT, seeds)
        self.deck = deck
        self.verdict = verify_centroid_orbit_closure(
            cent.astype(np.float64) / _FFT, S, tnp=tnp, fft_grid=_FFT)
        assert self.verdict.closed, "the arm's own set must be closed"
        perm, L = centroid_source_map_and_wrap(
            cent, S, tnp, _FFT, validate=True, extend_trs=True)
        self.n_mu = int(cent.shape[0])
        self.tables = QS.QirrTables(irr, sym, _q_irr_for(n_ibz), perm, L,
                                    n_spatial)
        self.X_ibz = _hermitian_ibz(n_ibz, self.n_mu, seed=seed)
        self.n_trs_rows = int((np.asarray(sym) >= n_spatial).sum())
        self.n_q_full = int(len(irr))

    def exercises(self):
        """What the unfold's machinery ACTUALLY does on this arm.

        Returns ``(n_nonidentity_perm_q, n_phased_elements, n_trs_q)`` — the
        three branches of ``unfold_isdf_operator`` that can silently be no-ops
        depending on which sym rows this deck's q-stars happen to select.
        A gate that does not check this is asserting an identity permutation
        against itself and calling it a round trip; see
        ``test_the_arms_between_them_exercise_every_unfold_branch``.
        """
        t = self.tables
        perm = np.asarray(t.sym_perm)
        L = np.asarray(t.L_table)
        irr = np.asarray(t.irr_idx_q)
        sym = np.asarray(t.sym_idx_q)
        q_irr = np.asarray(t.q_irr_frac)
        ident = np.arange(self.n_mu)
        n_perm = sum(1 for q in range(len(irr))
                     if not np.array_equal(perm[sym[q]], ident))
        n_phase = 0
        for q in range(len(irr)):
            qL = q_irr[irr[q]] @ L[sym[q]].T
            d = np.abs(qL[:, None] - qL[None, :])
            n_phase += int((np.abs(d - np.rint(d)) > 1e-12).sum())
        return n_perm, n_phase, int((sym >= t.n_sym_spatial).sum())

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


@pytest.fixture()
def perm_arm():
    """Si: 48 spatial ops, 64 q, and stars that reach NON-IDENTITY rows.

    gnppm cannot carry the permutation half of the claim, and it took a
    deliberate measurement to notice.  Its group has two spatial ops, and
    every one of its nine q's folds to its IBZ parent through sym row 0 or
    row 2 — the identity and time-reversal-times-the-identity — so its
    centroid permutation is the identity at every q and its umklapp wraps are
    zero at every q.  The TRS conjugation IS live there and that is what that
    arm is for; the double gather and the phase are not, and a round trip
    that never permutes is an identity asserted against itself.

    Si's 64-point grid reaches sym rows 0-23, whose permutations are genuinely
    non-trivial.  Four seeds rather than eight keep the orbit union (and so
    the (64, μ, μ) reference) small enough to stay a unit test.
    """
    return _Arm("si_cohsex_debug", seeds=_SEEDS_12[:4])


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
# 3. The sharded transport READS a wedge (since 2026-08-15), and still
#    refuses a file it genuinely cannot reconstruct
# ---------------------------------------------------------------------------

def test_the_slabio_plan_READS_a_wedge_when_the_file_carries_its_tables(
        trs_arm):
    """MIGRATED.  This cell used to assert the opposite.

    It pinned a refusal whose stated reason was COST — "the price of an
    all_to_all of an N_mu²-class object … has never been measured on a real
    interconnect" — and whose message claimed the transport *cannot* unfold,
    which ``file_io.tagged_arrays._unfold_wedge`` had already recorded as
    false.  MEASURED 2026-08-15 on 4×A100 NVLink at complex128: the unfold
    runs at 57.4 GiB/s against a 2.919 GiB/s disk path, and μ² cancels
    between the two sides, so the wedge wins by 6–17× at every size.  The
    refusal is lifted; the plan now takes the file's own tables and reads it.
    """
    from bse.bse_io import _MunuSlabPlan

    plan = _MunuSlabPlan((trs_arm.tables.n_q_ibz, trs_arm.n_mu, trs_arm.n_mu),
                         (trs_arm.n_q_full, 1, 1),
                         wedge_tables=trs_arm.tables)
    assert plan.is_wedge is True
    assert plan.n_rmu == trs_arm.n_mu and plan.nq == trs_arm.n_q_full
    # The read is sized to the WEDGE; the unfold to the full BZ happens after
    # it, in jax, which is the whole design.
    _off, shape, _spec = plan.request(trs_arm.n_mu)
    assert shape[0] == trs_arm.tables.n_q_ibz


def test_a_wedge_with_NO_tables_is_still_refused_and_named(trs_arm):
    """The refusal that survives, and it is now about reconstructibility.

    A q extent below the k-grid with no unfold tables is not a wedge this
    reader can take — it is a truncated or mis-stamped file, and re-deriving
    the tables from this run's ``sym`` is not offered, because a table that
    reconstructs the tensor must be the table that deconstructed it.
    """
    from bse.bse_io import _MunuSlabPlan

    with pytest.raises(ValueError) as exc:
        _MunuSlabPlan((5, trs_arm.n_mu, trs_arm.n_mu),
                      (trs_arm.n_q_full, 1, 1))
    msg = str(exc.value)
    assert "no q_irr unfold tables" in msg
    assert "truncated or mis-stamped" in msg
    # ...and this arm DOES name the key, because here it applies.
    assert "restart_q_storage=auto|ibz" in msg


def test_gamma_is_one_hyperslab_on_a_wedge_not_an_unfold(trs_arm):
    """The single-q ``V_q0`` route, which needs no collective at all.

    Γ is its own orbit parent under every point group, so on a wedge the
    single-q read is still ONE hyperslab — the same bytes at a different
    row.  That is asserted against the FILE'S OWN tables (identity
    permutation, zero wrap, not time-reversed), never assumed, so a wedge
    that reached Γ by a rotation would refuse rather than hand back a
    rotated block labelled q=0.
    """
    from bse.bse_io import _MunuSlabPlan

    plan = _MunuSlabPlan((trs_arm.tables.n_q_ibz, trs_arm.n_mu, trs_arm.n_mu),
                         (trs_arm.n_q_full, 1, 1),
                         wedge_tables=trs_arm.tables)
    row = plan.gamma_wedge_row()
    assert row is not None, (
        "Γ should be reachable by the identity on any orbit decomposition")
    off, shape, _spec = plan.request(trs_arm.n_mu, q_index=0)
    assert shape[0] == 1 and off[0] == row

    # Any OTHER single q on a wedge is a rotated image and must refuse.
    with pytest.raises(ValueError, match="only q=0|Only q=0"):
        plan.request(trs_arm.n_mu, q_index=1)


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


def test_the_probe_has_exactly_two_named_callers():
    """The q-storage question is asked in TWO places, and both are named.

    There is one PROBE — ``symmetry_maps.dataset_q_storage`` — and there are
    two readers that must ask it, in two layers that cannot import each
    other: the BSE reader (inside ``bse_io.is_q_wedge``) and the GW restart
    reader (inside ``file_io.tagged_arrays._qirr_wedge_tables``).  ``file_io``
    calls the service door directly because importing ``bse`` from
    ``file_io`` is uphill and the layering ratchet forbids it.

    THIS CELL USED TO REQUIRE EXACTLY ONE CALLER, and it was right to until
    2026-08-08: two implementations of "is this a q_irr file" is how a reader
    ends up disagreeing with the format about what it is holding, silently,
    because both answers are ``"full"`` most of the time.  What the landing
    census showed is that the danger is not a second CALLER — it is a second
    ANSWER.  The GW reader asking nothing at all was the worse failure: it
    took a wedge, said nothing, and died 200 lines later in jax's ufunc
    machinery on a shape mismatch that named neither the file nor the deck
    key.  So the rule becomes an allowlist rather than a count: these two
    call it, a third must justify itself here first.
    """
    _EXPECTED = {_BSE_IO.name: "is_q_wedge",
                 _TAGGED_ARRAYS.name: "_qirr_wedge_tables"}
    calls = {}
    for path in (_BSE_IO, _VQ_INTERP, _TAGGED_ARRAYS):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "dataset_q_storage"):
                calls.setdefault(path.name, 0)
                calls[path.name] += 1
    assert calls == {k: 1 for k in _EXPECTED}, (
        f"dataset_q_storage must be called exactly once in each of "
        f"{sorted(_EXPECTED)} and nowhere else on this surface; found "
        f"{calls}")
    for path, owner in ((_BSE_IO, _EXPECTED[_BSE_IO.name]),
                        (_TAGGED_ARRAYS, _EXPECTED[_TAGGED_ARRAYS.name])):
        fn = next(n for n in ast.walk(ast.parse(path.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == owner)
        assert "dataset_q_storage" in ast.unparse(fn), (
            f"the one call in {path.name} moved out of {owner}")


# ---------------------------------------------------------------------------
# The GW restart reader: it ALWAYS UNFOLDS (owner ruling 2026-08-08 ~13:20)
# ---------------------------------------------------------------------------

def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _write_gw_restart(path, arm, V_full_bz, *, wedge):
    """A complete GW restart file carrying ``arm``'s numbers.

    ``wedge=False`` writes ``V_full_bz`` the way every full-BZ run writes it.
    ``wedge=True`` writes the SAME RUN's pre-unfold block plus the tables that
    reconstruct it — the two files a reader must not be able to tell apart
    once it has read them.  ψ / enk are along because the reader requires
    ``psi_full_y`` and pads on its own axis.
    """
    import jax.numpy as jnp
    from file_io import write_restart_state_to_h5
    n_mu = arm.n_mu
    rng = np.random.default_rng(11)
    psi = (rng.standard_normal((2, 3, 1, n_mu))
           + 1j * rng.standard_normal((2, 3, 1, n_mu))).astype(np.complex128)
    enk = rng.standard_normal((2, 3)).astype(np.float64)
    write_restart_state_to_h5(
        path, n_rmu_logical=n_mu, V_qmunu=jnp.asarray(V_full_bz),
        enk_full=jnp.asarray(enk), mesh=_mesh_1x1(), mode="w",
        kgrid=(arm.n_q_full, 1, 1))
    write_restart_state_to_h5(
        path, n_rmu_logical=n_mu, psi_full_y=jnp.asarray(psi),
        mesh=_mesh_1x1(), mode="a")
    if wedge:
        # Replace V with the PRE-UNFOLD block and stamp it, which is what
        # ``tagged_arrays._stamp_qirr`` does on the production writer path
        # once SlabIO has released the file.
        from symmetry_maps import write_qirr_tensor
        with h5py.File(path, "a") as f:
            del f["V_qmunu"]
        write_qirr_tensor(path, "V_qmunu", arm.X_ibz, tables=arm.tables,
                          closure_verdict=arm.verdict)
        with h5py.File(path, "a") as f:
            f["V_qmunu"].attrs["V_ready"] = True
    return path


def _require_slabio():
    """Skip, naming the probe stage, when the phdf5 tile path is absent.

    Same idiom and same reason as ``test_file_io._require_slabio``.  The
    end-to-end cell below goes through ``load_restart_state_from_h5``, which
    is a SlabIO collective; on a machine without the FFI (WSL) that is a
    missing capability, not a result.  The gate that carries the CLAIM does
    not go through SlabIO at all and runs everywhere — see
    ``test_the_gw_reader_unfolds_a_wedge_bit_identically``.
    """
    from file_io.slab_io import probe_availability
    ok, stage, reason = probe_availability()
    if not ok:
        pytest.skip(f"SlabIO unavailable (probe stage '{stage}'): {reason}")


def test_the_arms_between_them_exercise_every_unfold_branch(
        trs_arm, perm_arm):
    """THE COVERAGE RATCHET: no branch of the unfold is a silent no-op.

    ``unfold_isdf_operator`` has three things it can do — permute the centroid
    axes, apply the umklapp phase, and conjugate on a TRS row — and each of
    them is skipped entirely when a deck's q-stars happen not to select a sym
    row that needs it.  MEASURED on the arms this file uses: gnppm's nine q's
    all fold through sym row 0 or 2, both of which carry the IDENTITY
    permutation and zero wraps, so on that arm the unfold permutes nothing
    and phases nothing while still passing a bit-identity round trip.  That
    is a real gap and it was invisible until it was measured: the deck is the
    one the landing census used, and both of this campaign's conjugation bugs
    lived in exactly this machinery.

    So the arms are asserted to cover the branches BETWEEN them, here, once,
    rather than each cell hoping its own fixture is rich enough.
    """
    g_perm, g_phase, g_trs = trs_arm.exercises()
    s_perm, s_phase, s_trs = perm_arm.exercises()

    assert g_trs > 0, "gnppm is the TRS arm; it must reach antiunitary rows"
    assert (g_perm, g_phase) == (0, 0), (
        "gnppm's stars used to fold through identity rows only.  If that has "
        "changed the docstrings above and in the perm_arm fixture are now "
        "wrong and must be re-measured, not merely re-run.")
    assert s_perm > 0, (
        "the Si arm exists to permute; if its stars stopped reaching "
        "non-identity sym rows, the double gather is untested everywhere")
    assert s_phase > 0, (
        "the Si arm must carry nonzero umklapp phase, or a dropped L_table "
        "would pass every cell in this file")


@pytest.mark.parametrize("arm_name", ["trs_arm", "perm_arm"])
@pytest.mark.parametrize("extra_pad", [0, 2])
def test_the_gw_reader_unfolds_a_wedge_bit_identically(
        request, tmp_path, extra_pad, arm_name):
    """THE GATE: unfold(wedge file) == what the producing run held.

    This is what replaced the refusal ``50db6299`` added.  That refusal was
    correct for its day — the GW reader did not unfold, and a wedge reaching
    it died 200 lines later in jax's ufunc machinery at ``W_q - V_q`` on
    ``(9, 399, 399)`` against ``(5, 399, 399)``, a traceback naming a
    subtraction and no part of the actual problem.  The owner's ruling of
    2026-08-08 ~13:20 is that the reader unfolds instead, so the assertion
    moves from "it refuses and says why" to "it returns what the run held".

    NOT A TOLERANCE.  The format stores the PRE-UNFOLD block, so the reader's
    unfold is the SAME FUNCTION ON THE SAME INPUTS the producer called.  The
    comparison is element-wise on the off-diagonals, never a norm, because
    both conjugation bugs this campaign shipped were diagonal-preserving.

    IT DOES NOT GO THROUGH SlabIO, deliberately.  What the change consists of
    is ``_qirr_wedge_tables`` (which tables does this file carry?) and
    ``_unfold_wedge`` (apply them); the transport between them is untouched
    by this branch and is unavailable on WSL.  So this cell hands
    ``_unfold_wedge`` exactly what SlabIO hands it — the wedge at the PADDED
    μ extent, zero-filled past the dataset, sharded ``P(None,'x','y')`` — and
    the end-to-end cell below proves the wiring where the FFI exists.

    THE PAD ARM (``extra_pad=2``) is the only place ``QirrTables.padded``
    runs on the GW path: the file stores the LOGICAL extent and the reader
    re-pads the TABLES to its own device count.  Its pad rows must come back
    exactly zero — a pad that acquired structure would be a permutation
    addressing centroids the file does not have.

    TWO ARMS, because one deck cannot carry the claim: gnppm reaches the
    antiunitary branch and permutes nothing, Si permutes and phases and never
    conjugates.  ``test_the_arms_between_them_exercise_every_unfold_branch``
    is what keeps that division honest.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.tagged_arrays import _qirr_wedge_tables, _unfold_wedge

    trs_arm = request.getfixturevalue(arm_name)
    path = str(tmp_path / "wedge.h5")
    trs_arm.write_wedge(path, name="V_qmunu")
    with h5py.File(path, "r") as f:
        tables = _qirr_wedge_tables(f)
        assert set(tables) == {"V_qmunu"}, (
            f"the probe must find exactly the wedge tensor; got {set(tables)}")
        raw = np.asarray(f["V_qmunu"][()])

    n_mu = trs_arm.n_mu
    n_pad = n_mu + extra_pad
    assert raw.shape == (trs_arm.X_ibz.shape[0], n_mu, n_mu), (
        "the file must store the LOGICAL μ extent on the WEDGE q axis")

    # Exactly what SlabIO returns: the padded shape, zero-filled past the
    # dataset, on the spec ``_munu_slab_request`` builds.
    mesh = _mesh_1x1()
    padded = np.pad(raw, ((0, 0), (0, extra_pad), (0, extra_pad)))
    arr = jax.device_put(jnp.asarray(padded),
                         NamedSharding(mesh, P(None, "x", "y")))

    got = np.asarray(_unfold_wedge(arr, tables["V_qmunu"], n_pad, mesh))
    want = np.pad(trs_arm.kernel(), ((0, 0), (0, extra_pad), (0, extra_pad)))

    assert got.shape == want.shape == (trs_arm.n_q_full, n_pad, n_pad)
    assert trs_arm.n_q_full > raw.shape[0], (
        "this deck must actually reduce, or the gate proves nothing")
    n_off = _assert_offdiag_elementwise(got, want, "GW reader unfold")
    assert n_off > 0
    assert np.array_equal(got, want), (
        "the unfolded wedge must be bit-identical to the array the "
        "producing run held, diagonal included")
    if extra_pad:
        assert not got[:, n_mu:, :].any(), "μ pad rows must stay zero"
        assert not got[:, :, n_mu:].any(), "ν pad cols must stay zero"


@pytest.mark.parametrize("extra_pad", [0, 2])
def test_the_gw_restart_reader_unfolds_a_wedge_end_to_end(
        trs_arm, tmp_path, monkeypatch, extra_pad):
    """The same claim through the REAL reader, wiring included.

    The cell above proves the unfold; this proves it is CONNECTED — that
    ``read_restart_state_from_h5`` probes in pass 1, reads the wedge through
    the transport unchanged, and unfolds before the caller sees it.  Two
    restart files carrying the same run's numbers, one full-BZ and one on the
    wedge, must be indistinguishable once read.

    Skipped where the phdf5 FFI is absent (WSL), because the reader is a
    SlabIO collective; the claim itself does not depend on this cell running.
    """
    from file_io import load_restart_state_from_h5

    _require_slabio()
    if extra_pad:
        monkeypatch.setenv("LORRAX_EXTRA_MU_PAD", str(extra_pad))
    else:
        monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)

    ref_full_bz = trs_arm.kernel()
    full_path = _write_gw_restart(str(tmp_path / "full.h5"), trs_arm,
                                  ref_full_bz, wedge=False)
    wedge_path = _write_gw_restart(str(tmp_path / "wedge.h5"), trs_arm,
                                   ref_full_bz, wedge=True)

    # The wedge really is smaller on disk, or this gate proves nothing.
    with h5py.File(wedge_path, "r") as f:
        assert f["V_qmunu"].shape[0] == trs_arm.X_ibz.shape[0]
    with h5py.File(full_path, "r") as f:
        assert f["V_qmunu"].shape[0] == trs_arm.n_q_full

    mesh = _mesh_1x1()
    got = np.asarray(load_restart_state_from_h5(wedge_path, mesh).V_qmunu)
    want = np.asarray(load_restart_state_from_h5(full_path, mesh).V_qmunu)

    assert got.shape == want.shape == (
        trs_arm.n_q_full, trs_arm.n_mu + extra_pad,
        trs_arm.n_mu + extra_pad)
    n_off = _assert_offdiag_elementwise(got, want, "GW reader unfold")
    assert n_off > 0
    assert np.array_equal(got, want), (
        "the unfolded wedge and the full-BZ file must be bit-identical, "
        "diagonal included")
    if extra_pad:
        assert np.array_equal(got[:, trs_arm.n_mu:, :], np.zeros_like(
            got[:, trs_arm.n_mu:, :])), "μ pad rows must re-read as zeros"
        assert np.array_equal(got[:, :, trs_arm.n_mu:], np.zeros_like(
            got[:, :, trs_arm.n_mu:])), "ν pad cols must re-read as zeros"


def test_the_gw_restart_reader_is_byte_identical_on_every_file_today(
        trs_arm, tmp_path, monkeypatch):
    """The other half: a full-BZ or legacy no-attr file is untouched.

    ``_qirr_wedge_tables`` runs on EVERY restart read, so a false positive
    here would put an unfold in front of every archived run in the tree.  A
    file with no q-storage attrs at all — what every restart written before
    this format existed looks like — must probe as ``full``, return an empty
    table map, and leave the read on the identical byte path.
    """
    from file_io.tagged_arrays import _qirr_wedge_tables, _unfold_wedge

    monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)
    path = str(tmp_path / "legacy.h5")
    trs_arm.write_legacy_full(path, name="V_qmunu")
    with h5py.File(path, "r") as f:
        assert _qirr_wedge_tables(f) == {}, (
            "a legacy no-attr file must yield no unfold tables at all")

    # ...and the no-op arm of the unfold returns the SAME OBJECT, so a
    # full-BZ read cannot acquire a jit, a copy, or a resharding.
    sentinel = object()
    assert _unfold_wedge(sentinel, None, 8, _mesh_1x1()) is sentinel
