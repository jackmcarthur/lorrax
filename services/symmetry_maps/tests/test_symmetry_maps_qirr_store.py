"""The §6 stress campaign for the q_irr store: round trips and refusals.

The owner's words make these gates half the point of the feature rather
than safety trim around it, so they are built to exercise the fold/unfold
machinery hard.  Five arms:

1. **Round trip on the TRS-ACTIVE decks, off-diagonals asserted
   ELEMENT-WISE.**  gnppm and bispinor each carry four time-reversal rows
   in ``star_tables_e9340d1.json`` — four of their five stars begin on a
   TRS row — so the antiunitary branch of the unfold is live.  Si is the
   NEGATIVE CONTROL and nothing else: MEASURED ``ntran=48`` and **zero**
   TRS rows, so silicon structurally cannot reach that branch, and a fix
   verified only on Si proves nothing about it.  That is precisely how the
   ``kin_ion`` unfold bug survived.
2. **The refusal suite**, each refusal beside its constructible twin: a
   non-closed set at write (the real production 960-centroid vector), a
   table-hash mismatch at read, a version mismatch, a RANK that
   contradicts the stamped version, a shape-versus-attr disagreement, a
   partial stamp, and the no-attr legacy path read byte-for-byte.  The
   rank cell is the one whose red half had to be built rather than
   flipped: a rank-4 tensor stamped version 1 passes every OTHER check
   in this file whenever its leading extent equals the wedge extent, so
   the cell asserts those four still pass before asserting the refusal —
   otherwise it would be green against a reader that has no rank guard.
3. **The persisted flag.**  A zero placeholder of exactly the right shape
   must not read as data.
4. **The umklapp arm.**  A synthetic non-symmorphic glide where dropping
   the phase moves the answer by order unity, so "the stored/read path
   keeps the phase" is a claim with weight behind it.
5. **Multi-rank**, in ``test_symmetry_maps_multiproc.py``, which imports
   the check bodies below rather than restating them.

WHY OFF-DIAGONALS, ELEMENT-WISE, AND WHY NOT A NORM.  Two separate
conjugation bugs shipped in this exact area, and **both were
diagonal-preserving and off-diagonal-destroying** — 183.61 eV of error with
the diagonal exactly zero.  Hermiticity, the electron count, the spectrum
and ``eqp.dat`` all stayed green straight through them.  A Frobenius norm
or a trace cannot see that class of failure, so
:func:`assert_offdiag_elementwise` builds the μ≠ν mask and compares every
element under it, and no assertion in this file reduces the residual to a
single scalar before comparing.

WHAT THE TRS ARM DOES AND DOES NOT COVER, MEASURED.  gnppm's and
bispinor's q-stars use symmetry rows ``{0, 2}`` only — the identity and the
TIME-REVERSED identity (``n_sym_spatial = 2``, so row 2 is TRS of row 0).
Their ``L_table`` is therefore zero on every row the q axis touches and the
umklapp phase is identically 1: MEASURED, dropping the phase from the
reference moves the answer by 0.000e+00, while dropping the conjugation
moves it by 1.760 relative (1.869 on the off-diagonals alone).  So the TRS
arm is a strong test of the antiunitary branch and NO test of the phase,
which is exactly why the umklapp arm is separate and synthetic instead of
being folded into it.

NO PRODUCTION CENTROID FILE IS REGENERATED.  ``centroids_frac_960.txt`` is
read as a TEST VECTOR for the refusal — the deck production actually runs
on, whose refusal is the normal path — and ``centroids_frac_144.txt`` is
read as the one orbit-closed set in the tree.  Everything else is
synthesised in memory.  Fixtures open ``'r'``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import _deck_stub
from lxkit.testing import require_devices
from symmetry_maps import (centroid_source_map_and_wrap, unfold_isdf_operator,
                           verify_centroid_orbit_closure)
from symmetry_maps import qirr_store as QS

_TESTS = os.path.dirname(os.path.abspath(__file__))
_STAR_TABLES = os.path.join(_TESTS, "data", "star_tables_e9340d1.json")

#: The synthetic FFT grid the in-memory centroid sets live on.  12 is
#: divisible by Px·Py = 4 and by 2, so the same geometry serves 1x1, 2x2
#: and 4x1 without a second set of seeds.
_FFT = np.array([12, 12, 12], dtype=np.int64)

#: Seeds for the TRS arm's centroid set.  The deck ops are {I, σ_z}, so a
#: seed with z ∈ {0, 6} is its own image (one point) and any other seed
#: contributes two: four of each gives **12** centroids, which divides 4.
_SEEDS_12 = ((0, 0, 0), (4, 5, 0), (2, 2, 6), (9, 7, 6),
             (1, 2, 3), (7, 3, 4), (1, 1, 1), (5, 8, 9))

#: IBZ q's in fractional reciprocal coordinates, with non-zero components
#: on every axis.  SYNTHETIC, and it has to be said out loud: these decks'
#: real q-grid is 3x3x1, so a q taken from the deck would have q_z = 0 and
#: would make ``exp(2πi q·L)`` identically 1 on the one axis their L can
#: be non-zero on — the umklapp term would go untested while looking
#: exercised.  The star STRUCTURE (which q folds to which parent under
#: which op) is the deck's; the q VALUES are chosen to keep the phase live.
_Q_IRR = np.array([[0.0, 0.0, 0.0],
                   [1 / 3, 0.0, 1 / 4],
                   [0.0, 1 / 3, 1 / 3],
                   [1 / 3, 1 / 3, 1 / 6],
                   [1 / 6, 2 / 3, 5 / 12]])

#: The decks whose q-stars actually begin on a time-reversal row.
TRS_DECKS = ("gnppm_debug", "bispinor_debug")

#: The bar for "this term is worth something": ORDER UNITY, relative.  The
#: two ablations below MEASURE 1.76 (1.87 off-diagonal) for the TRS
#: conjugation and 1.69 for the umklapp phase on the committed draws, and
#: the docstrings record those numbers — but the ASSERTION is 1.0, because
#: pinning a particular draw's ratio would make the gate fail on a reseed
#: for a reason that has nothing to do with the machinery.  What has to
#: hold is that dropping the term costs as much as the answer is worth.
_ABLATION_WEIGHT = 1.0

#: Bar for "the kernel agrees with the hand reference": RELATIVE, matching
#: the L-c harness.  The identity claim — read-back against
#: ``unfold_isdf_operator`` on the same tables — is held to BIT equality
#: instead, because it is the same function on the same inputs and nothing may
#: differ.
RTOL = 1e-13


# ---------------------------------------------------------------------------
# Geometry builders — shared with the L-c multiproc harness
# ---------------------------------------------------------------------------

def star_tables(deck):
    """``(irr_idx, sym_idx, n_sym_spatial, n_ibz)`` from the frozen JSON.

    The committed table stores ``irr_idx_k`` as FULL-BZ LABELS (gnppm's are
    the non-monotone ``[0, 2, 6, 8, 7]``), which is what the band-index
    star helpers want.  ``unfold_isdf_operator`` wants a COMPACT index into the
    stored wedge's leading axis, so the labels are mapped to their
    first-occurrence position here.  Doing that conversion in one place is
    the point: a q-axis table that indexed by label would gather rows that
    do not exist.
    """
    with open(_STAR_TABLES) as fh:
        d = json.load(fh)[deck]
    labels = d["star_first_occurrence_labels"]
    pos = {int(v): i for i, v in enumerate(labels)}
    irr = np.array([pos[int(v)] for v in d["irr_idx_k"]], dtype=np.int32)
    sym = np.array(d["sym_idx_k"], dtype=np.int32)
    return irr, sym, int(d["n_sym_spatial"]), len(labels)


def deck_syms(deck):
    """``(sym_matrices, tnp)`` from ``mf_header/symmetry``, opened ``'r'``."""
    import h5py

    with h5py.File(_deck_stub.deck_path(deck, "WFN.h5"), "r") as f:
        g = f["mf_header"]["symmetry"]
        n = int(g["ntran"][()])
        return g["mtrx"][:n], g["tnp"][:n]


def closed_centroid_set(sym_matrices, tnp, fft_grid, seeds):
    """An ORBIT-CLOSED centroid set, closed by construction not by luck.

    The union of the seeds' orbits under the BGW r-action
    ``r' = mtrx⁻¹·r + τ``.  A union of orbits is closed under the group by
    definition, so ``centroid_source_map_and_wrap(validate=True)`` passes and
    the closure is a property of this function rather than of a file that
    somebody might regenerate.
    """
    S = np.asarray(sym_matrices, dtype=np.float64)
    rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    tint = np.rint(np.asarray(tnp, dtype=np.float64) / (2.0 * np.pi)
                   * fft_grid).astype(np.int64)
    imgs = set()
    for r in np.asarray(seeds, dtype=np.int64):
        for s in range(S.shape[0]):
            imgs.add(tuple(((rinv[s] @ r + tint[s]) % fft_grid).tolist()))
    return np.array(sorted(imgs), dtype=np.int32)


def hermitian_ibz(n_q, n_mu, seed=7):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n_q, n_mu, n_mu))
         + 1j * rng.standard_normal((n_q, n_mu, n_mu)))
    return 0.5 * (a + np.swapaxes(a.conj(), -1, -2))


def hand_unfold(V_ibz, tables, *, phase=True, conj=True):
    """The per-element reference, in plain numpy::

        V_full[q, μ, ν] = exp(2πi q_irr·(L_{s,μ} − L_{s,ν}))
                          · V_ibz[i(q), α_s(μ), α_s(ν)]

    conjugated on TRS rows.  ``phase=False`` / ``conj=False`` build the two
    ABLATIONS the arms weigh themselves against — a gate whose term
    contributes nothing is a gate that passes while testing nothing, and
    the L-b module's first draft shipped exactly that.
    """
    irr, sym = np.asarray(tables.irr_idx_q), np.asarray(tables.sym_idx_q)
    n_mu = int(np.asarray(tables.sym_perm).shape[-1])
    out = np.zeros((len(irr), n_mu, n_mu), dtype=V_ibz.dtype)
    for iq in range(len(irr)):
        parent, s = int(irr[iq]), int(sym[iq])
        p = np.asarray(tables.sym_perm[s])
        blk = V_ibz[parent][np.ix_(p, p)]
        if phase:
            qL = (np.asarray(tables.L_table[s], dtype=float)
                  @ np.asarray(tables.q_irr_frac)[parent])
            ph = np.exp(2j * np.pi * qL)
            blk = ph[:, None] * blk * np.conj(ph)[None, :]
        take_conj = conj and s >= int(tables.n_sym_spatial)
        out[iq] = np.conj(blk) if take_conj else blk
    return out


class _Arm:
    """One complete test vector: tables, wedge, verdict, references."""

    def __init__(self, label, sym_matrices, tnp, irr, sym, n_sym_spatial,
                 n_ibz, centroid_idx, fft_grid, q_irr, seed=7):
        self.label = label
        self.sym_matrices = np.asarray(sym_matrices)
        self.tnp = np.asarray(tnp)
        self.fft_grid = np.asarray(fft_grid)
        self.centroid_idx = centroid_idx
        self.centroid_frac = centroid_idx.astype(np.float64) / fft_grid
        self.verdict = verify_centroid_orbit_closure(
            self.centroid_frac, self.sym_matrices, tnp=self.tnp,
            fft_grid=self.fft_grid)
        perm, L = centroid_source_map_and_wrap(
            centroid_idx, self.sym_matrices, self.tnp, self.fft_grid,
            validate=True, extend_trs=True)
        self.n_mu = int(centroid_idx.shape[0])
        self.tables = QS.QirrTables(irr, sym, q_irr[:n_ibz], perm, L,
                                    n_sym_spatial)
        self.X_ibz = hermitian_ibz(n_ibz, self.n_mu, seed=seed)
        self.n_trs_rows = int((np.asarray(sym) >= n_sym_spatial).sum())

    def reference(self, **kw):
        return hand_unfold(self.X_ibz, self.tables, **kw)

    def kernel(self, mesh):
        """``unfold_isdf_operator`` on the wedge — what the uncompressed path
        used."""
        import jax.numpy as jnp
        t = self.tables
        return unfold_isdf_operator(
            jnp.asarray(self.X_ibz), irr_idx=t.irr_idx_q, sym_idx=t.sym_idx_q,
            sym_perm=t.sym_perm, L_table=t.L_table, q_irr_frac=t.q_irr_frac,
            mesh_xy=mesh, n_sym_spatial=int(t.n_sym_spatial))


def trs_arm(deck):
    """A TRS-active deck's real star structure on a synthetic closed set."""
    S, tnp = deck_syms(deck)
    irr, sym, n_spatial, n_ibz = star_tables(deck)
    cent = closed_centroid_set(S, tnp, _FFT, _SEEDS_12)
    return _Arm(deck, S, tnp, irr, sym, n_spatial, n_ibz, cent, _FFT,
                _Q_IRR)


def si_arm():
    """The NEGATIVE CONTROL: Si's real tables and its real closed 144-set.

    Everything here is production except the tensor: the 48 ops, the 64→8
    star reduction, and ``centroids_frac_144.txt``, which is the one
    orbit-closed centroid file in the tree.  Its q-axis carries ZERO TRS
    rows, which is the whole reason it is a control and not a test.
    """
    S, tnp = deck_syms("si_cohsex_debug")
    irr, sym, n_spatial, n_ibz = star_tables("si_cohsex_debug")
    grid = np.array([24, 24, 24], dtype=np.int64)
    frac = np.loadtxt(os.path.join(_deck_stub.regression_dir(),
                                   "si_cohsex_debug",
                                   "centroids_frac_144.txt"))
    idx = (np.rint(frac * grid).astype(np.int64) % grid).astype(np.int32)
    q = np.stack([np.array([i / 4.0, (i % 3) / 3.0, (i % 5) / 5.0])
                  for i in range(n_ibz)])
    return _Arm("si_cohsex_debug", S, tnp, irr, sym, n_spatial, n_ibz, idx,
                grid, q, seed=3)


#: A GLIDE: {σ_z | τ = (1/2, 0, 0)}.  Order two, because applying it twice
#: gives {I | τ + σ_z·τ} = {I | (1, 0, 0)} ≡ the identity mod the lattice —
#: so {I, g} really is a group and the orbit really does close.  τ×grid =
#: (6, 0, 0) is integer on the 12-grid, so the images land on grid points.
_NONSYM_SYMS = np.stack([np.eye(3, dtype=np.int64),
                         np.diag([1, 1, -1]).astype(np.int64)])
_NONSYM_TNP = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
_NONSYM_IRR = np.array([0, 1, 1, 2, 2], dtype=np.int32)
_NONSYM_SYM = np.array([0, 1, 0, 3, 0], dtype=np.int32)


def umklapp_arm():
    """A non-symmorphic table whose L rows are non-zero and VARY across μ.

    The phase is ``exp(2πi q·(L_μ − L_ν))``, a DIFFERENCE, so a set whose
    centroids all share one L makes it identically 1 however large L is.
    The glide's τ shift plus the z mirror give L components on two axes and
    a set with points both on and off the mirror plane, so the difference
    is genuinely non-zero — and :func:`test_the_umklapp_phase_is_not_
    optional` measures what dropping it costs rather than assuming.
    """
    cent = closed_centroid_set(_NONSYM_SYMS, _NONSYM_TNP, _FFT, _SEEDS_12)
    return _Arm("nonsymmorphic-glide", _NONSYM_SYMS, _NONSYM_TNP,
                _NONSYM_IRR, _NONSYM_SYM, 2, 3, cent, _FFT, _Q_IRR, seed=11)


# ---------------------------------------------------------------------------
# The assertion the campaign turns on
# ---------------------------------------------------------------------------

def local_blocks(got, ref):
    """``[(mu_idx, nu_idx, got_block, ref_block)]`` — P>1-safe pairing.

    At P=1 there is one block and it is the whole array.  At P>1 a
    ``P(None,'x','y')``-sharded array spans non-addressable devices and
    ``np.asarray`` on it RAISES, so each rank works from its own
    ``addressable_shards``; the shard's ``index`` names which global (q, μ,
    ν) block it holds, which is also how PLACEMENT gets checked — an
    allgather would erase it.

    The reference may be sharded too (``unfold_isdf_operator``'s output is), so
    blocks are paired by their index rather than by position.  MEASURED
    writing this: the first draft indexed a host array unconditionally and
    the emulated 2x2 could not see it — the real four-process leg failed
    immediately with "spans non-addressable devices", which is exactly the
    class of failure single-process coverage cannot reach.
    """
    import jax
    if jax.process_count() == 1:
        g, r = np.asarray(got), np.asarray(ref)
        n = g.shape[-1]
        return [(np.arange(n), np.arange(n), g, r)]
    ref_by_index = None
    if hasattr(ref, "addressable_shards"):
        ref_by_index = {str(s.index): np.asarray(s.data)
                        for s in ref.addressable_shards}
    n_mu = int(got.shape[-1])
    out = []
    for s in got.addressable_shards:
        idx = tuple(s.index)
        mu = np.arange(n_mu)[idx[1]]
        nu = np.arange(n_mu)[idx[2]]
        blk = (ref_by_index[str(s.index)] if ref_by_index is not None
               else np.asarray(ref)[s.index])
        out.append((mu, nu, np.asarray(s.data), blk))
    return out


def assert_offdiag_elementwise(got, ref, label, *, rtol=0.0):
    """Compare EVERY μ≠ν element, and never through a scalar summary.

    THIS IS THE WHOLE POINT OF THE GATE.  Both conjugation bugs this area
    has already shipped were diagonal-preserving and off-diagonal-
    destroying — 183.61 eV of error with the diagonal exactly zero — and
    every scalar health check in the pipeline (hermiticity, electron count,
    the spectrum, ``eqp.dat``) stayed green through both.  A Frobenius norm
    or a trace CANNOT see that class of failure, so nothing here reduces
    the residual before comparing it.

    Two masks, reported separately, because they fail for different
    reasons: the off-diagonal (μ≠ν) block is where a wrong conjugation
    lands, and the diagonal is where such a bug hides.  ``rtol=0`` demands
    bit equality.  Runs per addressable shard, so the multi-rank leg makes
    the SAME element-wise claim rather than a weaker one.
    """
    n_off = n_diag = 0
    worst_off = worst_diag = 0.0
    for mu, nu, g, r in local_blocks(got, ref):
        assert g.shape == r.shape, f"{label}: {g.shape} != {r.shape}"
        offdiag = mu[:, None] != nu[None, :]
        scale = float(np.abs(r).max()) if r.size else 1.0
        for name, mask in (("off-diagonal (mu != nu)", offdiag),
                           ("diagonal (mu == nu)", ~offdiag)):
            sub_g, sub_r = g[:, mask], r[:, mask]
            if sub_g.size == 0:
                continue
            worst = float(np.abs(sub_g - sub_r).max())
            if rtol == 0.0:
                n_bad = int((sub_g != sub_r).sum())
                assert n_bad == 0, (
                    f"{label}: {n_bad} of {sub_g.size} {name} elements "
                    f"differ, worst {worst:.3e}; this comparison is "
                    f"element-wise on purpose — a norm would not see it")
            else:
                assert worst <= rtol * scale, (
                    f"{label}: {name} worst {worst:.3e} exceeds "
                    f"{rtol:.1e} x {scale:.3e}")
            if mask is offdiag:
                n_off += sub_g.size
                worst_off = max(worst_off, worst)
            else:
                n_diag += sub_g.size
                worst_diag = max(worst_diag, worst)
    assert n_off > 0, (
        f"{label}: no off-diagonal elements were compared at all; that is "
        f"the one outcome this assertion must never have")
    return n_off, n_diag, worst_off, worst_diag


# ---------------------------------------------------------------------------
# Check bodies — called by pytest cells here AND by the L-c multiproc CLI
# ---------------------------------------------------------------------------

def _tmp_h5(tag):
    d = tempfile.mkdtemp(prefix=f"qirr_{tag}_")
    return d, os.path.join(d, "restart.h5")


def _shard_max_abs_diff(out, ref):
    """max|out − ref| over this rank's blocks; see :func:`local_blocks`."""
    worst = 0.0
    for _, _, g, r in local_blocks(out, ref):
        worst = max(worst, float(np.abs(g - r).max()))
    return worst


def check_qirr_round_trip_on_a_trs_deck(mesh, deck="gnppm_debug"):
    """Write the wedge, read it back, compare ELEMENT-WISE off-diagonal.

    The read-back is compared against ``unfold_isdf_operator`` on the same
    wedge and the same tables — the array the uncompressed path actually used —
    and the bar is BIT equality, not a tolerance.  That is the design's central
    claim: because the format persists the PRE-UNFOLD block, reader and
    uncompressed path evaluate the same function on the same inputs, so
    they agree by construction on every element rather than by a property
    that has to hold.
    """
    arm = trs_arm(deck)
    assert arm.n_trs_rows >= 4, (
        f"{deck} lost its TRS rows; this arm would test nothing")
    tmpdir, path = _tmp_h5(deck)
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz,
                             tables=arm.tables,
                             closure_verdict=arm.verdict)
        got, hdr = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
        assert hdr.q_storage == "ibz" and hdr.was_unfolded
        assert hdr.data_ready is True
        kern = arm.kernel(mesh)
        d = _shard_max_abs_diff(got, kern)
        assert d == 0.0, (
            f"{deck}: read-back differs from unfold_isdf_operator on the "
            f"same wedge by {d:.3e}; the round trip is supposed to be "
            f"an IDENTITY")
        n_off, _, _, _ = assert_offdiag_elementwise(
            got, kern, f"{deck} identity")
        assert_offdiag_elementwise(got, arm.reference(),
                                   f"{deck} vs hand reference", rtol=RTOL)
        return (f"{deck}: {arm.n_trs_rows} TRS rows, n_mu={arm.n_mu}, "
                f"{arm.tables.n_q_ibz}->{arm.tables.n_q_full} q, "
                f"{n_off} off-diagonal elements exact")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def check_qirr_umklapp_phase_survives_the_store(mesh):
    """The umklapp phase is not optional, and the arm proves it is live.

    Ablation first: the same reference with the phase dropped differs from
    the true one by order unity.  Then the stored/read path is asserted
    against the true one.  Without the ablation this cell would pass on a
    geometry where the phase happened to be 1 — which is exactly what the
    TRS decks are, and why this arm exists separately.
    """
    arm = umklapp_arm()
    ref = arm.reference()
    nophase = arm.reference(phase=False)
    rel = float(np.abs(nophase - ref).max() / np.abs(ref).max())
    assert rel > _ABLATION_WEIGHT, (
        f"dropping the umklapp phase only moved the answer by {rel:.3e}; "
        f"this geometry does not exercise the phase and the arm is vacuous")
    tmpdir, path = _tmp_h5("umklapp")
    try:
        QS.write_qirr_tensor(path, "V_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        got, _ = QS.read_tensor(path, "V_qmunu", mesh_xy=mesh)
        d_true = _shard_max_abs_diff(got, arm.kernel(mesh))
        assert d_true == 0.0, (
            f"stored/read path differs from unfold_isdf_operator by "
            f"{d_true:.3e}")
        d_nophase = _shard_max_abs_diff(got, nophase)
        assert d_nophase > 0.0, (
            "the read path agrees with the PHASELESS reference; the stored "
            "L_table is not reaching the unfold")
        n_off, _, _, _ = assert_offdiag_elementwise(
            got, ref, "umklapp arm", rtol=RTOL)
        return (f"umklapp arm: dropping the phase costs rel {rel:.3f}, "
                f"{n_off} off-diagonal elements kept it")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mesh_here(px=1, py=1):
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, platform="cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py),
                ("x", "y"))


# ---------------------------------------------------------------------------
# Arm 1 — the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", TRS_DECKS)
def test_the_round_trip_is_exact_on_a_trs_active_deck(deck):
    check_qirr_round_trip_on_a_trs_deck(_mesh_here(), deck)


@pytest.mark.parametrize("deck", TRS_DECKS)
def test_the_round_trip_is_exact_on_an_emulated_2x2(deck):
    """The same body on four emulated devices, when the flag took.

    SKIPS below four through ``require_devices``, which skips and never
    asserts: the device count is a property of how the process was
    launched, not of the code under test.
    """
    check_qirr_round_trip_on_a_trs_deck(_mesh_here(2, 2), deck)


@pytest.mark.parametrize("deck", TRS_DECKS)
def test_the_trs_conjugation_is_what_the_arm_is_testing(deck):
    """ANTI-TAUTOLOGY, and the reason Si cannot stand in for these decks.

    Ablate the conjugation on the TRS rows and the answer moves by 1.76
    relative overall and 1.87 on the off-diagonals alone (MEASURED).  So
    the arm above is genuinely exercising the antiunitary branch.  The
    umklapp phase on these same decks is worth NOTHING — their q-stars use
    only rows {0, 2}, the identity and the TRS-identity, so L is zero
    everywhere the q axis reaches and dropping the phase moves the answer
    by exactly 0.  Both halves are asserted, because "this arm covers the
    conjugation" and "this arm does not cover the phase" are equally
    load-bearing and the second is why the umklapp arm is separate.
    """
    arm = trs_arm(deck)
    ref = arm.reference()
    n_mu = arm.n_mu
    off = ~np.eye(n_mu, dtype=bool)

    noconj = arm.reference(conj=False)
    rel = float(np.abs(noconj - ref).max() / np.abs(ref).max())
    rel_off = float(np.abs((noconj - ref)[:, off]).max()
                    / np.abs(ref[:, off]).max())
    assert rel > _ABLATION_WEIGHT and rel_off > _ABLATION_WEIGHT, (
        f"{deck}: dropping the TRS conjugation moved the answer by only "
        f"{rel:.3e} ({rel_off:.3e} off-diagonal); the antiunitary branch "
        f"is not live on this arm")

    nophase = arm.reference(phase=False)
    assert float(np.abs(nophase - ref).max()) == 0.0, (
        f"{deck}: the umklapp phase is no longer inert here.  That is not "
        f"a failure, but this docstring's claim about what the arm covers "
        f"has gone stale and the split with the umklapp arm needs re-"
        f"reading")
    used = sorted({int(s) for s in arm.tables.sym_idx_q.tolist()})
    assert used == [0, 2], f"{deck} q-stars now use sym rows {used}"


def test_silicon_is_a_negative_control_and_says_so():
    """Si round-trips exactly and PROVES NOTHING about time reversal.

    MEASURED: ``ntran = 48`` and **zero** TRS rows across all 64 k — every
    k is rotation-reachable, so silicon structurally cannot exercise the
    antiunitary branch.  This cell asserts the round trip (Si is the only
    arm here running on real production tables end to end: 48 real ops, the
    real 64→8 star reduction, and the real orbit-closed
    ``centroids_frac_144.txt``) AND asserts the zero, so that nobody later
    reads a green Si arm as coverage of the conjugation.  A fix verified
    only on Si is how the ``kin_ion`` unfold bug survived.
    """
    mesh = _mesh_here()
    arm = si_arm()
    assert arm.n_trs_rows == 0, (
        "si_cohsex_debug grew TRS rows; the control has become a test and "
        "the comment above is wrong")
    assert arm.verdict.closed, arm.verdict.describe()
    assert arm.n_mu == 144 and arm.tables.n_q_full == 64
    assert arm.tables.n_q_ibz == 8
    tmpdir, path = _tmp_h5("si")
    try:
        QS.write_qirr_tensor(path, "V_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        # The disk win, on the one production-shaped arm in this file.
        assert arm.tables.n_q_full / arm.tables.n_q_ibz == 8.0
        got, hdr = QS.read_tensor(path, "V_qmunu", mesh_xy=mesh)
        assert hdr.q_storage == "ibz"
        assert_offdiag_elementwise(got, arm.kernel(mesh), "si identity")
        assert_offdiag_elementwise(got, arm.reference(), "si vs reference",
                                   rtol=RTOL)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_umklapp_phase_is_not_optional():
    check_qirr_umklapp_phase_survives_the_store(_mesh_here())


def test_the_umklapp_arm_on_an_emulated_2x2():
    check_qirr_umklapp_phase_survives_the_store(_mesh_here(2, 2))


def test_unfold_false_hands_back_the_wedge_itself():
    """The small array, for a caller that wants to hold it.

    TWIN of the round trip: same file, and the wedge that comes back is
    bit-identical to what went in, so the unfold is the only thing the
    ``unfold=True`` path adds.
    """
    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("wedge")
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        wedge, hdr = QS.read_tensor(path, "W0_qmunu", unfold=False)
        assert np.array_equal(np.asarray(wedge), arm.X_ibz)
        assert hdr.q_storage == "ibz" and hdr.n_q_on_disk == 5
        assert hdr.n_q_full == 9
        with pytest.raises(ValueError, match="needs a mesh"):
            QS.read_tensor(path, "W0_qmunu")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Arm 2 — the refusal suite, each with its constructible twin
# ---------------------------------------------------------------------------

def test_the_writer_refuses_the_production_centroid_set():
    """REFUSAL 1, on the real vector: the 960-centroid production set.

    This is not a hypothetical.  ``si_cohsex_debug`` runs on
    ``centroids_frac_960.txt``, which is NOT orbit-closed — 47 of 48 ops
    violating, worst 1.318e-01 — so refusing is the branch production
    actually hits today, and it is why the deck key must keep full-BZ
    storage as the default until the owner rules on regenerating
    centroids.  A q_irr file written against that set would be silently
    unrecoverable: there is no α to invert with.

    TWIN: the orbit-closed 144-set through the identical call, accepted.
    """
    S, tnp = deck_syms("si_cohsex_debug")
    reg = os.path.join(_deck_stub.regression_dir(), "si_cohsex_debug")
    bad = verify_centroid_orbit_closure(
        np.loadtxt(os.path.join(reg, "centroids_frac_960.txt")), S, tnp=tnp)
    assert not bad.closed and bad.n_violating == 47

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("refuse")
    try:
        with pytest.raises(RuntimeError) as exc:
            QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz,
                                 tables=arm.tables, closure_verdict=bad)
        msg = str(exc.value)
        assert "refuses q_irr storage" in msg and "NOT CLOSED" in msg
        assert "47/48" in msg
        assert "s=1:" in msg, "the refusal must name the offending ops"
        assert not os.path.exists(path), (
            "the writer created the file before refusing; a half-written "
            "q_irr artifact is worse than none")

        # TWIN: the closed verdict goes through the same call.
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz,
                             tables=arm.tables, closure_verdict=arm.verdict)
        assert os.path.exists(path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_writer_will_not_write_without_the_question_being_asked():
    """There is no path to disk that skips the closure verdict.

    Neither argument, both arguments, and a verdict with the raw inputs
    beside it (which would stamp a measurement that is not the one being
    described) are all refusals.  TWIN: each of the two legal forms
    writes.
    """
    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("noverdict")
    try:
        with pytest.raises(ValueError, match="exactly one of"):
            QS.write_qirr_tensor(path, "W", arm.X_ibz, tables=arm.tables)
        with pytest.raises(ValueError, match="exactly one of"):
            QS.write_qirr_tensor(path, "W", arm.X_ibz, tables=arm.tables,
                                 closure_verdict=arm.verdict,
                                 centroids_frac=arm.centroid_frac)
        with pytest.raises(ValueError, match="already carries the answer"):
            QS.write_qirr_tensor(path, "W", arm.X_ibz, tables=arm.tables,
                                 closure_verdict=arm.verdict,
                                 sym_matrices=arm.sym_matrices)
        with pytest.raises(TypeError, match="CentroidClosureVerdict"):
            QS.write_qirr_tensor(path, "W", arm.X_ibz, tables=arm.tables,
                                 closure_verdict=True)
        # TWIN A: the verdict form.
        QS.write_qirr_tensor(path, "W_a", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        # TWIN B: the take-the-verdict-here form, which keeps the tnp/tau
        # exclusive contract all the way down.
        QS.write_qirr_tensor(path, "W_b", arm.X_ibz, tables=arm.tables,
                             centroids_frac=arm.centroid_frac,
                             sym_matrices=arm.sym_matrices, tnp=arm.tnp,
                             fft_grid=arm.fft_grid)
        with pytest.raises(ValueError, match="exactly one of tnp= or tau="):
            QS.write_qirr_tensor(path, "W_c", arm.X_ibz, tables=arm.tables,
                                 centroids_frac=arm.centroid_frac,
                                 sym_matrices=arm.sym_matrices)
        a, _ = QS.read_tensor(path, "W_a", unfold=False)
        b, _ = QS.read_tensor(path, "W_b", unfold=False)
        assert np.array_equal(np.asarray(a), np.asarray(b))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_tampered_table_refuses_at_read():
    """REFUSAL 2: the table hash.

    The tables are what make the tensor recoverable.  Edit one row of
    ``sym_perm`` in the file — the shape is untouched, the tensor is
    untouched, every extent still agrees — and the reader must refuse
    rather than gather a permutation of the wrong centroids at every q.

    TWIN: the same file, read before the edit.
    """
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("tamper")
    mesh = _mesh_here()
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)          # TWIN
        with h5py.File(path, "a") as f:
            perm = f["W0_qmunu__qirr/sym_perm"][()]
            perm[1, [0, 1]] = perm[1, [1, 0]]      # a legal permutation!
            f["W0_qmunu__qirr/sym_perm"][...] = perm
        with pytest.raises(ValueError, match="table hash mismatch"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_missing_table_group_refuses():
    """The tables must be IN the file; a tensor without them is a ruin."""
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("notables")
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        with h5py.File(path, "a") as f:
            del f["W0_qmunu__qirr/L_table"]
        with pytest.raises(ValueError, match="missing"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=_mesh_here())
        with h5py.File(path, "a") as f:
            del f["W0_qmunu__qirr"]
        with pytest.raises(ValueError, match="carries no"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=_mesh_here())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_version_mismatch_refuses():
    """REFUSAL 3: a reader must not read a layout it was not built for.

    Best-effort reading of an unknown version is how a format returns wrong
    numbers on the day it changes.  TWIN: the correct version reads.
    """
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("version")
    mesh = _mesh_here()
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        _, hdr = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)   # TWIN
        assert hdr.format_version == QS.QIRR_FORMAT_VERSION
        with h5py.File(path, "a") as f:
            f["W0_qmunu"].attrs["qirr_format_version"] = np.int64(
                QS.QIRR_FORMAT_VERSION + 1)
        with pytest.raises(ValueError, match="format version"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_rank_4_tensor_stamped_version_1_refuses_by_name():
    """REFUSAL 3b: THE HAZARD, CONSTRUCTED — and refused on the RANK.

    The version check above catches a file that SAYS it is another
    layout.  This catches the one that lies about it, which is the case
    the version number cannot reach: a writer that gained an axis
    without bumping, or a hand-edited attr.  A reader that trusts the
    stamp has no second opinion; the rank is one, because it is a
    property of the bytes.

    THE MALICIOUS FILE IS BUILT TO BE MAXIMALLY QUIET.  The leading
    extent is chosen EQUAL to the wedge extent, so ``shape[0]`` is the
    number a version-1 reader expects, ``shape[-1]`` is genuinely N_μ,
    and everything downstream of the guard agrees: the tables validate
    against both extents, the shape verdict is still ``"ibz"``, the
    stamped digest still matches the tables on disk, ``q_storage`` still
    cross-checks and the readiness flag is still set.  Those four are
    ASSERTED below rather than assumed — without them the cell would be
    refusing something for a reason that has nothing to do with the
    rank, and would pass on a reader that had no rank guard at all.

    That coincidence is one deck away, not a hypothetical: Si 4³ reduces
    64 q to 8 and an n_p = 4 multipole fit samples 8 frequencies.  Left
    unrefused, the caller gets 4-D bytes back with the frequency axis
    relabelled q and a header that says so.

    TWIN: the same tables and the same wedge at rank 3 — the honest
    version-1 file — read and unfold.
    """
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("rank")
    mesh = _mesh_here()
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        _, hdr = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)   # TWIN
        assert hdr.format_version == QS.QIRR_FORMAT_VERSION
        assert hdr.n_q_on_disk == arm.tables.n_q_ibz

        # THE COINCIDENCE, on purpose: n_omega EQUAL to the wedge extent.
        n_omega = arm.tables.n_q_ibz
        W = np.stack([arm.X_ibz * (1.0 + i) for i in range(n_omega)])
        assert W.ndim == 4 and W.shape[0] == W.shape[1], (
            "this cell is only a test of the rank guard while the leading "
            "extent matches the wedge extent; otherwise an extent check "
            "would catch it first and the guard would go unexercised")

        with h5py.File(path, "a") as f:
            attrs = dict(f["W0_qmunu"].attrs)
            del f["W0_qmunu"]
            f.create_dataset("W0_qmunu", data=W)
            for key, val in attrs.items():
                f["W0_qmunu"].attrs[key] = val

        # NOT VACUOUS: every version-1 check the guard precedes still
        # passes on these bytes.  The table validation reads the leading
        # extent as the q extent and calls the file a wedge; the digest,
        # the attr and the readiness flag are the writer's own.
        assert QS.validate_qirr_tables(
            arm.tables, int(W.shape[0]), arm.n_mu) == "ibz"
        assert attrs["qirr_table_hash"] == arm.tables.canonical().digest()
        assert attrs["q_storage"] == "ibz"
        assert bool(attrs["qirr_data_ready"]) is True
        assert int(attrs["qirr_format_version"]) == QS.QIRR_FORMAT_VERSION

        # THE REFUSAL.  Rank first, and it names BOTH numbers.
        for kw in ({"unfold": False}, {"mesh_xy": mesh}):
            with pytest.raises(ValueError) as exc:
                QS.read_tensor(path, "W0_qmunu", **kw)
            msg = str(exc.value)
            assert "qirr_format_version=1" in msg, msg
            assert "rank 3" in msg and "rank 4" in msg, msg
            assert str(tuple(int(s) for s in W.shape)) in msg, msg
            assert "RANK IS THE DISCRIMINANT" in msg, msg
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_format_plumbing_is_reachable_through_the_door():
    """The promotion, pinned where it has to hold: on the PACKAGE.

    A second store writing this layout needs the same handle opener, the
    same attr decoder, the same provenance stamp, the same table
    validation and the same version-attr spelling.  Reaching
    ``symmetry_maps.qirr_store`` for them is a past-the-door edge that
    ``tests/test_layering.py`` rule 6 counts, so they are on the door —
    and a name that resolves but is absent from ``__all__`` is one a
    consumer discovers is missing at run time rather than at import
    time.  Both are asserted, for the same reason the rename-compat
    suite asserts both.
    """
    import symmetry_maps

    for name in ("QIRR_VERSION_ATTR", "QIRR_RANK_BY_VERSION",
                 "QIRR_TABLE_SUFFIX", "QirrDest", "qirr_attr_str",
                 "qirr_generator_commit", "validate_qirr_tables"):
        assert hasattr(symmetry_maps, name), (
            f"``from symmetry_maps import {name}`` does not resolve; a "
            f"consumer would have to reach the qirr_store submodule for "
            f"it, which is the layering violation the promotion removed")
        assert name in symmetry_maps.__all__, (
            f"{name} resolves but is not in __all__, so ``import *`` "
            f"drops it")
        assert getattr(symmetry_maps, name) is getattr(QS, name), (
            f"the door's {name} is not the module's; two bindings are two "
            f"answers the day one of them moves")
    assert symmetry_maps.QIRR_RANK_BY_VERSION == {QS.QIRR_FORMAT_VERSION: 3}
    assert symmetry_maps.QIRR_VERSION_ATTR == "qirr_format_version"


def test_shape_and_attr_must_agree_and_the_shape_wins_the_argument():
    """REFUSAL 4: the cross-check, in both directions.

    ζ has inferred IBZ-versus-full from the stored q extent since before
    this format existed, so the SHAPE is the primary discriminant and
    ``q_storage`` is its cross-check.  Flip the attr to "full" on a wedge
    and the reader must refuse — not quietly believe the attr and return an
    unfolded tensor as though it were already full, and not quietly believe
    the shape and ignore a stamp that says otherwise.  A stamped-and-wrong
    attr is exactly what a cross-check exists to catch.

    TWIN: the attr as written, agreeing, reads.
    """
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("shape")
    mesh = _mesh_here()
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        _, hdr = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)   # TWIN
        assert hdr.q_storage == "ibz"
        with h5py.File(path, "a") as f:
            f["W0_qmunu"].attrs["q_storage"] = "full"
        with pytest.raises(ValueError, match="SHAPE is the primary"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
        # And the n_q_full stamp is cross-checked against the table too.
        with h5py.File(path, "a") as f:
            f["W0_qmunu"].attrs["q_storage"] = "ibz"
            f["W0_qmunu"].attrs["qirr_n_q_full"] = np.int64(11)
        with pytest.raises(ValueError, match="stamps n_q_full"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tables_that_do_not_describe_the_tensor_refuse_at_write():
    """Every extent the unfold silently clips on, refused at the door.

    ``promise_in_bounds`` gathers CLIP rather than raise, which is the
    exact failure shape of the TRS bug: an out-of-range sym row quietly
    became the last row and every time-reversed q came back wrong.  So each
    of these is a refusal with its own message.  TWIN: the consistent
    tables write.
    """
    arm = trs_arm("gnppm_debug")
    t = arm.tables
    tmpdir, path = _tmp_h5("extent")

    def bad(**kw):
        f = dict(irr_idx_q=t.irr_idx_q, sym_idx_q=t.sym_idx_q,
                 q_irr_frac=t.q_irr_frac, sym_perm=t.sym_perm,
                 L_table=t.L_table, n_sym_spatial=t.n_sym_spatial)
        f.update(kw)
        return QS.QirrTables(**f)

    try:
        QS.write_qirr_tensor(path, "ok", arm.X_ibz, tables=t,       # TWIN
                             closure_verdict=arm.verdict)
        cases = [
            (bad(q_irr_frac=t.q_irr_frac[:-1]), "do not describe"),
            (bad(sym_perm=t.sym_perm[:, :-1]), "must share one"),
            (bad(sym_idx_q=t.sym_idx_q[:-1]), "index the same q axis"),
            (bad(L_table=t.L_table[:, :, :2]), "L_table must be"),
            # Sym rows that do not COVER max(sym_idx_q): the exact shape
            # of the TRS bug, where an out-of-range row was clipped to the
            # last one and every time-reversed q came back wrong.
            (bad(sym_perm=t.sym_perm[:2], L_table=t.L_table[:2]),
             "reaches row 2"),
            # Rows that cover but are not the doubled TRS table: time
            # reversal keeps r fixed, so the augmented half must DUPLICATE
            # the spatial half and three rows cannot be that.
            (bad(sym_perm=t.sym_perm[:3], L_table=t.L_table[:3]),
             "TRS-augmented"),
            (bad(irr_idx_q=np.zeros_like(t.irr_idx_q)), "every stored IBZ"),
            (bad(n_sym_spatial=0), "n_sym_spatial must be positive"),
        ]
        for tables, pattern in cases:
            with pytest.raises(ValueError, match=pattern):
                QS.write_qirr_tensor(path, "bad", arm.X_ibz, tables=tables,
                                     closure_verdict=arm.verdict)
        with pytest.raises(ValueError, match=r"n_q_ibz, n_mu, n_mu"):
            QS.write_qirr_tensor(path, "bad", arm.X_ibz[0], tables=t,
                                 closure_verdict=arm.verdict)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_centroid_hash_is_the_readers_identity_check():
    """A consumer can insist the tensor was written against ITS set.

    ``sym_perm`` addresses centroids by row position, so a different set —
    or the same set in a different order — is a different permutation and
    would gather the wrong μ at every q.  TWIN: the matching hash reads.
    """
    arm = trs_arm("gnppm_debug")
    other = si_arm()
    tmpdir, path = _tmp_h5("hash")
    mesh = _mesh_here()
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh,             # TWIN
                       expect_centroid_hash=arm.verdict.centroid_hash)
        with pytest.raises(ValueError, match="was written against centroid"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh,
                           expect_centroid_hash=other.verdict.centroid_hash)
        digest = arm.tables.digest()
        QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh,             # TWIN
                       expect_table_hash=digest)
        with pytest.raises(ValueError, match="does not match the expected"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh,
                           expect_table_hash="sha256:" + "0" * 64)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_table_digest_moves_when_any_table_moves():
    """The digest has to be sensitive to every table, and to dtype drift.

    A digest that ignored ``L_table`` would let the umklapp phase decay
    silently; one that depended on the caller's dtype would refuse a
    perfectly good file because ``centroid_source_map_and_wrap`` returns
    ``L`` as int8.  Both are checked here.
    """
    arm = trs_arm("gnppm_debug")
    t = arm.tables
    base = t.digest()
    assert base.startswith("sha256:")

    # Dtype-insensitive: int8 L and int64 L are the same tables.
    wide = QS.QirrTables(t.irr_idx_q, t.sym_idx_q, t.q_irr_frac,
                         np.asarray(t.sym_perm, dtype=np.int64),
                         np.asarray(t.L_table, dtype=np.int32),
                         t.n_sym_spatial)
    assert wide.digest() == base

    for field, mutated in (
            ("irr_idx_q", np.roll(t.irr_idx_q, 1)),
            ("sym_idx_q", np.roll(t.sym_idx_q, 1)),
            ("q_irr_frac", t.q_irr_frac + 1e-9),
            ("sym_perm", t.sym_perm[::-1]),
            ("L_table", t.L_table + 1),
            ("n_sym_spatial", 1)):
        kw = dict(irr_idx_q=t.irr_idx_q, sym_idx_q=t.sym_idx_q,
                  q_irr_frac=t.q_irr_frac, sym_perm=t.sym_perm,
                  L_table=t.L_table, n_sym_spatial=t.n_sym_spatial)
        kw[field] = mutated
        assert QS.QirrTables(**kw).digest() != base, (
            f"the digest is blind to {field}")


# ---------------------------------------------------------------------------
# Arm 3 — backward compatibility, and the vacuous-presence gate
# ---------------------------------------------------------------------------

def test_a_legacy_file_with_no_attrs_reads_byte_for_byte():
    """REFUSAL 5's twin, and the backward-compatibility contract itself.

    Every restart file written before this format is full-BZ and carries no
    version attr, so "no attrs" is read as ``q_storage='full'`` and the
    array comes back UNCHANGED — no tables consulted, no unfold, no mesh
    required.  The file here is hand-written with plain h5py, exactly as a
    pre-format writer would have left it, and the comparison is
    byte-for-byte on the raw buffer rather than elementwise on floats.
    """
    import h5py

    tmpdir, path = _tmp_h5("legacy")
    try:
        legacy = hermitian_ibz(9, 12, seed=99)
        with h5py.File(path, "w") as f:
            f.create_dataset("W0_qmunu", data=legacy)
        got, hdr = QS.read_tensor(path, "W0_qmunu")
        assert hdr.is_legacy and hdr.format_version is None
        assert hdr.q_storage == "full"
        assert hdr.data_ready is None and hdr.centroid_hash is None
        assert hdr.n_q_on_disk == 9
        assert np.asarray(got).tobytes() == legacy.tobytes()
        # No mesh was needed and none was passed: nothing unfolded.
        assert not hdr.was_unfolded
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_half_stamped_file_refuses_instead_of_reading_as_legacy():
    """The hole the "no attrs means full" rule would otherwise leave open.

    A dataset carrying ``q_storage="ibz"`` but no version attr would fall
    through the legacy branch and be handed back UNFOLDED as though it were
    already full-BZ — a wedge silently presented as a whole BZ.  So a
    PARTIAL stamp refuses, and only a completely bare dataset takes the
    compatibility path.

    TWIN: the bare dataset, one attr short of this one, reads fine.
    """
    import h5py

    tmpdir, path = _tmp_h5("halfstamp")
    try:
        legacy = hermitian_ibz(5, 12, seed=99)
        with h5py.File(path, "w") as f:
            f.create_dataset("bare", data=legacy)
            ds = f.create_dataset("half", data=legacy)
            ds.attrs["q_storage"] = "ibz"
        QS.read_tensor(path, "bare")                               # TWIN
        with pytest.raises(ValueError, match="but no 'qirr_format_version'"):
            QS.read_tensor(path, "half")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_zero_placeholder_never_reads_as_data():
    """THE VACUOUS-PRESENCE GATE, in the q_irr format.

    ``gw_init`` allocates a full-size zero ``W0`` unconditionally, so the
    restart gate's ``os.path.exists`` is always true and a run whose
    ``persist_w0`` never fired finds a file of exactly the right shape full
    of zeros and proceeds.  That is the mechanism behind the April BSE
    incident: a plausible excitonic spectrum out of an all-zero screening
    tensor, with every shape check green.  A q_irr format that inherited
    the property would just make the zero file eight times smaller.

    So the reader refuses on ``qirr_data_ready=False``.  The placeholder
    here is built by the real writer, not forged — it passes every other
    check in the file, which is precisely why presence cannot be the test.

    TWIN: the same dataset name, written with real data, reads.
    """
    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("placeholder")
    mesh = _mesh_here()
    try:
        QS.allocate_qirr_placeholder(
            path, "W0_qmunu", (arm.tables.n_q_ibz, arm.n_mu, arm.n_mu),
            tables=arm.tables, closure_verdict=arm.verdict)
        with pytest.raises(ValueError, match="PLACEHOLDER, not"):
            QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)

        # Everything ELSE about the placeholder is correct, which is the
        # whole hazard: shape, tables, hashes and closure all agree.
        zeros, hdr = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh,
                                    require_persisted=False)
        assert hdr.data_ready is False
        assert hdr.q_storage == "ibz"
        assert hdr.centroid_hash == arm.verdict.centroid_hash
        assert not np.asarray(zeros).any(), "the placeholder is not zeros"

        # TWIN: real data over the top, and the flag flips.
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        got, hdr2 = QS.read_tensor(path, "W0_qmunu", mesh_xy=mesh)
        assert hdr2.data_ready is True
        assert np.abs(np.asarray(got)).max() > 0.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_provenance_is_stamped_the_way_kin_ion_stamps_it():
    """A file that cannot be told from a stale one is a file nobody checks.

    ``kin_ion.h5`` records the generator commit and the content hash of
    what it was made from, because a broken committed fixture once went a
    month unnoticed for want of exactly that.  The q_irr format follows:
    the writer's commit, the write time, the writer's name, the closure
    verdict AS MEASURED, and whatever the caller can prove, under a
    ``prov_`` prefix that cannot collide with the format's own attrs.
    """
    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("prov")
    try:
        hdr = QS.write_qirr_tensor(
            path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
            closure_verdict=arm.verdict,
            provenance={"deck": "gnppm_debug", "n_centroids": 12})
        assert hdr.provenance["deck"] == "gnppm_debug"
        assert int(hdr.provenance["n_centroids"]) == 12
        assert hdr.provenance["qirr_writer"] == "symmetry_maps.qirr_store"
        assert hdr.provenance["qirr_generator_commit"]
        assert "T" in hdr.provenance["qirr_written_utc"]
        assert hdr.closure_verdict.startswith("closed ")
        assert hdr.closure_worst_residual == arm.verdict.worst_residual
        assert hdr.closure_tol == arm.verdict.tol
        _, back = QS.read_tensor(path, "W0_qmunu", unfold=False)
        assert back.provenance["deck"] == "gnppm_debug"
        assert back.closure_verdict == hdr.closure_verdict
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_reader_refuses_a_name_that_is_not_there():
    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("missing")
    try:
        QS.write_qirr_tensor(path, "W0_qmunu", arm.X_ibz, tables=arm.tables,
                             closure_verdict=arm.verdict)
        QS.read_tensor(path, "W0_qmunu", unfold=False)             # TWIN
        with pytest.raises(KeyError, match="not in this file"):
            QS.read_tensor(path, "V_qmunu", unfold=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_group_handle_and_a_path_are_the_same_door():
    """``h5group_or_path``, both spellings, one file.

    Phase 3 will hand this an already-open ``h5py.File`` from inside a
    writer that owns the handle; the tests hand it a path.  If those were
    different code paths only one of them would be covered.
    """
    import h5py

    arm = trs_arm("gnppm_debug")
    tmpdir, path = _tmp_h5("handle")
    try:
        with h5py.File(path, "w") as f:
            grp = f.create_group("restart")
            QS.write_qirr_tensor(grp, "W0_qmunu", arm.X_ibz,
                                 tables=arm.tables,
                                 closure_verdict=arm.verdict)
        with h5py.File(path, "r") as f:
            got, hdr = QS.read_tensor(f["restart"], "W0_qmunu",
                                      unfold=False)
            assert np.array_equal(np.asarray(got), arm.X_ibz)
            assert hdr.q_storage == "ibz"
        with pytest.raises(TypeError, match="File/Group or a path"):
            QS.read_tensor(object(), "W0_qmunu")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_wedge_that_is_the_whole_bz_is_labelled_full():
    """The degenerate case, labelled the way ζ has always labelled it.

    When the group does not reduce the q-grid at all (ntran=1, or a nosym
    run) the "wedge" IS the full BZ, there is nothing to unfold, and the
    honest ``q_storage`` is ``"full"`` — for writer and reader alike, so
    the cross-check cannot fire spuriously on a run that did nothing wrong.
    """
    n_q, n_mu = 4, 12
    tables = QS.QirrTables(
        irr_idx_q=np.arange(n_q, dtype=np.int32),
        sym_idx_q=np.zeros(n_q, dtype=np.int32),
        q_irr_frac=np.zeros((n_q, 3)),
        sym_perm=np.tile(np.arange(n_mu, dtype=np.int32), (2, 1)),
        L_table=np.zeros((2, n_mu, 3), dtype=np.int8),
        n_sym_spatial=1)
    arm = trs_arm("gnppm_debug")
    X = hermitian_ibz(n_q, n_mu, seed=5)
    tmpdir, path = _tmp_h5("nosym")
    try:
        hdr = QS.write_qirr_tensor(path, "V_qmunu", X, tables=tables,
                                   closure_verdict=arm.verdict)
        assert hdr.q_storage == "full"
        got, back = QS.read_tensor(path, "V_qmunu")
        assert back.q_storage == "full" and not back.was_unfolded
        assert np.array_equal(np.asarray(got), X)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Arm 8 — the μ PAD does not reach disk (SHARDING_RULES §2)
# ---------------------------------------------------------------------------
# THE RULE, AND WHY IT BINDS THIS FORMAT.  The producer bakes the μ pad into
# ``sym_perm``/``L_table`` once at construction — identity tail on the
# permutation, zero tail on the wrap — and the pad width is
# ``padded_mu_extent(n_rmu, device_count())``.  It is DEVICE-COUNT-DEPENDENT,
# and SHARDING_RULES §2 forbids those in a restart artifact for the reason a
# restart artifact exists: a file written on four ranks must read on eight.
# Every tensor in the restart format already obeys that rule
# (``tagged_arrays._mu_logical_shape`` clips each one on the way out).  The
# unfold tables are new, and they must obey it too.
#
# So: the WRITER strips, having first asserted the tail really is a pad, and
# the READER re-applies its OWN.  The three refusals below are the assertion,
# and they are the point — a "pad" that is not one is real structure, and
# stripping it would produce a file this format could not invert.


def _pad(arm, n_pad):
    """``arm``'s tables and wedge, re-padded to ``n_pad`` the way a producer
    would: identity tail on the permutation, zero tail on the wrap and on
    the tensor."""
    n_log = arm.n_mu
    tail = np.broadcast_to(np.arange(n_log, n_pad, dtype=np.int32),
                           (arm.tables.sym_perm.shape[0], n_pad - n_log))
    perm = np.concatenate([np.asarray(arm.tables.sym_perm, np.int32), tail],
                          axis=1)
    L = np.asarray(arm.tables.L_table)
    Lp = np.concatenate(
        [L, np.zeros((L.shape[0], n_pad - n_log, 3), dtype=L.dtype)], axis=1)
    tables = QS.QirrTables(arm.tables.irr_idx_q, arm.tables.sym_idx_q,
                           arm.tables.q_irr_frac, perm, Lp,
                           arm.tables.n_sym_spatial)
    X = np.zeros((arm.X_ibz.shape[0], n_pad, n_pad), dtype=arm.X_ibz.dtype)
    X[:, :n_log, :n_log] = arm.X_ibz
    return tables, X


def test_the_pad_is_stripped_on_write_and_the_file_states_the_logical_extent():
    """The file must not be able to say how many devices wrote it."""
    arm = trs_arm("gnppm_debug")
    n_pad = arm.n_mu + 4
    tables, X = _pad(arm, n_pad)
    d, path = _tmp_h5("padstrip")
    try:
        hdr = QS.write_qirr_tensor(path, "V_qmunu", X, tables=tables,
                                   closure_verdict=arm.verdict,
                                   n_rmu_logical=arm.n_mu, mode="w")
        assert hdr.n_rmu_logical == arm.n_mu
        import h5py
        with h5py.File(path, "r") as f:
            assert f["V_qmunu"].shape[-1] == arm.n_mu, (
                f"the pad reached disk: dataset μ extent "
                f"{f['V_qmunu'].shape[-1]} against logical {arm.n_mu}")
            assert int(f["V_qmunu"].attrs["qirr_n_rmu_logical"]) == arm.n_mu
            g = f["V_qmunu__qirr"]
            assert g["sym_perm"].shape[-1] == arm.n_mu
            assert g["L_table"].shape[1] == arm.n_mu
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_padded_write_reads_back_bit_identical_at_a_DIFFERENT_pad():
    """THE CLAIM THIS ARM EXISTS FOR: write at one device count, read at
    another, and the unfold is still the identity it is on the unpadded
    path.

    Written with a pad of +4 and read with a pad of +8 — two different
    device counts, one file.  The read-back is compared against
    ``unfold_isdf_operator`` on the +8-padded wedge, element-wise, which is
    exactly what the uncompressed path would have held in memory on the
    reading side.  If the pad were stored rather than reconstructed, this
    is the cell that would fail.
    """
    import jax
    import jax.numpy as jnp
    mesh = _mesh_here()
    arm = trs_arm("gnppm_debug")
    tables_w, X_w = _pad(arm, arm.n_mu + 4)
    tables_r, X_r = _pad(arm, arm.n_mu + 8)
    d, path = _tmp_h5("padcross")
    try:
        QS.write_qirr_tensor(path, "V_qmunu", X_w, tables=tables_w,
                             closure_verdict=arm.verdict,
                             n_rmu_logical=arm.n_mu, mode="w")
        got, hdr = QS.read_tensor(path, "V_qmunu", mesh_xy=mesh,
                                  n_mu_padded=arm.n_mu + 8)
        assert hdr.q_storage == "ibz" and hdr.was_unfolded
        assert got.shape[-1] == arm.n_mu + 8
        ref = unfold_isdf_operator(
            jnp.asarray(X_r), irr_idx=tables_r.irr_idx_q,
            sym_idx=tables_r.sym_idx_q, sym_perm=tables_r.sym_perm,
            L_table=tables_r.L_table, q_irr_frac=tables_r.q_irr_frac,
            mesh_xy=mesh, n_sym_spatial=int(tables_r.n_sym_spatial))
        d_max = _shard_max_abs_diff(got, ref)
        assert d_max == 0.0, (
            f"read-back at pad +8 differs from the unfold at pad +8 by "
            f"{d_max:.3e}; the round trip is supposed to be an IDENTITY "
            f"across device counts, which is the whole reason the pad is "
            f"reconstructed rather than stored")
        assert_offdiag_elementwise(got, ref, "cross-pad identity")
        # And the pad rows really are pad: exactly zero, on both axes.
        n = arm.n_mu
        blk = np.asarray(jax.device_get(got))
        assert not np.any(blk[:, n:, :]) and not np.any(blk[:, :, n:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_permutation_tail_that_is_not_the_identity_refuses():
    """RED TWIN 1.  A non-identity tail is not a pad — it is a table that
    addresses real centroids past the declared logical extent, and
    stripping it would discard a permutation this format then could not
    invert."""
    arm = trs_arm("gnppm_debug")
    tables, X = _pad(arm, arm.n_mu + 4)
    perm = np.array(tables.sym_perm)
    perm[0, arm.n_mu] = 0                       # tail no longer identity
    bad = QS.QirrTables(tables.irr_idx_q, tables.sym_idx_q,
                        tables.q_irr_frac, perm, tables.L_table,
                        tables.n_sym_spatial)
    d, path = _tmp_h5("badperm")
    try:
        with pytest.raises(ValueError, match="not the identity pad"):
            QS.write_qirr_tensor(path, "V_qmunu", X, tables=bad,
                                 closure_verdict=arm.verdict,
                                 n_rmu_logical=arm.n_mu, mode="w")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_nonzero_wrap_tail_refuses():
    """RED TWIN 2.  A non-zero ``L_table`` tail carries umklapp phase that
    stripping would drop — silently, and only on the q's that wrap."""
    arm = trs_arm("gnppm_debug")
    tables, X = _pad(arm, arm.n_mu + 4)
    L = np.array(tables.L_table)
    L[0, arm.n_mu, 0] = 1
    bad = QS.QirrTables(tables.irr_idx_q, tables.sym_idx_q,
                        tables.q_irr_frac, tables.sym_perm, L,
                        tables.n_sym_spatial)
    d, path = _tmp_h5("badL")
    try:
        with pytest.raises(ValueError, match=r"L_table's μ tail"):
            QS.write_qirr_tensor(path, "V_qmunu", X, tables=bad,
                                 closure_verdict=arm.verdict,
                                 n_rmu_logical=arm.n_mu, mode="w")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_tensor_whose_pad_rows_are_not_zero_refuses():
    """RED TWIN 3.  Pad rows are zeros BY CONSTRUCTION; anything else in
    them is real data, and stripping would delete it.  This is the arm
    that makes the strip safe rather than merely convenient."""
    arm = trs_arm("gnppm_debug")
    tables, X = _pad(arm, arm.n_mu + 4)
    X[0, arm.n_mu, 0] = 1e-30                   # far below any tolerance
    d, path = _tmp_h5("badtail")
    try:
        with pytest.raises(ValueError, match="NOT exactly zero"):
            QS.write_qirr_tensor(path, "V_qmunu", X, tables=tables,
                                 closure_verdict=arm.verdict,
                                 n_rmu_logical=arm.n_mu, mode="w")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_an_unpadded_write_is_byte_identical_to_before_this_arm_existed():
    """The default path did not move.  ``n_rmu_logical=None`` is what every
    caller wrote before the pad question existed, and it must still produce
    the same file — the stamped logical extent then simply equals the
    stored one."""
    mesh = _mesh_here()
    arm = trs_arm("gnppm_debug")
    d, path = _tmp_h5("nopad")
    try:
        hdr = QS.write_qirr_tensor(path, "V_qmunu", arm.X_ibz,
                                   tables=arm.tables,
                                   closure_verdict=arm.verdict, mode="w")
        assert hdr.n_rmu_logical == arm.n_mu == hdr.n_mu
        got, _ = QS.read_tensor(path, "V_qmunu", mesh_xy=mesh)
        assert _shard_max_abs_diff(got, arm.kernel(mesh)) == 0.0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unavailable_actions_round_trip_reader_padding_and_digest():
    """Canonical unavailable rows survive both reader widths without an identity substitute."""
    table = QS.QirrTables(
        irr_idx_q=np.array([0, 0]), sym_idx_q=np.array([0, 2]),
        q_irr_frac=np.array([[0.25, 0., 0.]]),
        sym_perm=np.array([[0, 1], [-1, -1], [0, 1], [-1, -1]]),
        L_table=np.zeros((4, 2, 3), dtype=np.int32), n_sym_spatial=2)
    assert QS.validate_qirr_tables(table, 1, 2) == "ibz"
    for extent in (4, 16):
        padded = table.padded(extent)
        assert np.all(padded.sym_perm[[1, 3]] == -1)
        assert QS.validate_qirr_tables(padded, 1, extent) == "ibz"
        assert padded.logical(2).digest() == table.digest()
    from dataclasses import replace
    with pytest.raises(ValueError, match="selected centroid action is unavailable"):
        QS.validate_qirr_tables(replace(table, sym_idx_q=np.array([0, 3])), 1, 2)
    corrupt = table.sym_perm.copy()
    corrupt[1, 0] = 0
    with pytest.raises(ValueError, match="bijection or wholly unavailable"):
        QS.validate_qirr_tables(replace(table, sym_perm=corrupt), 1, 2)
