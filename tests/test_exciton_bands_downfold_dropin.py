"""The three ways ``bse.exciton_bands`` failed on a downfolded bundle.

Measured 2026-08-10 (``PIPELINE_HEALTH.md``, step 5): a 936 -> 189 downfold of
a silicon 4x4x4 parent broke the exciton-band driver THREE independent ways
while ``bse.bse_jax`` read the same bundle fine.  That asymmetry is the whole
subject of this file, and it has one cause: ``bse_jax`` only READS the stored
tensors, while ``exciton_bands`` REBUILDS objects in the same ISDF basis — psi
at finite Q, and the exchange tile off the grid.  Rebuilding needs two things
that reading does not, and neither is derivable from the bundle's shape.

Each mechanism gets a cell and a RED TWIN, the convention ``test_downfold.py``
established for this initiative: a gate that has never been seen to fail is a
hope with an assert in front of it.

  1. THE CENTROID TABLE.  The htransform leg fits psi against a coordinate
     table.  A downfolded bundle carried none, and the deck's ``centroids_file``
     names the parent's — which is not in the child's directory.  Failure was a
     bare ``FileNotFoundError`` out of ``np.loadtxt``.
  2. THE CHILD EVALUATION POINTS.  The one whole-state QRCP basis is selected
     from full Bloch states, then evaluated directly at the downfold's ordered
     child centroid rows.  No parent-width projected wavefunction is formed;
     the restart's mu remains the authority on the result.
  3. ZETA.  Off-grid exchange interpolates a stored ``zeta_q.h5``, which the
     downfold did not write.  It transports as ``zeta_S = conj(T) zeta_L`` —
     the head vector's map at every G rather than only at G = 0 — and that is
     exactly the map under which V rebuilt from zeta is the congruence the
     bundle already stores.

Cells 1 and 2 import the driver (jax, and on this tree the FFI gate behind
it); cell 3 is pure numpy/jax algebra and needs neither a deck nor a bundle.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402


def _mu_pad(n):
    """The device-legal μ extent the drivers themselves pad to.

    A CELL MUST BE ABLE TO EXPRESS ITSELF ON THE MESH IT FINDS.  Measured
    2026-08-10 at four real GPUs (`owedlegs_0810/_logs/p4gates_gpu4.log`):
    two cells in this file were written at literal μ extents — 3 kept rows,
    an ``(nq, 4, 9)`` transfer — that do not divide a 2×2 mesh, so ``pjit``
    refused their output shardings BY NAME while the driver policy they gate
    was healthy.  μ_S generically does not divide the mesh (the real deck
    selects 185 at 2×2), so the answer is not a rounder literal: it is the
    pad the production path already carries.
    ``runtime.padding.padded_mu_extent`` is the single source of truth for
    it, and ``exciton_bands.build_conduction_stacks`` takes ``n_rmu_pad``
    as an argument precisely so its caller can supply it.  At one device the
    extent is unchanged, so these cells' 1×1 numbers are untouched.
    """
    from runtime.padding import padded_mu_extent
    return int(padded_mu_extent(int(n), int(jax.device_count())))


def _driver():
    """The real driver module, or a skip that says why it could not load.

    ``bse.exciton_bands`` runs ``initialize_communicator_stack()`` at import,
    which enforces the required FFI layer.  A checkout without the built
    ``.so`` cannot import ANY driver, and these cells are about driver policy
    rather than about the FFI, so they say so and stand down rather than
    reporting the environment as a downfold defect.
    """
    try:
        from bse import exciton_bands
    except RuntimeError as exc:                               # noqa: BLE001
        pytest.skip(f"driver import needs the FFI layer: {exc}")
    return exciton_bands


def _bundle(tmp_path, *, n_rmu, provenance=None):
    """A minimal restart file: only what ``read_downfold_provenance`` reads."""
    import h5py
    path = tmp_path / f"isdf_tensors_{n_rmu}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("n_rmu_logical", data=np.int64(n_rmu))
        if provenance is not None:
            g = f.create_group("downfold_provenance")
            for k, v in provenance.items():
                if k == "keep_idx":
                    g.create_dataset("keep_idx",
                                     data=np.asarray(v, dtype=np.int64))
                else:
                    g.attrs[k] = v
    return str(path)


def _table(path, n_rows, seed=0):
    rng = np.random.default_rng(seed)
    np.savetxt(path, rng.random((n_rows, 3)), fmt="%.12f")
    return str(path)


# ---------------------------------------------------------------------------
# 1.  The centroid table: which one, and where it is recorded
# ---------------------------------------------------------------------------

def test_a_plain_bundle_still_uses_the_decks_own_centroid_table(tmp_path):
    """NO PROVENANCE ⇒ the deck, and no slice.  This is the splash proof.

    Every natively-fitted bundle in existence takes this arm, so it has to be
    the program the driver ran before any of this: the deck's path, resolved
    against the input file's directory, and ``keep_idx = None`` so
    ``build_conduction_stacks`` traces its pre-existing shape.
    """
    xb = _driver()
    deck = _table(tmp_path / "centroids_frac_8.txt", 8)
    path, keep = xb.resolve_isdf_basis(
        _bundle(tmp_path, n_rmu=8), {"centroids_file": "centroids_frac_8.txt"},
        str(tmp_path / "cohsex.in"), n_rmu_bundle=8, log=lambda *a, **k: None)
    assert os.path.abspath(path) == os.path.abspath(deck)
    assert keep is None


def test_a_downfolded_bundle_takes_the_parent_table_off_its_provenance(tmp_path):
    """THE FIX FOR MECHANISM 1, and it is not the deck's file.

    The deck beside a downfolded bundle names the PARENT's table (it is the
    parent's deck, copied), and that file is not in the child's directory —
    which is the measured ``FileNotFoundError``.  The bundle records where the
    parent basis actually lives, so the driver asks the bundle.
    """
    xb = _driver()
    parent = _table(tmp_path / "centroids_frac_16_parent.txt", 16)
    b = _bundle(tmp_path, n_rmu=4, provenance={
        "parent_mu": np.int64(16), "mu_small": np.int64(4),
        "parent_file": "/nowhere/isdf_tensors_16.h5",
        "parent_centroids_file": parent,
        "keep_idx": [1, 3, 7, 11]})
    path, keep = xb.resolve_isdf_basis(
        b, {"centroids_file": "centroids_frac_16.txt"},
        str(tmp_path / "cohsex.in"), n_rmu_bundle=4, log=lambda *a, **k: None)
    assert os.path.abspath(path) == os.path.abspath(parent)
    assert list(np.asarray(keep)) == [1, 3, 7, 11]


def test_a_downfolded_bundle_with_no_resolvable_parent_table_refuses(tmp_path):
    """RED TWIN: the refusal has to name the fix, not the traceback.

    The pre-fix behaviour was ``np.loadtxt`` raising ``FileNotFoundError`` on a
    path the user never typed, with nothing to say the file was missing BECAUSE
    the bundle is a downfold.  This asserts the announce-or-refuse framing the
    rest of the tree uses: which basis is needed, why the parent's rather than
    the small one, and the two ways to supply it.
    """
    xb = _driver()
    b = _bundle(tmp_path, n_rmu=4, provenance={
        "parent_mu": np.int64(16), "mu_small": np.int64(4),
        "keep_idx": [1, 3, 7, 11]})
    with pytest.raises(SystemExit) as exc:
        xb.resolve_isdf_basis(
            b, {"centroids_file": "centroids_frac_16.txt"},
            str(tmp_path / "cohsex.in"), n_rmu_bundle=4,
            log=lambda *a, **k: None)
    msg = str(exc.value)
    assert "downfolded bundle" in msg
    assert "parent_centroids_file" in msg
    assert "coordinate authority" in msg


def test_a_deck_table_of_the_parents_size_is_accepted_over_nothing(tmp_path):
    """The deck is a FALLBACK, not an irrelevance.

    A user who has copied the parent's table beside the child's deck has
    supplied exactly the right thing, and the driver must not refuse them
    merely because the bundle's own record is stale or absent.
    """
    xb = _driver()
    deck = _table(tmp_path / "centroids_frac_16.txt", 16)
    b = _bundle(tmp_path, n_rmu=4, provenance={
        "parent_mu": np.int64(16), "mu_small": np.int64(4),
        "parent_centroids_file": str(tmp_path / "gone.txt"),
        "keep_idx": [1, 3, 7, 11]})
    path, keep = xb.resolve_isdf_basis(
        b, {"centroids_file": "centroids_frac_16.txt"},
        str(tmp_path / "cohsex.in"), n_rmu_bundle=4, log=lambda *a, **k: None)
    assert os.path.abspath(path) == os.path.abspath(deck)
    assert keep is not None


def test_a_table_of_the_wrong_size_is_not_taken_for_the_parent_basis(tmp_path):
    """RED TWIN: a same-name table of the WRONG width must be refused.

    Silently fitting in a basis that is not the one ``keep_idx`` indexes would
    attach the wrong coordinates to the right numbers — the failure this
    initiative's whole verification story is built around.
    """
    xb = _driver()
    _table(tmp_path / "centroids_frac_16.txt", 9)          # wrong width
    b = _bundle(tmp_path, n_rmu=4, provenance={
        "parent_mu": np.int64(16), "mu_small": np.int64(4),
        "keep_idx": [1, 3, 7, 11]})
    with pytest.raises(SystemExit):
        xb.resolve_isdf_basis(
            b, {"centroids_file": "centroids_frac_16.txt"},
            str(tmp_path / "cohsex.in"), n_rmu_bundle=4,
            log=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 2.  The basis the fit runs in, and who is the authority on the result
# ---------------------------------------------------------------------------

class _Bundle:
    """The two fields ``build_conduction_stacks`` reads off an htransform."""

    def __init__(self, psi, eps):
        self.psi_rmu_Y = psi
        self.enk_full = eps


def _stacks_case(mu_src, mu_out, *, nQ=2, nk=2, nc=2, ns=1):
    """The driver call, with ``n_rmu_pad`` resolved the way the driver does.

    ``mu_out`` used to be passed as BOTH the logical μ and the padded μ,
    which is only legal on a 1×1 mesh — the μ axis of the returned caches
    carries an ``x`` (and a ``y``) sharding constraint, so its extent has to
    divide the mesh.  ``main`` reads ``n_rmu_pad`` off the bundle's own
    metadata for exactly this reason; here :func:`_mu_pad` recomputes it
    from the same helper, and the pad columns are sliced back off so every
    caller below keeps asserting against the logical width it asked for.
    """
    mesh = __import__("common.collectives", fromlist=["resolve_mesh"]
                      ).resolve_mesh()
    rng = np.random.default_rng(5)
    psi = jnp.asarray(rng.normal(size=(nQ * nk, nc, ns, mu_src))
                      + 1j * rng.normal(size=(nQ * nk, nc, ns, mu_src)))
    eps = jnp.asarray(rng.normal(size=(nQ * nk, nc)))
    xb = _driver()
    out = xb.build_conduction_stacks(
        _Bundle(psi, eps), nQ, nk, nc, nc, mu_out, _mu_pad(mu_out), mesh)
    got = np.asarray(jax.device_get(out[0]))
    assert np.array_equal(got[..., mu_out:],
                          np.zeros_like(got[..., mu_out:])), (
        "the pad columns are not zero, so slicing them off is not a "
        "restriction — a pad carrying data would make every identity below "
        "depend on where the mesh happened to put the boundary")
    return np.asarray(psi), got[..., :mu_out]


def test_an_unsliced_parent_width_cache_is_refused_by_the_restarts_mu():
    """RED TWIN: forget the slice and the run must STOP, not pad its way out.

    Without the refusal, a parent-width psi against a mu_S bundle reaches
    ``jnp.pad`` with a negative width — or, at a different mu, pads silently
    and contracts conduction caches against a W in a different basis.  The
    restart's mu is the authority; everything else asserts against it.
    """
    with pytest.raises(ValueError, match="restart bundle stores mu"):
        _stacks_case(8, 3)


def test_the_unsliced_path_is_bit_identical_when_nothing_was_downfolded():
    """SPLASH PROOF: ``keep_idx=None`` traces the pre-existing program.

    A natively-fitted bundle's htransform width already equals the restart's
    mu, so the new argument must be a no-op there — not "close", identical.
    """
    psi, got = _stacks_case(8, 8)
    assert np.array_equal(got, psi.reshape(2, 2, 2, 1, 8))


# ---------------------------------------------------------------------------
# 3.  Zeta: transported, and the transport is the head vector's map
# ---------------------------------------------------------------------------

def _random_case(seed=3, nq=3, mu_s=4, mu_l=9, ngk=6):
    rng = np.random.default_rng(seed)
    T = (rng.normal(size=(nq, mu_s, mu_l))
         + 1j * rng.normal(size=(nq, mu_s, mu_l)))
    zL = (rng.normal(size=(nq, mu_l, ngk))
          + 1j * rng.normal(size=(nq, mu_l, ngk)))
    v = rng.random(ngk) + 0.1
    return T, zL, v


def _V_from_zeta(z, v):
    """``V[mu,nu] = sum_G conj(z_mu(G)) v(G) z_nu(G)`` — the definition."""
    return np.einsum("mg,g,ng->mn", np.conj(z), v, z)


def test_transported_zeta_reproduces_the_congruence_the_bundle_stores():
    """THE FIX FOR MECHANISM 3: ``zeta_S = conj(T) zeta_L`` and nothing else.

    The downfold already congruences the stored tile, ``V_S = T V_L T-dagger``.
    A transported zeta is only worth writing if V REBUILT from it is that same
    matrix — otherwise the small bundle holds two mutually inconsistent
    descriptions of one interaction and every shape check passes.  Substituting
    ``conj(T) zeta_L`` into the definition gives the congruence exactly, which
    is what this asserts at machine precision.
    """
    T, zL, v = _random_case()
    for q in range(T.shape[0]):
        zS = np.conj(T[q]) @ zL[q]
        want = T[q] @ _V_from_zeta(zL[q], v) @ T[q].conj().T
        got = _V_from_zeta(zS, v)
        assert np.max(np.abs(got - want)) < 1e-10 * np.max(np.abs(want))


def test_zeta_without_the_conjugate_breaks_the_congruence():
    """RED TWIN: ``T zeta_L`` gives the right SHAPE and the wrong matrix.

    Exactly the failure mode ``transform_head_vector``'s own twin records for
    ``g0`` (2.347 eV -> 0.211 eV on si_bse_debug, every gate green).  zeta is
    that vector at every G, so it inherits the trap at full strength and needs
    its own red twin rather than borrowing g0's.
    """
    T, zL, v = _random_case()
    q = 0
    zS_wrong = T[q] @ zL[q]
    want = T[q] @ _V_from_zeta(zL[q], v) @ T[q].conj().T
    got = _V_from_zeta(zS_wrong, v)
    assert np.max(np.abs(got - want)) > 0.1 * np.max(np.abs(want))


def test_transported_zeta_at_G0_is_the_transported_head_vector():
    """The two transports have to be ONE transport, and this ties them.

    ``G0_mu_nu`` is zeta's G = 0 column and rides
    ``gw.downfold.transform_head_vector``; the zeta writer applies the same map
    at every G.  If the two ever drift apart the bundle's head and its
    off-grid exchange describe different bases — so the writer cross-checks
    them on every run, and this is that check with the deck taken out.

    SHAPED ON THE MESH, not on 1×1 — see :func:`_mu_pad`.  ``_random_case``
    hands back a ``(3, 4, 9)`` transfer, and 9 divides no square mesh; the
    μ_L axis is the one ``transform_head_vector`` shards on ``x``, so the
    literal shape refused at 2×2.  T and the ζ column are carried here on
    the device-legal μ extents with exactly-zero pads — the same operand
    ``run_downfold`` builds — and the logical block is sliced back off.
    """
    from common.collectives import resolve_mesh
    from gw import downfold
    T, zL, _v = _random_case()
    mesh = resolve_mesh()
    nq, mu_s, mu_l = T.shape
    mu_s_pad, mu_l_pad = _mu_pad(mu_s), _mu_pad(mu_l)
    T_pad = np.zeros((nq, mu_s_pad, mu_l_pad), dtype=np.complex128)
    T_pad[:, :mu_s, :mu_l] = T
    g0_pad = np.zeros(mu_l_pad, dtype=np.complex128)
    g0_pad[:mu_l] = zL[0][:, 0]
    T_x = jax.lax.with_sharding_constraint(
        jnp.asarray(T_pad), NamedSharding(mesh, P(None, None, "x")))
    g0_L = jnp.asarray(g0_pad)
    g0_S = np.asarray(jax.device_get(
        downfold.transform_head_vector(g0_L, T_x, 0, mesh)))[:mu_s]
    from_zeta = (np.conj(T[0]) @ zL[0])[:, 0]
    assert np.max(np.abs(from_zeta - g0_S)) < 1e-12 * np.max(np.abs(g0_S))


def test_zeta_q_labels_are_matched_against_the_bundles_grid_not_assumed():
    """The transfer is indexed by the restart's q axis; zeta's is its own.

    ``T[q]`` must meet the zeta slot that carries the SAME momentum.  A
    permuted pairing writes every q's zeta against another q's transfer, which
    is a bundle full of plausible numbers, so the labels are matched when they
    are available and the fallback says out loud that it is one.
    """
    from gw import downfold_run
    kgrid = (2, 2, 1)
    idx = np.stack(np.meshgrid(np.arange(2), np.arange(2), np.arange(1),
                               indexing="ij"), axis=-1).reshape(-1, 3)
    order = [3, 0, 2, 1]

    class _ZL:
        kpoints = (idx[order] / np.asarray(kgrid, dtype=float))

    perm = downfold_run._zeta_q_to_restart_q(_ZL(), kgrid, 4,
                                             print_fn=lambda *a, **k: None)
    assert list(perm) == order


def test_a_short_q_label_list_falls_back_to_the_identity_and_says_so():
    """RED TWIN of the branch above: the reconstruction arm must be VISIBLE.

    A full-BZ zeta written from a symmetry-reduced WFN carries an IBZ-length
    ``rk``, and the writer's order is then the wrapped C-order grid — the same
    assumption ``vq_interp`` makes on the read side.  Taking it silently is
    how a q-axis assumption becomes folklore, so the branch announces itself.
    """
    from gw import downfold_run
    said = []

    class _ZL:
        kpoints = np.zeros((1, 3))

    perm = downfold_run._zeta_q_to_restart_q(_ZL(), (2, 2, 1), 4,
                                             print_fn=lambda *a, **k: said.append(a))
    assert list(perm) == [0, 1, 2, 3]
    assert said, "the reconstruction branch was taken silently"


# ---------------------------------------------------------------------------
# 1b.  the stamp that says WHICH points — in the one algebra that reads it
# ---------------------------------------------------------------------------

def test_the_centroid_stamp_is_the_fft_index_hash_not_the_text_files():
    """Found while teaching the driver to VERIFY a parent table with this hash.

    ``centroids_charge_md5`` is defined by ``gw_init._centroid_table_md5`` as
    md5 over the int64 FFT-GRID INDEX table, and ``gw_init`` compares against
    it in exactly that algebra.  The downfold used to stamp ``md5(bytes of the
    .txt)`` — a perfectly good hash of a DIFFERENT object, which could
    therefore never match, so a downfolded bundle handed to a fresh GW run was
    told its own centroid table was some other table.  Both sides now use one
    spelling, and this pins the two halves of it: the fractional-to-index
    arithmetic (including the wrap of an index that rounds up onto the grid
    extent, which ``load_centroids`` does and a naive ``round`` does not) and
    the byte layout the hash is taken over.
    """
    import hashlib
    from gw import downfold_run
    from file_io.centroids import load_centroids

    grid = (4, 6, 8)
    frac = np.array([[0.0, 0.0, 0.0],
                     [0.25, 0.5, 0.125],
                     [0.999999, 0.1, 0.9]])          # last row rounds ONTO 4
    idx = downfold_run._frac_to_fft_idx(frac, grid)
    assert idx[2, 0] == 0, "an index that rounds up onto nx must wrap to 0"

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        np.savetxt(fh.name, frac, fmt="%.12f")
        _f, from_loader, _n = load_centroids(fh.name, grid)
    assert np.array_equal(idx, from_loader), (
        "the downfold's index arithmetic has drifted from the loader's, so "
        "its stamp describes points no consumer will derive")

    want = hashlib.md5(np.ascontiguousarray(
        idx.astype(np.int64)).tobytes()).hexdigest()
    assert downfold_run._centroid_table_md5(idx) == want
    # RED TWIN: the hash of the text file is a different number, which is the
    # whole defect — it looks like a content hash and matches nothing.
    with open(fh.name, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() != want
    os.unlink(fh.name)


# ---------------------------------------------------------------------------
# 3b.  and the refusal when there is no zeta to interpolate
# ---------------------------------------------------------------------------

def test_ongrid_needs_no_zeta_and_does_not_look_for_one(tmp_path):
    """``--vq-mode=ongrid`` is exact on-grid and reads no zeta at all.

    This is the mode that WORKS on a bundle whose lineage has none, so the
    refusal must never fire for it — otherwise the fix would take away the one
    route that was available.
    """
    xb = _driver()
    assert xb.require_zeta_for_interp(
        _bundle(tmp_path, n_rmu=4), "ongrid", (4, 4, 4)) == ""


def test_interp_without_a_zeta_refuses_and_names_the_downfold(tmp_path):
    """RED TWIN for mechanism 3's reader side: framing, not a bare OSError.

    The message has to carry three things the h5py error cannot: that this
    bundle is a downfold, that the downfold DOES transport zeta so an absent
    one means an IBZ-only parent, and that ongrid needs none.
    """
    xb = _driver()
    b = _bundle(tmp_path, n_rmu=4, provenance={
        "parent_mu": np.int64(16), "keep_idx": [1, 3, 7, 11]})
    with pytest.raises(SystemExit) as exc:
        xb.require_zeta_for_interp(b, "interp", (4, 4, 4))
    msg = str(exc.value)
    assert "DOWNFOLD" in msg and "ongrid" in msg and "IBZ" in msg
