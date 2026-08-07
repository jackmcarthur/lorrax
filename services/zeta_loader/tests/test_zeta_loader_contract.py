"""Layer L-a: the door's contract, on synthetic files, in seconds, on WSL.

Everything here is single-device or ``mesh=None``.  Nothing here needs the
phdf5 FFI, and that is a property of the SURFACE rather than an accident of
the fixtures: the probe and ``write_g0_mu`` are pure h5py, the refusals all
fire before any transport is reached, and ``read_zeta_G_local`` is serial
h5py BY CONTRACT (the local plan exists so a rank-0 diagnostic is not a
collective).  The one thing that does need a transport —
``read_zeta_G_slab`` — appears here only through its REFUSALS; its numbers
come from the L-b/L-c tiers in ``test_zeta_loader_multiproc.py``.

WHAT THIS FILE PINS THAT NOTHING PINNED BEFORE

* **The header surface.**  Survey §1.1: the ~40 dynamically-bound
  ``bind_mf_attrs`` / ``bind_isdf_attrs`` names are "the loader's LARGEST
  consumed surface and they are not enumerated anywhere in the class — a
  service extraction must pin them explicitly or the door is undefined."
  :func:`test_the_header_surface_production_consumes_is_pinned` is that
  enumeration.
* **The D5 agreement** (header ``ngkmax`` vs the ``zeta_q_G`` G axis), whose
  red twin needs a file no in-tree builder can produce — see
  ``zeta_synth``'s docstring.
* **The probe's never-raises contract**, fed the inputs that would break a
  narrower ``except``.
* **``write_g0_mu``'s logical-extent guard**, which is what turns "the
  caller clipped the μ axis correctly" from a convention into a check.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:                          # bare-checkout / CLI runs
    sys.path.insert(0, _TESTS)

import zeta_synth as Z                                         # noqa: E402
from zeta_loader import (                                      # noqa: E402
    ZetaFileProbe, ZetaLoader, probe_zeta_file, write_g0_mu)


def _needs_host_tree():
    ok, why = Z.host_tree_state()
    if not ok:
        pytest.skip(why)


# ===========================================================================
# 0. The fixtures themselves — anti-tautology
# ===========================================================================

def test_the_local_pad_sentinel_agrees_with_the_shared_one():
    """``zeta_synth.pad_sentinel`` is a COPY, so it is checked against the
    original.

    ``common.gvec_fft_box.fft_box_pad_sentinel`` is the one definition; this
    suite transcribes it so the fixtures can be built with no host tree (the
    isolation subprocess needs exactly that).  A copy nobody compares is a
    drift waiting to happen, and the drift would be invisible: the pad rows
    would agree with THEMSELVES and every ``gvecs()`` cell would still pass
    while measuring a grid the production reader never sees.  Odd extents
    are included because that is where the two plausible spellings
    (``fftfreq`` vs ``-n//2``) actually differ.
    """
    _needs_host_tree()
    from common.gvec_fft_box import fft_box_pad_sentinel
    for grid in ((8, 8, 8), (16, 16, 16), (24, 24, 80), (6, 6, 10),
                 (15, 15, 61), (5, 5, 5), (1, 2, 3)):
        want, _flat = fft_box_pad_sentinel(grid)
        assert Z.pad_sentinel(grid) == tuple(int(v) for v in want), grid


def test_the_synthetic_header_matches_the_real_writers(tmp_path):
    """The hand-written groups read back through BOTH production readers.

    ``zeta_synth`` writes ``mf_header`` and ``isdf_header`` with raw h5py so
    it can build states ``IsdfHeader.build`` validates away.  The cost of
    that is a second copy of the on-disk layout, and this is the cell that
    keeps the copy honest: if ``file_io.mf_header`` or
    ``file_io.isdf_header`` ever reads a dataset this builder does not
    write, it fails HERE — loudly, once — instead of turning every other
    cell in this suite into a test of a file format nothing produces.
    """
    _needs_host_tree()
    from file_io.isdf_header import read_isdf_header
    from file_io.mf_header import read_mf_header

    path, _payload = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3,
                                   ngkmax=4, fft_grid=(8, 8, 8),
                                   kgrid=(2, 2, 2))
    mf = read_mf_header(path)
    assert [int(v) for v in mf.fft_grid] == [8, 8, 8]
    assert [int(v) for v in mf.kgrid] == [2, 2, 2]
    assert int(mf.nspin) == 1 and int(mf.ntran) == 2

    isdf = read_isdf_header(path)
    assert isdf.zeta_layout == "G_flat"
    assert isdf.zeta_is_done is True
    assert isdf.n_rmu == 3
    assert isdf.ngkmax == 4
    assert isdf.ngk_per_q is not None and isdf.zeta_cutoff_ry == 10.0


# ===========================================================================
# 1. probe_zeta_file — the truth table, and the never-raises CONTRACT
# ===========================================================================

def test_probe_reports_a_path_that_does_not_exist(tmp_path):
    p = probe_zeta_file(tmp_path / "nope.h5")
    assert p.exists is False and p.readable is False
    # NOT an error: a missing file is a legitimate answer to "what is on
    # disk here", and both call sites' next move is "fit it".  Reporting it
    # as an error would make the normal first-run path look like a fault.
    assert p.error is None
    assert (p.dataset_name, p.mu_extent, p.zeta_done, p.r_mu_fft_idx) == \
        (None, None, None, None)


def test_probe_reports_a_zero_byte_file(tmp_path):
    """The crashed-job state, which is what motivated the never-raises rule."""
    p0 = tmp_path / "empty.h5"
    p0.write_bytes(b"")
    p = probe_zeta_file(p0)
    assert p.exists is True and p.readable is False
    assert p.error and ":" in p.error          # "<ExceptionType>: <message>"
    assert p.dataset_name is None and p.zeta_done is None


def test_probe_reports_a_directory_and_a_non_hdf5_file(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    pd = probe_zeta_file(d)
    assert pd.exists is True and pd.readable is False and pd.error

    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"this is not HDF5, it is a sentence." * 40)
    pj = probe_zeta_file(junk)
    assert pj.exists is True and pj.readable is False and pj.error


@pytest.mark.parametrize("bad", [
    None, 123, 4.5, object(), b"\x00\x01", ["a", "list"], {"a": "dict"},
    "", "/", "/proc/self/mem", "\x00embedded-nul",
])
def test_probe_never_raises_for_any_input(bad):
    """**NEVER RAISES** is a CONTRACT, not a convenience.

    Both production call sites (``gw_init._check_zeta_h5_matches_basis`` and
    ``_zeta_reuse_ok``'s extent probe) run BEFORE a loader could exist and
    must print-and-continue, because the thing they are about to do is
    OVERWRITE the file.  A probe that raised would turn "the stale ζ here is
    garbage, refitting" into a traceback at startup.

    The inputs are chosen to break a narrower ``except``: ``None`` and the
    non-path objects fail in ``os.fspath`` (TypeError, before any h5py);
    ``"\\x00embedded-nul"`` fails inside ``os.path.exists`` (ValueError on
    Linux); ``/proc/self/mem`` exists and is not readable as HDF5 (OSError
    from the native layer, which has no Python class of its own).
    """
    p = probe_zeta_file(bad)
    assert isinstance(p, ZetaFileProbe)
    assert isinstance(p.exists, bool) and isinstance(p.readable, bool)
    assert p.readable is False


def test_probe_reads_a_valid_g_flat_file(tmp_path):
    path, _payload = Z.build_gflat(tmp_path / "z.h5", n_q=3, n_rmu=5,
                                   ngkmax=4)
    p = probe_zeta_file(path)
    assert p.exists is True and p.readable is True and p.error is None
    assert p.dataset_name == "zeta_q_G"
    assert p.mu_extent == 5                    # axis 1 for G-flat
    assert p.zeta_done is True
    assert p.r_mu_fft_idx is not None
    assert p.r_mu_fft_idx.shape == (5, 3) and p.r_mu_fft_idx.dtype == np.int64


def test_probe_reads_a_legacy_r_space_file(tmp_path):
    """The dispatch is ``(('zeta_q_G', 1), ('zeta_q', 2))`` — μ moves axis.

    This is THE copy of that truth (survey §2.3 V3): the two hand-written
    copies in ``gw_init`` probed ``f['zeta_q']`` ONLY for months after the
    G-flat migration, so the guard silently passed on exactly the production
    files it was written to protect.  Both rows are exercised here, and the
    axis is what distinguishes them.
    """
    path, _payload = Z.build_rspace(tmp_path / "z.h5", n_q=2, n_rtot=8,
                                    n_rmu=4)
    p = probe_zeta_file(path)
    assert p.readable is True
    assert p.dataset_name == "zeta_q"
    assert p.mu_extent == 4                    # axis 2 for r-space
    assert p.zeta_done is True


def test_probe_reports_a_header_with_no_zeta_dataset(tmp_path):
    """Killed between the header write and the first chunk.  A real state."""
    path = Z.build_isdf_header_only(tmp_path / "z.h5", n_rmu=7,
                                    zeta_is_done=False)
    p = probe_zeta_file(path)
    assert p.readable is True and p.error is None
    assert p.dataset_name is None and p.mu_extent is None
    # The header's opinion survives even with no dataset to compare it to.
    assert p.zeta_done is False
    assert p.r_mu_fft_idx is not None and p.r_mu_fft_idx.shape[0] == 7


@pytest.mark.parametrize("done,omit,want", [(True, False, True),
                                            (False, False, False),
                                            (True, True, None)])
def test_probe_zeta_done_is_three_valued(tmp_path, done, omit, want):
    """``None`` is NOT ``False`` — the record's docstring says so, in bold.

    A missing flag means "this file predates the flag" (legacy, treated as
    complete by ``isdf_header._read_group``); ``False`` means "a writer
    stamped this and never came back".  The test a caller wants is
    ``probe.zeta_done is False``, and collapsing the two would make every
    legacy ζ look like a torn one.
    """
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4,
                             zeta_is_done=done, omit_zeta_is_done=omit)
    assert probe_zeta_file(path).zeta_done is want


def test_probe_r_mu_fft_idx_is_absent_when_the_group_is(tmp_path):
    """No ``isdf_header`` at all: readable, but nothing ζ-shaped to report."""
    path = str(tmp_path / "z.h5")
    Z.write_mf_header(path)                    # mf_header ONLY
    p = probe_zeta_file(path)
    assert p.readable is True and p.error is None
    assert p.zeta_done is None and p.r_mu_fft_idx is None
    assert p.dataset_name is None


def test_the_probe_record_has_no_equality_operator(tmp_path):
    """``eq=False``, and the reason is that the generated one would RAISE.

    :attr:`r_mu_fft_idx` is an array; a dataclass-generated ``__eq__`` would
    compare it elementwise and then raise on the truth value of the result.
    A record whose equality operator raises is worse than one that has none,
    so the field is pinned rather than left to a future "why is this
    eq=False?" cleanup.
    """
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    a, b = probe_zeta_file(path), probe_zeta_file(path)
    assert a == a                              # identity
    assert a != b                              # two reads of one file, not ==
    assert not ZetaFileProbe.__dataclass_params__.eq


# ===========================================================================
# 2. write_g0_mu — the one sanctioned post-close serial append
# ===========================================================================

def test_write_g0_mu_round_trips(tmp_path):
    import h5py as h5
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=3, n_rmu=5, ngkmax=4)
    g0 = (np.arange(15, dtype=np.float64).reshape(3, 5)
          + 1j * np.arange(15, dtype=np.float64).reshape(3, 5) * 0.25)
    shape = write_g0_mu(path, g0, n_rmu_expected=5)
    assert shape == (3, 5)
    with h5.File(path, "r") as f:
        np.testing.assert_array_equal(np.asarray(f["g0_mu"]), g0)


def test_write_g0_mu_deletes_and_recreates_at_a_new_shape(tmp_path):
    """A re-run at a different centroid count must not hit a stale dataset.

    ``del f['g0_mu']`` then ``create_dataset`` — NOT ``f['g0_mu'][...] =``,
    which is what a shape-preserving write would be and which would refuse
    the moment the centroid count moved.  That is the behaviour the raw
    four lines in ``gw_init`` had; the door keeps it and says why.
    """
    import h5py as h5
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=3, n_rmu=5, ngkmax=4)
    write_g0_mu(path, np.ones((3, 5), dtype=np.complex128), n_rmu_expected=5)
    second = np.full((2, 9), 7.0 + 1j, dtype=np.complex128)
    assert write_g0_mu(path, second, n_rmu_expected=9) == (2, 9)
    with h5.File(path, "r") as f:
        assert f["g0_mu"].shape == (2, 9)
        np.testing.assert_array_equal(np.asarray(f["g0_mu"]), second)


def test_write_g0_mu_refuses_a_padded_array(tmp_path):
    """RED TWIN of the logical-extent rule.

    ``g0_mu`` must be written at the LOGICAL centroid count.  A caller that
    hands over the mesh-PADDED array puts exact-zero rows on disk that every
    later reader takes for real ζ̃(q, G=0) values — a silent wrong number,
    not a crash.  ``n_rmu_expected`` is the guard, and this is the case
    where it returns FALSE: the same call that passes above, with the pad
    left on.
    """
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=3, n_rmu=5, ngkmax=4)
    padded = np.zeros((3, 8), dtype=np.complex128)      # 5 logical in 8
    padded[:, :5] = 1.0
    with pytest.raises(ValueError) as ei:
        write_g0_mu(path, padded, n_rmu_expected=5)
    msg = str(ei.value)
    assert "LOGICAL centroid count" in msg
    assert "g0[..., :n_rmu]" in msg            # the FIX is in the message
    assert "8" in msg and "5" in msg           # both numbers named
    # …and the file is untouched: a refusal must not half-write.
    import h5py as h5
    with h5.File(path, "r") as f:
        assert "g0_mu" not in f


def test_write_g0_mu_refuses_a_scalar(tmp_path):
    """A 0-d array has no μ axis for ``n_rmu_expected`` to check against.

    Reported as its own refusal rather than as an IndexError out of
    ``arr.shape[-1]``, because the two have different fixes.
    """
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    with pytest.raises(ValueError, match="scalar"):
        write_g0_mu(path, 3.0, n_rmu_expected=5)


def test_write_g0_mu_without_the_guard_writes_whatever_it_is_given(tmp_path):
    """The guard is OPTIONAL, and this is what optional costs.

    Stated as a test rather than left implicit: with ``n_rmu_expected``
    omitted the padded array above lands on disk unchallenged.  That is the
    argument for every call site passing it, and it is the behaviour a
    future reader of this function needs to know about.
    """
    import h5py as h5
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    padded = np.zeros((3, 8), dtype=np.complex128)
    assert write_g0_mu(path, padded) == (3, 8)
    with h5.File(path, "r") as f:
        assert f["g0_mu"].shape == (3, 8)


# ===========================================================================
# 3. ZetaLoader refusals at open
# ===========================================================================

def test_open_refuses_an_unfinished_zeta(tmp_path):
    """``zeta_is_done=False`` — the read-side provenance gate.

    Until this landed NOTHING read the flag, so a ζ left behind by a job
    that died mid-write was indistinguishable from a complete one and its
    undefined trailing q-blocks flowed straight into V_q → W → Σ with rc=0.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", zeta_is_done=False)
    with pytest.raises(ValueError) as ei:
        ZetaLoader(path)
    msg = str(ei.value)
    assert "zeta_is_done=False" in msg
    assert "LORRAX_ALLOW_PARTIAL_ZETA" in msg          # the override, named
    assert "restart=false" in msg                      # the fix, named


def test_the_partial_zeta_override_actually_overrides(monkeypatch, tmp_path):
    """The other half: the debugging escape hatch opens the same file."""
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", zeta_is_done=False)
    monkeypatch.setenv("LORRAX_ALLOW_PARTIAL_ZETA", "1")
    with ZetaLoader(path) as zl:
        assert zl.zeta_is_done is False                # reported, not hidden
        assert zl.n_q_on_disk == 2


def test_open_refuses_a_g_flat_file_whose_header_lists_more_mu(tmp_path):
    """Header μ vs dataset μ, G-flat: the header count is a FLOOR.

    In G-flat layout μ is the on-disk axis padded to the mesh, so a dataset
    with MORE rows than the header lists is normal.  Fewer is corrupt.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4,
                             header_n_rmu=9)
    with pytest.raises(ValueError, match="9 centroids but zeta_q_G has only"):
        ZetaLoader(path)


def test_a_g_flat_dataset_wider_than_the_header_is_accepted(tmp_path):
    """RED TWIN of the floor: the direction that must NOT refuse.

    Without this, an equality check would look identical to the floor check
    on every test that exists and would refuse every mesh-padded production
    ζ on the first 4-rank run.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4,
                             dataset_n_rmu=8)
    with ZetaLoader(path) as zl:
        assert zl.n_rmu == 3 and zl.n_rmu_disk == 8


def test_open_refuses_an_r_space_file_whose_header_mu_disagrees(tmp_path):
    """The r-space arm of the same agreement — here it IS an equality."""
    _needs_host_tree()
    path, _p = Z.build_rspace(tmp_path / "z.h5", n_q=2, n_rtot=8, n_rmu=4,
                              header_n_rmu=6)
    with pytest.raises(ValueError, match="6 centroids but zeta_q has 4"):
        ZetaLoader(path)


def test_open_refuses_when_header_ngkmax_and_the_dataset_G_axis_disagree(
        tmp_path):
    """THE D5 REFUSAL, and the whole reason it exists.

    This loader offers TWO plans over one file and they size their G axis
    from DIFFERENT places: the collective plan (``_read_g_flat_disk``) takes
    ``ngkmax`` from the HEADER (``gvec_components.shape[-1]``), the local
    plan (``read_zeta_G_local``) takes it from the DATASET's own G axis.
    The writer sets both from one value, so they agree on any file it
    produced — and if they ever did not, the two plans would silently read
    DIFFERENT EXTENTS of the same file and nothing would say so.  The check
    moved out of ``_ZetaGTiles.__init__`` (where only the consumer that
    happened to build tiles got it) into ``__init__``, where every caller
    does.

    This is the red twin of the agreement ``check_local_vs_collective_
    identity`` (leg L-c) relies on: that check would report byte-identity
    from a file where the two plans read the same extent by luck, and this
    one is why luck is not required.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4,
                             header_ngkmax=6)
    with pytest.raises(ValueError) as ei:
        ZetaLoader(path)
    msg = str(ei.value)
    assert "ngkmax=6" in msg                   # the header's number
    assert "G axis is 4" in msg                # the dataset's number
    assert "collective plan" in msg and "local plan" in msg
    assert "Refit" in msg


def test_a_matching_ngkmax_opens(tmp_path):
    """RED TWIN of D5: the agreeing file must open, or the check is a wall."""
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4)
    with ZetaLoader(path) as zl:
        assert int(zl.ngkmax_zeta) == int(zl.n_G_sph_disk) == 4


# ===========================================================================
# 4. The r-space DATA surface is gone; the HEADER surface is not
# ===========================================================================

def test_the_header_surface_still_opens_r_space_files(tmp_path):
    """``__init__`` keeps opening them, deliberately.

    Several callers legitimately want only the crystal block / the centroid
    table, and the header surface is layout-independent.  It is the DATA
    methods that refuse.
    """
    _needs_host_tree()
    path, _p = Z.build_rspace(tmp_path / "z.h5", n_q=2, n_rtot=8, n_rmu=4)
    with ZetaLoader(path) as zl:
        assert zl.zeta_layout == "r_space"
        assert zl.n_q_on_disk == 2
        assert zl.n_rtot_disk == 8 and zl.n_rmu_disk == 4
        assert zl.n_G_sph_disk is None
        assert int(zl.n_rtot) == 8 * 8 * 8     # from fft_grid, not the disk


@pytest.mark.parametrize("call", [
    ("read_zeta_G_slab", dict(q_offset=0, q_count=1, mu_offset=0, mu_count=1)),
    ("read_zeta_G_local", None),
    ("load", dict()),
])
def test_every_data_method_refuses_an_r_space_file_by_name(tmp_path, call):
    """The 2026-08-07 removal, named by each of the three data methods.

    A refusal that says only "unsupported layout" sends the reader to guess;
    these say WHAT was removed, WHY (``fit_zeta_to_h5`` hardcodes
    ``zeta_layout='G_flat'``, so nothing has written r-space since the
    migration) and WHAT TO DO (refit).
    """
    _needs_host_tree()
    name, kwargs = call
    path, _p = Z.build_rspace(tmp_path / "z.h5", n_q=2, n_rtot=8, n_rmu=4)
    with ZetaLoader(path) as zl:
        fn = getattr(zl, name)
        with pytest.raises(ValueError) as ei:
            fn(0) if kwargs is None else fn(**kwargs)
        msg = str(ei.value)
        assert "zeta_layout='r_space'" in msg
        assert name in msg
        assert "2026-08-07" in msg
        assert "G-flat writer" in msg
        assert "HEADER surface of this loader still" in msg


def test_read_zeta_G_local_refuses_after_close(tmp_path):
    """The serial handle went with ``close()``; saying so beats an h5py
    ``ValueError: Not a dataset`` out of a closed file object."""
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    zl = ZetaLoader(path)
    assert zl.read_zeta_G_local(0).shape == (3, 4)
    zl.close()
    with pytest.raises(RuntimeError, match="is closed"):
        zl.read_zeta_G_local(0)
    zl.close()                                 # idempotent


# ===========================================================================
# 5. Header-only mode (mesh=None)
# ===========================================================================

def test_header_only_refuses_the_collective_reads_naming_the_missing_mesh(
        tmp_path):
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    with ZetaLoader(path) as zl:               # mesh=None by default
        for fn, kw in ((zl.read_zeta_G_slab,
                        dict(q_offset=0, q_count=1, mu_offset=0, mu_count=1)),
                       (zl.load, dict())):
            with pytest.raises(RuntimeError) as ei:
                fn(**kw)
            msg = str(ei.value)
            assert "HEADER-ONLY" in msg and "mesh=None" in msg
            assert "mesh_xy" in msg            # the fix, named
        with pytest.raises(RuntimeError, match="HEADER-ONLY"):
            _ = zl.slab_io


def test_header_only_read_zeta_G_local_works_and_returns_the_payload(tmp_path):
    """THE SERIAL PLAN NEEDS NO TRANSPORT, and that is load-bearing.

    ``mesh=None`` means no SlabIO handle was ever opened, so on a stack with
    no phdf5 FFI — this WSL box, and the production container's rank-0
    diagnostics — the local plan is the ONLY ζ read there is.  Asserting the
    exact payload rather than merely "it did not raise" is what makes this a
    read test instead of a smoke test.
    """
    _needs_host_tree()
    path, payload = Z.build_gflat(tmp_path / "z.h5", n_q=3, n_rmu=5, ngkmax=4)
    with ZetaLoader(path) as zl:
        assert zl._header_only is True
        got = zl.read_zeta_G_local(slice(None))
        np.testing.assert_array_equal(got, payload)
        assert got.dtype == np.complex128


@pytest.mark.parametrize("key", [
    0, 2, slice(1, 3), slice(None),
    (0, slice(0, 2)), (slice(0, 2), slice(1, 4), slice(0, 3)),
    (1, 3, 2), (slice(None), 0), np.s_[..., 0],
])
def test_read_zeta_G_local_is_byte_identical_to_raw_h5py(tmp_path, key):
    """``read_zeta_G_local(key)`` returns EXACTLY ``dataset[key]``.

    That is the contract V4 was absorbed under: ``_ZetaGTiles.__getitem__``
    in ``bse/vq_interp.py`` used to hold its own ``h5py.File`` and index it
    directly, and the door's job is to own the handle and the lifecycle
    WITHOUT changing what an index means.  Byte identity against the raw
    handle over ints, slices, tuples and an ellipsis is what makes
    "delegates" checkable rather than asserted.
    """
    _needs_host_tree()
    import h5py as h5
    path, _payload = Z.build_gflat(tmp_path / "z.h5", n_q=4, n_rmu=6,
                                   ngkmax=5)
    with h5.File(path, "r") as f:
        want = f["zeta_q_G"][key]
    with ZetaLoader(path) as zl:
        got = zl.read_zeta_G_local(key)
    assert np.asarray(got).shape == np.asarray(want).shape
    assert np.asarray(got).dtype == np.asarray(want).dtype
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


# ===========================================================================
# 6. THE HEADER SURFACE PIN  (survey §1.1: "not enumerated anywhere")
# ===========================================================================

#: The attribute names PRODUCTION consumes off a ``ZetaLoader``, enumerated.
#: Sourced from survey §1.1 (the binder inventory) and §1.3 (the call-site
#: table): ``v_q_g_flat.py:315,339``, ``v_q_bispinor.py:160``,
#: ``vq_interp.py:206,213,223,340-353`` (the widest header consumer in the
#: tree, 11 attributes in one call), ``zeta_projection.py:255-315``,
#: ``gw_init.py:1200``.
#:
#: THIS TUPLE IS THE DOOR'S HEADER SURFACE.  Before the extraction it existed
#: nowhere: the names are bound DYNAMICALLY by ``bind_mf_attrs`` /
#: ``bind_isdf_attrs`` from a NamedTuple's ``_fields`` and a hand-written
#: binder body, so grep was the only inventory and a binder that stopped
#: binding one of them would have failed at a call site, in production,
#: rather than here.
_BOUND_MF = (
    "nspin", "nspinor", "nkpts", "nbands", "kgrid", "fft_grid", "kpoints",
    "ifmax", "ifmin", "ngk", "ntran", "sym_matrices", "translations",
    "bvec", "avec", "adot", "bdot", "blat", "alat", "cell_volume",
    "recip_volume", "nat", "atom_types", "atom_positions",
)
_BOUND_ISDF = (
    "density", "vertex_mu_L", "r_mu_fft_idx", "r_mu_crystal", "n_rmu",
    "zeta_is_done", "zeta_layout", "gvec_components", "ngk_per_q",
    "ngkmax_zeta", "zeta_cutoff_ry", "fit_provenance",
)
#: Derived in ``__init__`` (or a property), not bound: these are the loader's
#: OWN statements about the file and are consumed by ``vq_interp.py:213``
#: and ``zeta_projection.py:255``.
_DERIVED = ("n_q_on_disk", "n_rtot_disk", "n_rmu_disk", "n_G_sph_disk",
            "n_q_full", "q_layout", "n_rtot")

HEADER_SURFACE = _BOUND_MF + _BOUND_ISDF + _DERIVED


def test_the_header_surface_production_consumes_is_pinned(tmp_path):
    """Every name in :data:`HEADER_SURFACE`, present on a real open loader.

    Survey §1.1: the bound attributes "are the loader's LARGEST consumed
    surface and they are not enumerated anywhere in the class — a service
    extraction must pin them explicitly or the door is undefined."  This is
    that enumeration, and it is a PIN in both directions: a binder that drops
    a name fails here, and a caller that starts depending on a name outside
    this tuple has to add it here first.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=3, ngkmax=4)
    with ZetaLoader(path) as zl:
        missing = [n for n in HEADER_SURFACE if not hasattr(zl, n)]
        assert missing == [], f"the door lost these header attributes: {missing}"
        # Types, not just presence: a name that is there but ``None`` on a
        # G-flat file is the failure mode a hasattr check cannot see.
        assert int(zl.nspin) == 1
        assert [int(v) for v in zl.fft_grid] == [8, 8, 8]
        assert [int(v) for v in zl.kgrid] == [2, 2, 2]
        assert np.asarray(zl.sym_matrices).shape == (48, 3, 3)
        assert int(zl.ntran) == 2
        assert np.asarray(zl.bvec).shape == (3, 3)
        assert np.asarray(zl.adot).shape == (3, 3)
        assert float(zl.blat) > 0 and float(zl.cell_volume) > 0
        assert np.asarray(zl.ifmax).shape == (1, 3)
        assert np.asarray(zl.kpoints).shape == (3, 3)
        assert int(zl.vertex_mu_L) == 0
        assert np.asarray(zl.r_mu_fft_idx).shape == (3, 3)
        assert int(zl.n_rmu) == 3
        assert zl.zeta_layout == "G_flat"
        assert np.asarray(zl.gvec_components).shape == (2, 3, 4)
        assert np.asarray(zl.ngk_per_q).shape == (2,)
        assert int(zl.ngkmax_zeta) == 4
        assert float(zl.zeta_cutoff_ry) == 10.0
        assert zl.zeta_is_done is True
        # Derived.
        assert zl.n_q_on_disk == 2 and zl.n_rmu_disk == 3
        assert zl.n_G_sph_disk == 4 and zl.n_rtot_disk == 512
        assert zl.n_q_full == 8                # prod(kgrid=(2,2,2))
        assert zl.q_layout == "ibz"            # 2 on disk != 8 full
        assert zl.n_rtot == 512


def test_the_pin_covers_everything_the_binders_actually_bind(tmp_path):
    """RED TWIN of the pin: the pin must not be a SUBSET nobody notices.

    A hand-written tuple drifts by omission — a new header field lands, the
    tuple does not grow, and the "surface" quietly stops describing the
    door.  So the tuple is diffed against what the binders bind: every
    ``MfHeader`` field and every attribute ``bind_isdf_attrs`` sets must be
    accounted for, either in :data:`HEADER_SURFACE` or in the explicitly
    EXCLUDED list below (names no production call site reads).  A new field
    lands in NEITHER and this cell fails, which is the only way the
    enumeration stays true.
    """
    _needs_host_tree()
    from file_io.mf_header import MfHeader

    #: mf_header fields nothing in the tree reads off a ZetaLoader.  Named
    #: rather than dropped: "not consumed" is a claim, and the census that
    #: backs it is survey §1.3's call-site table.
    excluded = {"version", "flavor", "ecutwfc", "shift", "kweights",
                "energies", "occs", "ng", "ecutrho", "cell_symmetry",
                "ngkmax"}
    fields = set(MfHeader._fields)
    unaccounted = fields - set(_BOUND_MF) - excluded
    assert unaccounted == set(), (
        f"mf_header grew fields the header-surface pin does not mention: "
        f"{sorted(unaccounted)}.  Add them to _BOUND_MF (a caller reads "
        f"them) or to `excluded` (nothing does) — silence is what makes an "
        f"enumeration stop being one.")

    # The isdf half has no _fields to diff against (bind_isdf_attrs is a
    # hand-written body), so diff against a REAL bound object instead.
    path, _p = Z.build_gflat(tmp_path / "z.h5")
    probe_obj = type("_Probe", (), {})()
    from file_io.isdf_header import bind_isdf_attrs, read_isdf_header
    bind_isdf_attrs(probe_obj, read_isdf_header(path))
    bound = {k for k in vars(probe_obj) if not k.startswith("_")}
    assert bound - set(_BOUND_ISDF) == set(), (
        f"bind_isdf_attrs binds names the pin does not list: "
        f"{sorted(bound - set(_BOUND_ISDF))}")
    assert set(_BOUND_ISDF) - bound == set(), (
        f"the pin lists isdf names the binder no longer binds: "
        f"{sorted(set(_BOUND_ISDF) - bound)}")


def test_q_layout_reads_full_bz_when_the_disk_holds_every_q(tmp_path):
    """The other arm of ``q_layout``.  ``prod(kgrid)`` rows on disk."""
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=8, n_rmu=3, ngkmax=4,
                             kgrid=(2, 2, 2))
    with ZetaLoader(path) as zl:
        assert zl.n_q_full == 8 and zl.q_layout == "full_bz"


# ===========================================================================
# 7. gvecs() / ngk_valid() — the VALIDATED G-list surface (design D2)
# ===========================================================================

def _ragged(tmp_path, *, n_q=4, n_rmu=3, ngkmax=6, fft_grid=(8, 8, 8)):
    """A file whose per-q spheres are genuinely ragged.  Returns (path, ngk)."""
    ngk = np.asarray([ngkmax, ngkmax - 1, ngkmax - 3, ngkmax][:n_q],
                     dtype=np.int32)
    path, payload = Z.build_gflat(tmp_path / "zr.h5", n_q=n_q, n_rmu=n_rmu,
                                  ngkmax=ngkmax, fft_grid=fft_grid,
                                  ngk_per_q=ngk)
    return path, ngk, payload


def test_gvecs_and_ngk_valid_mirror_the_wfn_loader_surface(tmp_path):
    """Shape, dtype, pad and the explicit-q-subset resolution.

    ``gvecs`` is where the ONLY read-time check that
    ``isdf_header/gvec_components`` agrees with the ``mf_header`` FFT grid
    lives, which is why design D2 KEPT it against the survey's deletion
    listing: ``v_q_g_flat`` and ``vq_interp`` consume ``gvec_components``
    RAW today, so the service should be offering the validated accessor
    rather than deleting it.
    """
    _needs_host_tree()
    from common.gvec_fft_box import fft_box_pad_sentinel
    grid = (8, 8, 8)
    path, ngk, _payload = _ragged(tmp_path, fft_grid=grid)
    # ANTI-TAUTOLOGY: the fixture must actually pad, or every pad assertion
    # below is vacuous.  (test_gvec_padded_layout.py:208's pattern.)
    assert int(ngk.min()) < int(ngk.max()), "fixture must actually pad"
    sent, _flat = fft_box_pad_sentinel(grid)

    with ZetaLoader(path) as zl:
        g = zl.gvecs(q="ibz")
        nv = zl.ngk_valid(q="ibz")
        assert g.shape == (len(ngk), int(zl.ngkmax_zeta), 3)
        assert g.dtype == np.int32
        assert nv.tolist() == ngk.tolist()
        for j in range(g.shape[0]):
            n = int(nv[j])
            assert np.array_equal(
                g[j, n:], np.broadcast_to(sent, (g.shape[1] - n, 3))), \
                f"q={j}: pad rows are not the sentinel"
        # Explicit q-subset resolves like WfnLoader's explicit k-list.
        assert np.array_equal(zl.gvecs(q=[2, 0]), g[[2, 0]])
        assert zl.ngk_valid(q=[2, 0]).tolist() == [int(ngk[2]), int(ngk[0])]


def test_gvecs_refuses_a_components_table_built_on_another_grid(tmp_path):
    """RED TWIN of the components/FFT-grid agreement.

    The components table and ``mf_header/gspace/FFTgrid`` are stored
    SEPARATELY and the components mean nothing without the grid they were
    built on.  Rewriting the pad rows to ANOTHER grid's sentinel is what
    that disagreement looks like on disk; it must be caught at READ time
    rather than silently overwritten by the re-pad inside
    ``pad_gvecs_to_sentinel``.
    """
    _needs_host_tree()
    import h5py as h5
    from common.gvec_fft_box import fft_box_pad_sentinel

    grid = (8, 8, 8)
    path, ngk, _payload = _ragged(tmp_path, fft_grid=grid)
    bad = str(tmp_path / "zeta_bad.h5")
    shutil.copy(path, bad)
    other, _ = fft_box_pad_sentinel((6, 6, 10))
    j = int(np.argmin(ngk))
    with h5.File(bad, "a") as f:
        ds = f["isdf_header/gvec_components"]
        arr = ds[...]
        arr[j, :, int(ngk[j]):] = np.asarray(other)[:, None]
        ds[...] = arr

    with ZetaLoader(bad) as zl:
        with pytest.raises(ValueError) as ei:
            zl.gvecs(q="ibz")
        msg = str(ei.value)
        assert "not the pad sentinel" in msg
        assert f"row q={j}" in msg
        assert "(8, 8, 8)" in msg               # the grid it was read on
    # …and the UNMUTATED file still passes, or the refusal proves nothing
    # about the mutation.
    with ZetaLoader(path) as zl:
        assert zl.gvecs(q="ibz").shape[0] == len(ngk)


def test_gvecs_refuses_an_r_space_file_naming_the_missing_table(tmp_path):
    _needs_host_tree()
    path, _p = Z.build_rspace(tmp_path / "z.h5", n_q=2, n_rtot=8, n_rmu=4)
    with ZetaLoader(path) as zl:
        with pytest.raises(ValueError, match="no per-q G-list"):
            zl.gvecs()
        with pytest.raises(ValueError, match="no per-q logical extent"):
            zl.ngk_valid()


def test_the_ragged_pad_slots_are_zero_on_disk(tmp_path):
    """The writer's contract, asserted on the fixture the reader trusts.

    ``_read_g_flat_disk`` says "pad slots at ``j >= ngk[q]`` are zero by
    writer construction, so the caller can ignore them".  Every consumer
    downstream of the reader relies on that; nothing checked it.
    """
    _needs_host_tree()
    path, ngk, payload = _ragged(tmp_path)
    assert int(ngk.min()) < payload.shape[2], "fixture must actually pad"
    with ZetaLoader(path) as zl:
        for q in range(len(ngk)):
            tile = np.asarray(zl.read_zeta_G_local(q))
            n = int(ngk[q])
            assert np.count_nonzero(tile[:, :n]) == tile[:, :n].size
            assert not tile[:, n:].any(), f"q={q}: pad slots are not zero"


# ===========================================================================
# 8. load()'s remaining refusals
# ===========================================================================

def test_load_refuses_full_bz_on_an_ibz_file_naming_the_post_v_q_unfold(
        tmp_path):
    """``q='full_bz'`` keeps its NotImplementedError verbatim (design D3).

    It fires BEFORE the transport refusal on purpose: a caller on a machine
    with no phdf5 FFI is told that it asked for something this reader does
    not do, rather than being told about a transport it was never going to
    reach.  That ORDER is the assertion — this cell runs at ``mesh=None``,
    where the transport refusal is armed and must not win.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, kgrid=(2, 2, 2))
    with ZetaLoader(path) as zl:
        assert zl.q_layout == "ibz"
        with pytest.raises(NotImplementedError) as ei:
            zl.load(q="full_bz")
        assert "unfold_v_q" in str(ei.value)


@pytest.mark.parametrize("bad,match", [
    ("sideways", "must be 'ibz' or 'full_bz'"),
    ([], "non-empty"),
    ([0, 99], r"out of \[0, 2\)"),
    (np.zeros((2, 2), dtype=np.int32), "must be 1-D"),
])
def test_resolve_q_refuses_bad_q_specs(tmp_path, bad, match):
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2)
    with ZetaLoader(path) as zl:
        with pytest.raises(ValueError, match=match):
            zl.load(q=bad)


def test_resolve_mu_refuses_a_strided_slice(tmp_path):
    """A μ stride has no on-disk expression, and the refusal says so —
    BEFORE the transport gets a word in.

    ``load``'s stated rule is that refusals which are facts about the FILE
    AND THE REQUEST come before the one that is a fact about the STACK.
    Step 2 of this service found the μ refusal violating it (``_resolve_mu``
    ran after ``_ = self.slab_io``, so at ``mesh=None`` a strided μ slice
    reported the TRANSPORT refusal); this cell documented the defect and
    registered it.  Step 4 moved the call, and this cell now PINS the
    stated order: at ``mesh=None`` the stride refusal fires, and the
    transport refusal still fires for a request that is otherwise valid —
    which is the ordering's red twin, because a regression that swaps the
    two back turns the first assertion into the HEADER-ONLY RuntimeError
    again and this test goes red.
    """
    _needs_host_tree()
    path, _p = Z.build_gflat(tmp_path / "z.h5", n_q=2, n_rmu=6)
    with ZetaLoader(path) as zl:
        # FIXED ORDER: the request refusal wins at mesh=None…
        with pytest.raises(ValueError, match="step 1"):
            zl.load(mu=slice(0, 6, 2))
        # …and a VALID request still reaches the stack refusal.
        with pytest.raises(RuntimeError, match="HEADER-ONLY"):
            zl.load(mu=slice(0, 6))
        # The refusal itself, reached directly, names the fix.
        with pytest.raises(ValueError, match="step 1"):
            zl._resolve_mu(slice(0, 6, 2))
        # …and the unstrided forms resolve, so the refusal is not a wall.
        assert zl._resolve_mu(None) == (0, 6)
        assert zl._resolve_mu(slice(1, 4)) == (1, 4)
        assert zl._resolve_mu((2, 5)) == (2, 5)
