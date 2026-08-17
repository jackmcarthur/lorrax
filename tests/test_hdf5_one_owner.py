"""One HDF5 library instance per open file — the registry, at the seam.

WHAT THIS SUITE IS ABOUT.  The deployed run environment maps TWO
independent HDF5 library instances (h5py's bundled wheel libhdf5 and the
FFI's cray libhdf5_parallel), and ``file_io.mpa_store``'s files are
touched through both.  Audit A1 names that as the standing explanation
for the metallic driver's iteration-3 rank-0 segfault.
:mod:`file_io.hdf5_owner` turns the condition from silent into either a
counted measurement or a named refusal; this suite pins WHICH, because a
guard that refuses the benign case would refuse the whole pipeline and a
guard that allows the undefined case buys nothing.

HOST-SIDE, REAL STORES.  Every gate below runs on an actual fit store
built by ``allocate_fit_store``/``write_fit_block``/``finalize_fit_store``
— the real input shape, reached through the real ``mpa_store`` entry
points — with the FFI side of each scenario declared through the same
``note_open`` call ``SlabIO`` itself makes.  The phdf5 FFI is not
available in every environment this suite runs in, and faking the CLAIM
rather than the transport is what keeps the guard's semantics testable
without one; the transport's own leg is the R6 run log's probe lines.
"""

import os
import tempfile

import numpy as np
import pytest

from file_io import hdf5_owner as HO
from file_io import mpa_store as MS


@pytest.fixture(autouse=True)
def _clean_registry():
    HO.reset_for_test()
    yield
    HO.reset_for_test()


@pytest.fixture()
def fit_store():
    """A real, finalized MPA fit store on disk."""
    d = tempfile.mkdtemp(prefix="hdf5_owner_")
    path = os.path.join(d, "mpa_fit_sc_0000.h5")
    n_q, n_mu, n_p = 1, 4, 2
    MS.allocate_fit_store(path, n_q=n_q, n_mu=n_mu, n_p=n_p,
                          energy_unit="Ry")
    rng = np.random.default_rng(7)
    Om = (rng.normal(size=(n_p, n_mu, n_mu))
          - 1j * rng.uniform(0.01, 0.1, size=(n_p, n_mu, n_mu)))
    Bp = rng.normal(size=(n_p, n_mu, n_mu)) + 0j
    diag = {"condition": np.full((n_mu, n_mu), 2.0),
            "backward_error": np.full((n_mu, n_mu), 1e-14)}
    MS.write_fit_block(path, 0, np.arange(n_mu), Om, Bp, diag)
    MS.finalize_fit_store(
        path, certification={"condition_max_allowed": 1e6,
                             "backward_error_max_allowed": 1e-8})
    try:
        yield path
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# The refusal, on the real input shape
# ---------------------------------------------------------------------------

def test_h5py_read_is_refused_while_the_ffi_holds_a_write_handle(fit_store):
    """The undefined case, named at the door instead of segfaulting.

    This is the iteration-3 shape: the FFI has the fit store open for
    writing (``write_fit_block_collective``'s ``SlabIO(mode='a')``) and a
    serial-h5py reader walks in on the same path.  Two metadata caches,
    one of them dirty.  The measured symptoms on this stack are "file
    signature not found" (job 7888644) and a native rank-0 segfault; the
    guard converts both into a message that names the file, both holders
    and the fix.
    """
    token = HO.note_open(fit_store, HO.STACK_FFI, "a",
                         where="test: SlabIO(mode='a')")
    try:
        with pytest.raises(RuntimeError) as exc:
            MS.fit_completion_ledger(fit_store)
        msg = str(exc.value)
        assert "one-owner-per-file" in msg
        assert "audit A1" in msg and "claims/0110" in msg
        assert os.path.basename(fit_store) in msg
        # got / want / fix, and both holders named.
        assert "got   :" in msg and "want  :" in msg and "fix   :" in msg
        assert "SlabIO(mode='a')" in msg
    finally:
        HO.note_close(fit_store, token)


def test_the_ffi_is_refused_while_h5py_holds_a_write_handle(fit_store):
    """Symmetric: the guard is about the PAIR, not about who arrived first."""
    with MS._h5(fit_store, "a", where="test: rank-0 ledger commit"):
        with pytest.raises(RuntimeError, match="one-owner-per-file"):
            HO.note_open(fit_store, HO.STACK_FFI, "r",
                         where="test: SlabIO(mode='r')")


def test_read_only_cross_stack_concurrency_is_allowed_and_counted(fit_store):
    """The WFN.h5 shape: two libraries reading a file nobody mutates.

    ``wfn_loader`` holds an h5py handle and a SlabIO handle on ``WFN.h5``
    read-only for the whole run.  Neither cache can diverge from the
    other when nothing writes, so refusing this would refuse the pipeline
    for no defect.  It is still recorded — the path shows up as touched
    by both stacks, which is what the probe reports.
    """
    token = HO.note_open(fit_store, HO.STACK_FFI, "r",
                         where="test: SlabIO(mode='r')")
    try:
        ledger = MS.fit_completion_ledger(fit_store)      # must not raise
        assert ledger["complete"] is True
    finally:
        HO.note_close(fit_store, token)
    assert os.path.abspath(fit_store) in HO.shared_paths()


def test_same_stack_reopens_stay_legal(fit_store):
    """Two h5py handles on one file are h5py's problem, not this guard's."""
    with MS._h5(fit_store, "r", where="test: outer"):
        MS.fit_completion_ledger(fit_store)               # must not raise


# ---------------------------------------------------------------------------
# The door's lifetime contract
# ---------------------------------------------------------------------------

def test_the_door_releases_its_claim_when_the_body_raises(fit_store):
    """Close-on-exception, or the guard poisons the path it protects.

    A door that leaked its registration on the error path would leave the
    registry believing an h5py handle is still live, and every later
    legitimate FFI open on that file — including the recovery path — would
    refuse.  A guard that fails closed FOREVER after one exception is
    worse than no guard.
    """
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with MS._h5(fit_store, "a", where="test: failing writer"):
            raise Boom("the writer died mid-block")

    # The claim is gone: the FFI can open, and so can h5py.
    token = HO.note_open(fit_store, HO.STACK_FFI, "a", where="test: after")
    HO.note_close(fit_store, token)
    assert MS.fit_completion_ledger(fit_store)["complete"] is True


def test_a_failed_open_does_not_leave_a_claim(fit_store):
    """Same rule one level up: an open that never succeeded holds nothing."""
    missing = os.path.join(os.path.dirname(fit_store), "not_here.h5")
    with pytest.raises(Exception):
        with MS._h5(missing, "r", where="test: absent file"):
            pass
    token = HO.note_open(missing, HO.STACK_FFI, "a", where="test: after")
    HO.note_close(missing, token)


# ---------------------------------------------------------------------------
# Sequential alternation: counted by default, refused under strict
# ---------------------------------------------------------------------------

def test_sequential_alternation_is_counted_not_refused(fit_store, monkeypatch):
    """What the pipeline does today, and what the probe reports.

    Open/close through one library, then open/close through the other.
    Every close flushes, so this mostly works — which is exactly why it is
    allowed and why it is the thing that has to be MEASURED rather than
    assumed away.
    """
    monkeypatch.delenv(HO.POLICY_ENV, raising=False)
    for _ in range(3):
        HO.note_close(fit_store,
                      HO.note_open(fit_store, HO.STACK_FFI, "a",
                                   where="test: SlabIO write"))
        MS.fit_completion_ledger(fit_store)
    rows = {r[0]: r for r in HO.alternation_rows()}
    row = rows[os.path.abspath(fit_store)]
    assert row[1] >= 3 and row[2] >= 3, "both stacks opened it"
    assert row[3] >= 5, f"alternations counted, got {row[3]}"
    assert row[4] >= 1, "alternations involving a write are counted apart"


def test_strict_policy_refuses_the_alternation_it_counts(fit_store,
                                                         monkeypatch):
    """``LORRAX_HDF5_ONE_OWNER=strict`` is the target state A1 names.

    The fixture built this store through h5py, so the FIRST foreign open
    is already the alternation — which is the point: under strict the
    refusal lands on the handoff itself, not one round-trip later, and it
    names the counts on both sides so the reader can see how much of the
    file each library has touched.
    """
    monkeypatch.setenv(HO.POLICY_ENV, "strict")
    with pytest.raises(RuntimeError) as exc:
        HO.note_close(fit_store,
                      HO.note_open(fit_store, HO.STACK_FFI, "a",
                                   where="test: SlabIO write"))
    msg = str(exc.value)
    assert "strict policy" in msg and "audit A1" in msg
    assert "last opened through h5py" in msg
    # And the same handoff is merely COUNTED under the default policy.
    monkeypatch.setenv(HO.POLICY_ENV, "measure")
    HO.note_close(fit_store,
                  HO.note_open(fit_store, HO.STACK_FFI, "a",
                               where="test: SlabIO write"))


def test_an_unknown_policy_refuses_by_name(monkeypatch):
    monkeypatch.setenv(HO.POLICY_ENV, "yes")
    with pytest.raises(ValueError, match="LORRAX_HDF5_ONE_OWNER"):
        HO.policy()


# ---------------------------------------------------------------------------
# The /proc/self/maps probe
# ---------------------------------------------------------------------------

def test_the_probe_counts_mapped_libhdf5_objects_and_prints_them():
    """GATE-7 style: measured from the live process, printed, not asserted.

    The COUNT is environment-dependent (one library on a matched stack,
    two on the deployed Perlmutter one), so this pins the shape of the
    measurement rather than its value: h5py is imported by this suite, so
    at least one libhdf5 must be mapped, every name must look like one,
    and the printed line must carry the count and the locking setting.
    """
    import h5py                                            # noqa: F401

    libs = HO.mapped_hdf5_libraries()
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("/proc/self/maps is not available here")
    assert libs, "h5py is imported, so its libhdf5 must be mapped"
    assert all(n.lower().startswith("libhdf5") and ".so" in n for n in libs)
    assert len(set(libs)) == len(libs), "distinct objects, not occurrences"

    lines = []
    out = HO.probe("unit-test", print_fn=lines.append)
    assert out["libraries"] == libs
    joined = "\n".join(lines)
    assert "[hdf5-probe unit-test]" in joined
    assert f"{len(libs)} core mapped" in joined
    assert "HDF5_USE_FILE_LOCKING=" in joined


def test_the_probe_does_not_count_hl_as_a_second_library():
    """``libhdf5_hl`` is a WRAPPER, not a second HDF5 instance.

    h5py's wheel maps both ``libhdf5-<hash>.so`` and
    ``libhdf5_hl-<hash>.so``; the second calls into the first and shares
    its cache.  Counting it would make a perfectly safe single-stack
    process report TWO libraries — and :func:`probe`'s escalation keys on
    that count, so it would report A1's unsafe condition on a machine
    where the condition does not hold.  Measured on the deployed stack:
    the true count is 2 cores (h5py's + the FFI's cray build) and 1
    companion.
    """
    core, companion = HO._mapped_hdf5_objects()
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("/proc/self/maps is not available here")
    assert not any(n.lower().startswith("libhdf5_hl") for n in core), (
        f"libhdf5_hl counted as a core library: {core}")
    assert all(n.lower().startswith("libhdf5") for n in core + companion)
    # The cray parallel build IS a core library despite its long name.
    assert HO._HDF5_COMPANION_PREFIXES and not (
        "libhdf5_parallel_gnu_123.so.200".startswith(
            HO._HDF5_COMPANION_PREFIXES)), (
        "the cray parallel build must classify as core, not companion")


def test_the_probe_names_the_unsafe_condition_when_it_holds(fit_store,
                                                            monkeypatch):
    """Two libraries mapped AND a file written through both = A1's condition.

    Reported as UNSAFE-BY-A1 with the files named.  Measured, not
    asserted: the run proceeds, because the alternation is sequential and
    that is what the pipeline does today.  ``strict`` turns the same
    measurement into the refusal.
    """
    # The deployed pair, faked: this suite's own process maps only h5py's
    # core object (the FFI .so is not loaded under JAX_PLATFORMS=cpu), so
    # the condition has to be constructed to be tested.
    monkeypatch.setattr(
        HO, "_mapped_hdf5_objects",
        lambda: (["libhdf5-9e18f0c6.so.320.0.0",
                  "libhdf5_parallel_gnu_123.so.200"],
                 ["libhdf5_hl-69da89c9.so.320.0.0"]))
    HO.note_close(fit_store,
                  HO.note_open(fit_store, HO.STACK_FFI, "a",
                               where="test: SlabIO write"))
    MS.fit_completion_ledger(fit_store)

    monkeypatch.delenv(HO.POLICY_ENV, raising=False)
    lines = []
    out = HO.probe("sc_0002", print_fn=lines.append)       # must not raise
    assert os.path.abspath(fit_store) in out["unsafe"]
    joined = "\n".join(lines)
    assert "UNSAFE-BY-A1" in joined
    assert os.path.basename(fit_store) in joined

    monkeypatch.setenv(HO.POLICY_ENV, "strict")
    with pytest.raises(RuntimeError, match="probe refusal"):
        HO.probe("sc_0002", print_fn=lambda _s: None)


# ---------------------------------------------------------------------------
# The churn cut this guard gives teeth to
# ---------------------------------------------------------------------------

def test_the_pole_reader_needs_a_mesh_and_says_why():
    """The host path stays :func:`read_poles`; the reader is the mesh one."""
    with pytest.raises(ValueError, match="PoleReader requires mesh_xy"):
        MS.open_pole_reader("ignored.h5", mesh_xy=None)


def test_read_poles_still_reads_the_host_path_whole(fit_store):
    """The one-shot door is unchanged for mesh-less callers."""
    Om, Bp = MS.read_poles(fit_store, pole_slice=slice(0, 1), to_unit="Ry")
    assert Om.shape[0] == 1 and Bp.shape == Om.shape
    with pytest.raises(IndexError, match="outside"):
        MS.read_poles(fit_store, pole_slice=slice(5, 9))
