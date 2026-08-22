"""The metadata half of the phdf5 transport, pinned at source level.

WHAT LANDED (2026-08-22) AND WHY IT NEEDS A GATE.  Two C entry points,
``lrx_phdf5_dataset_geometry`` and ``lrx_phdf5_read_whole``, let LORRAX
learn a dataset's shape and read a scalar back WITHOUT opening the file
through a second HDF5 library.  They close two registered defects:

* ``get_dipole_mtxels --parallel-transport-out`` refused itself at P=4 on
  every rank, because ``_FfiBackend._introspect_dataset`` opened
  ``parallel_transport.h5`` with serial h5py while the FFI held the same
  path ``mode='a'`` — which ``file_io.hdf5_owner`` refuses, correctly.  The
  refusal landed AFTER the expensive PT tensor and BEFORE ``dipole.h5``;
* ``gw.qsgw_head.load_parallel_transport_head`` could not read its own
  artifact's scalars.  ``SlabIO.read_slab`` cannot express a rank-0 read
  (a scalar dataspace has no hyperslab), so the loader died on the FIRST
  stamp, ahead of every diagnostic it was written to emit.

WHY THESE TESTS ARE SOURCE-LEVEL.  A C ABI entry point exists in FOUR
places that must agree — the C++ definition, the loader's rename table,
the loader's ``argtypes`` declaration, and the Python wrapper — and the
classic failure is adding it to three.  Nothing at run time can catch that
on a machine whose ``.so`` predates the change, which is every machine
until a rebuild lands; a text gate can, today, on a login node, with no
jax, no GPU and no FFI library.  Same reasoning as
``test_slab_io_routing.test_cpp_stripe_policy_transcribes_the_python_one``.

EVERY SCANNER HAS A RED TWIN, per ``tests/test_layering.py``'s standing
rule: each ``test_*_can_fail`` feeds the same predicate a synthetic source
that does contain the defect and asserts it is caught.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_API_CC = _ROOT / "src" / "ffi" / "cpp" / "phdf5" / "api.cc"
_CONTEXT_CC = _ROOT / "src" / "ffi" / "cpp" / "phdf5" / "context.cc"
_SHARD_INDEX_H = _ROOT / "src" / "ffi" / "cpp" / "phdf5" / "shard_index.h"
_LOADER = _ROOT / "src" / "ffi" / "common" / "ffi_loader.py"
_SLAB_FFI = _ROOT / "src" / "file_io" / "_slab_io_ffi.py"
_SLAB_IO = _ROOT / "src" / "file_io" / "slab_io.py"
_QSGW_HEAD = _ROOT / "src" / "gw" / "qsgw_head.py"

#: The two entry points this file exists for.
_METADATA_ENTRY_POINTS = ("lrx_phdf5_dataset_geometry", "lrx_phdf5_read_whole")


def _c_entry_points(src: str) -> set:
    """Names defined through the ``LRX_C_ENTRY`` macro in a C++ source."""
    return set(re.findall(r"LRX_C_ENTRY\((\w+)\)\s*\(", src))


def _binder_names(src: str) -> set:
    """Names in ``ffi_loader._SHARED_C_ENTRY_POINTS``.

    Line-based, not a lazy ``\\(.*?\\)`` — the tuple carries explanatory
    comments and one of them contains a parenthesised date, which a lazy
    match ends on.  That is not hypothetical: it is what this scanner did
    on its first run, and it reported the two new entry points missing
    while they sat three lines below the cut.
    """
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("_SHARED_C_ENTRY_POINTS = ("):
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip() == ")":
                    return set(re.findall(
                        r'"(\w+)"', "\n".join(lines[i:j])))
            raise AssertionError(
                "_SHARED_C_ENTRY_POINTS has no closing ')' on its own line")
    raise AssertionError("ffi_loader._SHARED_C_ENTRY_POINTS not found")


# ---------------------------------------------------------------------------
# The four places a C entry point has to appear
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _METADATA_ENTRY_POINTS)
def test_the_metadata_entry_points_are_defined_in_cpp(name):
    assert name in _c_entry_points(_API_CC.read_text()), (
        f"{name} is not defined through LRX_C_ENTRY in {_API_CC}.  Without "
        f"the macro the host leg would export the UNSUFFIXED name and "
        f"collide with the CUDA leg under RTLD_GLOBAL (cpp/common/c_abi.h).")


@pytest.mark.parametrize("name", _METADATA_ENTRY_POINTS)
def test_the_metadata_entry_points_are_renamed_on_the_host_leg(name):
    """Every ``LRX_C_ENTRY`` name must be in the loader's rename table.

    ``LRX_C_ENTRY`` appends ``_host`` on the host build, so Python cannot
    reach the symbol under its plain name unless ``_bind_c_abi`` rebinds
    it.  A name missing here works on CUDA and silently does not exist on
    the CPU leg — the worst split, because it only shows up on the
    platform nobody tested.
    """
    assert name in _binder_names(_LOADER.read_text()), (
        f"{name} is missing from ffi_loader._SHARED_C_ENTRY_POINTS, so the "
        f"host leg's {name}_host is unreachable from Python.")


def test_every_c_entry_point_in_api_cc_is_in_the_rename_table():
    """The general form of the test above — no name may be forgotten."""
    defined = _c_entry_points(_API_CC.read_text())
    bound = _binder_names(_LOADER.read_text())
    missing = sorted(defined - bound)
    assert not missing, (
        f"api.cc defines {missing} through LRX_C_ENTRY but "
        f"ffi_loader._SHARED_C_ENTRY_POINTS does not list them; on the host "
        f"leg those symbols carry a _host suffix and nothing rebinds them.")


def test_the_rename_table_scan_can_fail():
    """Red twin: a name only in the C++ must be reported."""
    defined = _c_entry_points(
        'int LRX_C_ENTRY(lrx_phdf5_brand_new)(int64_t a) { return 0; }')
    bound = _binder_names('_SHARED_C_ENTRY_POINTS = (\n    "lrx_phdf5_open",\n)')
    assert sorted(defined - bound) == ["lrx_phdf5_brand_new"]


@pytest.mark.parametrize("name", _METADATA_ENTRY_POINTS)
def test_the_metadata_entry_points_are_hasattr_guarded(name):
    """A library built before 2026-08-22 exports neither name.

    Declaring ``argtypes`` on a missing symbol raises at LOAD, which would
    strand every worktree pinned to the deployed Aug-7 pair over an entry
    point those libraries never had.  The ratchet belongs on the artifact
    (``test_so_acceptance.py``), not on the loader — the same reasoning
    ``_bind_c_abi``'s docstring already records.
    """
    src = _LOADER.read_text()
    assert f'hasattr(lib, "{name}")' in src, (
        f"{name}'s argtypes declaration must sit behind a hasattr guard in "
        f"ffi_loader._declare_phdf5; an older .so would otherwise fail to "
        f"load at all.")


def test_the_capability_probe_checks_both_names():
    """``has_phdf5_metadata_api`` must not accept a half-built library."""
    src = _LOADER.read_text()
    body = src[src.index("def has_phdf5_metadata_api"):]
    body = body[:body.index("\n\n\n")]
    for name in _METADATA_ENTRY_POINTS:
        assert name in body, (
            f"has_phdf5_metadata_api does not check {name}.  A library that "
            f"exports one and not the other is not a capability, and the "
            f"caller would take the FFI route and then fail on the other "
            f"call.")


# ---------------------------------------------------------------------------
# The dtype tag table, which has three transcriptions
# ---------------------------------------------------------------------------
def test_the_dtype_tag_table_matches_the_cpp_enum():
    """``ffi_loader._DTYPE_TAG`` and ``dt::Tag`` are one table.

    They are read by different languages off the same integers: Python
    passes the tag to ``lrx_phdf5_ensure_dataset`` / ``lrx_phdf5_read_whole``
    and C++ maps it to an ``hid_t``.  A drift here does not raise — it
    reads the file as the wrong type.
    """
    interface = (_ROOT / "src" / "ffi" / "cpp" / "phdf5"
                 / "phdf5_interface.h").read_text()
    cpp = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"k(F32|F64|S32|S64|C64|C128)\s*=\s*(\d+)", interface)}
    expected = {"F32": 1, "F64": 2, "S32": 3, "S64": 4, "C64": 5, "C128": 6}
    assert cpp == expected, f"dt::Tag drifted: {cpp}"

    loader = _LOADER.read_text()
    block = re.search(r"_DTYPE_TAG = \{(.*?)\}", loader, re.S)
    assert block, "ffi_loader._DTYPE_TAG not found"
    py = {k: int(v) for k, v in re.findall(
        r'"(\w+)":\s*(\d+)', block.group(1))}
    assert py == {
        "float32": 1, "float64": 2, "int32": 3,
        "int64": 4, "complex64": 5, "complex128": 6,
    }, f"_DTYPE_TAG drifted: {py}"


def test_the_tag_table_is_invertible():
    """``_TAG_DTYPE`` must be the exact inverse of ``_DTYPE_TAG``.

    It is built by comprehension today, which makes this a pin on that
    staying true rather than an arithmetic check: two dtype names sharing
    a tag would silently drop one, and ``phdf5_dataset_geometry`` would
    then report the survivor's dtype for both.
    """
    loader = _LOADER.read_text()
    block = re.search(r"_DTYPE_TAG = \{(.*?)\}", loader, re.S)
    py = {k: int(v) for k, v in re.findall(
        r'"(\w+)":\s*(\d+)', block.group(1))}
    assert len(set(py.values())) == len(py), (
        f"two dtype names share a tag in _DTYPE_TAG: {py}")


# ---------------------------------------------------------------------------
# The callers: one HDF5 stack per file
# ---------------------------------------------------------------------------
def test_the_qsgw_head_loaders_do_not_import_h5py():
    """``gw.qsgw_head`` reads ``parallel_transport.h5`` through SlabIO only.

    Both head loaders used to open the artifact with serial h5py for its
    stamps and then open SlabIO for the payload.  The ORDERING was right —
    the h5py handle closed first — but it is still a second HDF5 library
    instance on a file the FFI wrote, which is the cohabitation class
    audit A1 exists to retire, and it is the route the owner's PHDF5-only
    ruling forbids outright.
    """
    src = _QSGW_HEAD.read_text()
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    # Docstrings legitimately mention h5py (they explain what was removed);
    # an IMPORT is the thing that cannot come back.
    assert not re.search(r"^\s*import\s+h5py", code, re.M), (
        "gw/qsgw_head.py imports h5py again.  Its stamps belong in "
        "SlabIO.read_small, in the same read-only handle as the payload.")


def test_the_introspect_prefers_the_ffi():
    src = _SLAB_FFI.read_text()
    body = src[src.index("def _introspect_dataset"):]
    body = body[:body.index("\n    def _platform")]
    ffi_at = body.index("phdf5_dataset_geometry(")
    h5py_at = body.index("import h5py")
    assert ffi_at < h5py_at, (
        "_introspect_dataset must ask the FFI FIRST; the serial-h5py "
        "introspect is the fallback for a library that predates the entry "
        "point, not the primary route.")
    assert "_announce_legacy_introspect" in body, (
        "the fallback must announce itself — announce-or-refuse; a run does "
        "not get to map a second HDF5 library silently.")
    assert 'if self.mode != "r"' in body, (
        "the fallback must REFUSE on a handle that can write: there is "
        "nothing legal to fall back to there, and hdf5_owner would refuse "
        "it anyway with a message that names no repair.")


def test_read_small_is_a_separate_door_from_read_slab():
    """A scalar is not a degenerate hyperslab, and the API says so."""
    src = _SLAB_IO.read_text()
    assert "def read_small(" in src, (
        "SlabIO.read_small is the rank-0/scalar reader; without it "
        "load_parallel_transport_head has no way to read its own stamps.")
    ffi = _SLAB_FFI.read_text()
    assert "def read_whole(" in ffi
    # The refusal read_small exists to route around must STAY in place: a
    # rank-0 request through read_slab is a caller mistake, not a feature.
    assert 'raise ValueError(f"{op} {name!r}: slab shape must be non-empty")' \
        in ffi, (
            "the non-empty-slab refusal must remain: read_slab cannot serve "
            "a scalar dataspace, and silently routing one to read_whole "
            "would hide a sharding mistake in a caller that meant an array.")


# ---------------------------------------------------------------------------
# The descriptor classifier
# ---------------------------------------------------------------------------
#: Every impossible-descriptor value on record, with the ``f64`` it decodes
#: to.  Sources: SLAB_IO_ROOT_CAUSE_AUDIT §A/S3 (2026-08-15, JID 57038615),
#: the two 2026-08-17 register sightings, and KNOWN_FAILURES L1.
_RECORDED_DESCRIPTORS = (
    4462667732332943029,
    -9223372036854775808,
    -4642951212449158796,
    4596944070643295330,
)


def test_the_descriptor_classifier_exists_and_is_wired_in():
    """It must be CALLED, not merely defined.

    An instrument that is present and unreachable reports success from a
    path that never ran — this project's most-repeated failure shape.
    """
    assert "descriptor_forensics" in _SHARD_INDEX_H.read_text()
    read_cc = (_ROOT / "src" / "ffi" / "cpp" / "phdf5" / "read_ffi.cc").read_text()
    write_cc = (_ROOT / "src" / "ffi" / "cpp" / "phdf5" / "write_ffi.cc").read_text()
    assert read_cc.count("descriptor_forensics(") >= 2, (
        "read_ffi.cc must decode the operand on BOTH descriptor refusals "
        "(negative offset/valid_shape, and logical-slab-out-of-bounds).")
    assert "descriptor_forensics(" in write_cc


def test_every_recorded_descriptor_decodes_as_a_small_double():
    """The claim the classifier's message rests on, checked against the data.

    If a future sighting does NOT decode this way, the stale-allocation
    hypothesis is wrong for it and the message says so — but the four on
    record must all fit, or the message is asserting more than the
    evidence.  This is arithmetic on published numbers, so it needs no
    compiler and no library.
    """
    import struct
    for raw in _RECORDED_DESCRIPTORS:
        as_f64 = struct.unpack("<d", struct.pack("<q", raw))[0]
        mag = abs(as_f64)
        assert as_f64 == 0.0 or 1e-30 < mag < 1e30, (
            f"{raw} decodes to {as_f64!r}, which the classifier would NOT "
            f"call a plausible double — the register's stale-allocation "
            f"reading does not cover this value.")


def test_the_descriptor_classifier_can_fail():
    """Red twin: a genuinely widened int32 must NOT read as a small double.

    This is the discriminating case, and it is why the classifier is worth
    having: an int32 sign-extended or zero-extended into an int64 leaves
    the top half all-ones or all-zeros, which as a double is a denormal or
    an absurd magnitude — never the ~1e-2 the record shows.
    """
    import struct
    for raw in (-1, 1, 2 ** 31, -(2 ** 31), 0xFFFFFFFF):
        as_f64 = struct.unpack("<d", struct.pack("<q", raw))[0]
        mag = abs(as_f64)
        plausible = (as_f64 == 0.0) or (1e-30 < mag < 1e30)
        assert not plausible, (
            f"widened-int32 pattern {raw} decoded to {as_f64!r}, which the "
            f"classifier would misreport as a stale float64 allocation.")
