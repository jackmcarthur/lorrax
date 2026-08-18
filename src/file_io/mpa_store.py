"""Frequency-resolved W restart tensors and the B/Omega fit store.

This module owns MPA sample and pole bytes.  It remains dependency-light:
nothing here imports from ``gw``, the ``symmetry_maps`` door is imported
lazily through :func:`_qs`, and every ``jax`` import is inside the function
that needs it — so importing THIS MODULE costs no jax.  (Importing the
``file_io`` PACKAGE does, and has since the wfn_loader extraction:
``file_io/__init__.py`` imports ``wfn_loader`` at module scope and that
imports jax.  Re-measured 2026-08-15; this sentence used to claim the
package was jax-free and it has not been for some time.)

WHAT THIS FORMAT IS.  The multipole-W fit needs W_c evaluated on the
double-parallel sampling grid — 2·n_p complex frequencies on two lines
ϖ₁ and ϖ₂ (``docs/theory/THEORY_mpa_implementation.md``) — before it can
fit anything, and the
owner's memory constraint is exact: a SMALL NUMBER of W_q(μ,ν) copies
fit in memory at once, but NOT all ω_i.  So every frequency goes to the
restart file, with the frequency axis LEADING::

    (n_omega, n_q_on_disk, N_mu, N_mu)      complex128

prepended to the per-frequency wedge layout the q_irr checkpoint
landed.  The leading position is the point: a slab ``ds[i]`` is a
contiguous ``(n_q, N_mu, N_mu)`` read that is bit-identical to what a
frequency-free file would hold, so the axis is REMOVABLE later without
touching any downstream reader — the fit stage can graduate to holding
one frequency at a time, or the axis can be dropped entirely for a
static-W run, and neither is a format migration.
:func:`read_w_slab` is that read — the HOST-SEAM spelling of it; the
production reader is :func:`read_w_slab_collective`, which reads the same
bytes through SlabIO.  The removability claim has a test.

THE WEDGE APPLIES PER FREQUENCY.  W_c(q, ω) transforms under the space
group exactly as W_c(q) does at each ω separately — the symmetry
operation acts on (q, μ, ν) and does not touch ω — so the stored tables
(``irr_idx_q``, ``sym_idx_q``, ``q_irr_frac``, ``sym_perm``,
``L_table``, ``n_sym_spatial``) are shared across the whole frequency
axis and unfolding is ``unfold_isdf_operator`` applied slab by slab.
One table group per tensor, not one per frequency: n_omega copies of a
table that cannot differ is n_omega chances for them to differ.

VERSION 2, AND WHY IT IS NOT OPTIONAL.  The frequency axis bumps
``qirr_format_version`` from 1 to 2.  It has to, and the reason is the
sharpest failure this format has: a version-1 reader handed a
``(n_omega, n_q_ibz, N_mu, N_mu)`` dataset stamped version 1 takes
``ds.shape[0]`` as the q extent and ``ds.shape[-1]`` as the μ extent,
and BOTH ARE THE RIGHT SHAPE — the tables validate, the shape-vs-attr
cross-check agrees, nothing refuses — whenever ``n_omega`` happens to
equal the wedge extent.  Si 4³ reduces 64 q to 8 and an n_p = 4 fit
samples 8 frequencies; that coincidence is one deck away, not one in a
million.  So: the writer stamps 2, ``qirr_store.read_tensor`` refuses
any rank but 3 under its own stamp, and
:func:`_refuse_unless_rank_matches_version` discriminates on the RANK
before any reader here believes an extent.

PRESENCE IS NEVER READINESS, PER FREQUENCY.  ``gw_init`` allocates a
full-size zero ``W0`` before the screening that fills it exists, and the
April BSE incident was a plausible excitonic spectrum out of an all-zero
screening tensor that passed every shape check.  A frequency-resolved
file makes that worse, not better: the producer fills ω slabs one at a
time, so a file with 16 slabs allocated and 9 written is a state the
pipeline REACHES ROUTINELY rather than a state it crashes into.  The
``data_ready`` ledger in the ``<name>__mpa`` group carries one bool per
frequency and :func:`read_w_slab` refuses an unstamped slab by index.
The scalar ``qirr_data_ready`` attr is stamped beside it as
``all(ledger)`` so that any reader honouring the v1 attr gets the
CONSERVATIVE answer, and a disagreement between the two refuses.

THE μ PAD DOES NOT REACH DISK.  Inherited unchanged from the q_irr
checkpoint and restated because it is the rule most easily lost when a
layout gains an axis: stored tables and tensors are LOGICAL, the pad
width is ``padded_mu_extent(n_rmu, device_count())`` and therefore
device-count-dependent, and SHARDING_RULES §2 forbids such a quantity
in a restart artifact because a file written on four ranks must read on
eight.  Readers re-pad against their OWN count via ``n_mu_padded=``.

THE COLUMN READER IS THE MEMORY ARGUMENT.  Per-element plasmon-pole
fits want a few ν columns ACROSS ALL FREQUENCIES, never a full
(N_μ, N_μ) frequency slab and never all of ω for a full row-block.
:func:`read_w_columns` is that read (again the host seam; production goes
through :func:`read_w_columns_collective`) and :func:`choose_column_budget`
is its arithmetic; the budget is sized so the returned block costs about
what ONE (N_μ, N_μ) tile costs, which is the unit the owner's
constraint is stated in.  The block is 1-D SHARDED ON THE ROW AXIS
ONLY — never 2-D — because the fit is elementwise in (μ, ν) and a
second split on the column axis buys nothing while making every rank's
column count a function of the mesh shape.

EVERY FILE THIS MODULE TOUCHES IS TOUCHED THROUGH TWO HDF5 LIBRARY
INSTANCES, and that is the module's sharpest operational constraint.
The bytes go through ``SlabIO`` (the FFI's cray parallel libhdf5); the
ledger, the attrs and the unfold tables go through h5py (its own
bundled libhdf5).  Two instances, two metadata caches, two open-file
tables, one file — undefined the moment they overlap with a writer
(audit A1; sandbox claims/0110;
``docs/architecture/slab_io.md#one-owner``).  So:

* **every** h5py open in this module goes through :func:`_h5`, which
  declares to :mod:`file_io.hdf5_owner` and is refused BY NAME if the
  FFI holds a live handle on the same path.  A new h5py open added
  outside that door is a defect, not a shortcut;
* the collective readers do their h5py work BEFORE the collective
  handle opens and none after — :class:`PoleReader` holds ONE
  ``SlabIO`` across an iteration's pole batches for exactly that
  reason;
* the alternation is COUNTED, per path, and reported: measured 2026-08-15
  at **1027 cross-library alternations on one file in one SC
  iteration**, most of them this module's own fit-block writer.
  ``LORRAX_HDF5_ONE_OWNER=strict`` turns the count into a refusal.

Testing note: everything below is exercised host-side with plain h5py
at LOGICAL extents.  The phdf5 FFI is not built on WSL, so the format
is tested at its seams the way the symmetry lane tested the q_irr
format; the ``SlabIO`` write path (where each rank contributes its own
(μ, ν) hyperslab and no rank holds the whole array) has its Perlmutter
leg in ``tests/multi_device/mpa_fit_stream_gate.py``, and
:func:`stamp_w_omega` exists for exactly that split — the producer
writes the bytes with the machinery it already has and this stamps
them.  The serial family is the HOST TEST SEAM those suites run on —
**zero ``src`` callers, by design**; production writes and reads
through the ``*_collective`` forms.  Re-counted 2026-08-15, the family
is larger than this note used to say, and the full list matters because
"no src caller" is otherwise read as "dead":
:func:`allocate_w_omega`, :func:`write_w_slab`, :func:`read_w_slab`,
:func:`read_w_columns`, :func:`allocate_fit_store`,
:func:`write_fit_block`, :func:`write_head_fit`, :func:`read_head_fit`,
:func:`read_fit_block`, :func:`read_fit_tensors` and :func:`read_poles`.
Each has a ``*_collective`` twin (or, for :func:`read_poles`,
:class:`PoleReader`) that production uses instead.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import os
import sys

import numpy as np

#: The frequency-resolved layout.  Registered ALONGSIDE version 1 rather
#: than replacing it: a v1 file is still a v1 file and still reads, and
#: :data:`QIRR_FORMAT_VERSIONS_READABLE` is the set the widened reader
#: accepts.  The value lives here rather than in ``qirr_store`` only
#: while this module is staged outside the format layer; when it moves,
#: it moves next to ``QIRR_FORMAT_VERSION``.
QIRR_FORMAT_VERSION_FREQ = 2

#: Every version :func:`_refuse_unless_rank_matches_version` accepts.
#: A reader that
#: reads an unknown version best-effort returns wrong numbers on the day
#: the layout changes; a reader that accepts a KNOWN version without
#: checking the rank returns wrong numbers on the day the layout gains
#: an axis, which is this day.
QIRR_FORMAT_VERSIONS_READABLE = (1, 2)

#: Rank of the stored dataset THIS format adds — version 2 is
#: ``(n_omega, n_q, N_μ, N_μ)``.  The rank is the discriminant, not a
#: consistency nicety; see :func:`_refuse_unless_rank_matches_version`.
#:
#: Version 1's rank is NOT restated here.  It is
#: ``symmetry_maps.QIRR_RANK_BY_VERSION``'s to state and
#: ``qirr_store.read_tensor``'s to enforce, and the format layer asks
#: extenders in as many words to compose (``{**QIRR_RANK_BY_VERSION, 2:
#: 4}``) rather than to restate — a second copy of "version 1 is rank 3"
#: is a second thing to update on the day it is not.  Composed through
#: :func:`_rank_by_version` because the door is imported lazily.
_MPA_RANK = 4


def _rank_by_version():
    """Every version's rank: the format layer's table, plus ours."""
    return {**_qs().QIRR_RANK_BY_VERSION, QIRR_FORMAT_VERSION_FREQ: _MPA_RANK}

#: Sibling group holding the ω grid, its protocol provenance, and the
#: per-frequency readiness ledger.  Beside the tensor and never
#: elsewhere, for the same reason the unfold tables are: a tensor whose
#: sampling grid lives in another file is a tensor that silently decays
#: when anything upstream is regenerated.
MPA_GROUP_SUFFIX = "__mpa"

#: Sibling group holding the staged-fit completion ledger.
MPA_FIT_SUFFIX = "__mpafit"

#: Tiny scalar q->0 fit beside, but independent of, the body pole tensors.
MPA_HEAD_SUFFIX = "__mpahead"

# The fit keeps the W wedge's q map beside its poles.  The tables belong to
# ``symmetry_maps``; this is only the dataset name under which that service
# files them.
FIT_TABLE_OWNER = "Omega_p"
FIT_ENERGY_UNIT_ATTR = "mpa_fit_energy_unit"
FIT_ENERGY_UNITS = {"Ry": 1.0, "Ha": 2.0}

#: The two per-element fit diagnostics the format REQUIRES; everything
#: else a fit measures is extra.  Single source for the names derived
#: from them on disk — the ``fit_<k>`` datasets, the ``block_<k>_max``
#: journal columns and the ``<k>_max_allowed`` certification attrs — so
#: a third required diagnostic cannot be added at one site and missed at
#: another.
REQUIRED_DIAGNOSTICS = ("condition", "backward_error")

#: Every scalar-head fit model :func:`read_head_fit` knows how to
#: interpret.  The reader refuses an unknown model rather than serving
#: poles whose fitting protocol nobody can name.
_HEAD_FIT_MODELS = (
    "fixed_dft_gn",
    "dft_direct_loewner",
    "bgw_q0shift_loewner",
    "qsgw_direct_loewner",
    "qsgw_schur_loewner",
    "dft_direct_companion",
    "bgw_q0shift_companion",
    "qsgw_direct_companion",
    "qsgw_schur_companion",
    "dft_direct_thiele",
    "bgw_q0shift_thiele",
    "qsgw_direct_thiele",
    "qsgw_schur_thiele",
)

#: Bump when the fit store's layout changes.  Independent of the W
#: format's version: the two files have separate lifetimes and a reader
#: of one is not a reader of the other.
MPA_FIT_FORMAT_VERSION = 1

#: Attr marking the leading frequency axis.  Its presence is the ATTR
#: half of the rank cross-check, and the string names what is removable.
_FREQ_ATTR = "mpa_freq_axis"
_FREQ_ATTR_VALUE = "leading"

#: Sampling-protocol keys every W(ω) file must carry, with the units the
#: theory plan states them in.  Required rather than defaulted: a fit
#: whose partition α nobody recorded is a fit nobody can reproduce, and
#: α is the one parameter that differs between insulators (1) and metals
#: (2) while changing nothing about the shapes.
_SAMPLING_REQUIRED = ("varpi", "n_p", "alpha", "omega_max")

#: Everything the ω-grid digest covers, in a fixed order.
_SAMPLING_ORDER = ("protocol", "varpi", "n_p", "alpha", "omega_max")

#: The attrs version 2 adds on top of the version-1 q_irr set — the
#: EXACT difference between the two stamps, which is what makes the
#: removability claim checkable: set these aside and the version number,
#: and a v2 file's attrs must equal a v1 file's attr for attr.  An
#: explicit closed set rather than an ``mpa_`` prefix match, so that an
#: attr added later to one format and not the other fails the comparison
#: instead of being swallowed by a ``startswith``; the sampling portion
#: is derived from :data:`_SAMPLING_ORDER`, the one list of sampling
#: keys.  (``mpa_writer`` is deliberately absent: it says BY WHAT, not
#: WHAT, and belongs with the timestamps the comparison already
#: exempts.)
_MPA_OWNED_ATTRS = (
    _FREQ_ATTR, "mpa_n_omega", "mpa_omega_units",
    *("mpa_" + key for key in _SAMPLING_ORDER),
    "mpa_grid_hash",
)

#: Occupation-provenance stamps, in a fixed order; attr name = "mpa_" +
#: key.  Written only when an occupation state is supplied, so insulating
#: stores stay byte-identical.  Asserted at REUSE sites only: a same-run
#: assert would compare the store to the state that just wrote it (the
#: same circularity that retired fit_identity), so the consumers are the
#: cross-run reader and the QSGW per-iteration log.
_OCC_STAMP_ORDER = (
    "occ_hash", "mu_ry", "smearing_family", "smearing_width_ry",
    "occ_nelec")


def _occ_stamp_values(occupation_state):
    """The five stamp values from a duck-typed occupation state."""
    st = occupation_state
    return {
        "occ_hash": str(st.occ_hash),
        "mu_ry": float(st.mu_ry),
        "smearing_family": str(st.smearing_family),
        "smearing_width_ry": float(st.smearing_width_ry),
        "occ_nelec": float(st.n_electrons),
    }


def stamp_occupation_provenance(obj, occupation_state):
    """Stamp the ``mpa_<key>`` occupation attrs on an h5 object."""
    for key, val in _occ_stamp_values(occupation_state).items():
        obj.attrs["mpa_" + key] = val


def read_occupation_stamps(src, *, mode="r"):
    """Return the fit store's occupation stamps, or None if unstamped."""
    qs = _qs()
    with _h5(src, mode) as grp:
        if ("mpa_" + _OCC_STAMP_ORDER[0]) not in grp.attrs:
            return None
        return {
            "occ_hash": qs.qirr_attr_str(grp, "mpa_occ_hash"),
            "mu_ry": float(grp.attrs["mpa_mu_ry"]),
            "smearing_family": qs.qirr_attr_str(grp, "mpa_smearing_family"),
            "smearing_width_ry": float(grp.attrs["mpa_smearing_width_ry"]),
            "occ_nelec": float(grp.attrs["mpa_occ_nelec"]),
        }


def assert_occupation_stamps(src, occupation_state, *, where="fit store"):
    """REUSE-site gate: the store's occupations are the run's occupations.

    Refuses an unstamped store outright — a metallic reuse of a store
    that predates the stamps cannot certify compatibility and must be
    regenerated.  Never called on the same-run write path (W4: stamps are
    asserted at REUSE sites only; claim 0194).

    WHAT IS ACTUALLY COMPARED, because the summary line over-reaches:
    ``occ_hash``, ``smearing_family``, ``smearing_width_ry`` and
    ``occ_nelec``.  **``mu_ry`` is NOT compared** — it is carried in the
    failure message for context only, so a store whose chemical potential
    differs from the run's passes this gate.

    ``occupation_state`` is duck-typed and must expose ``occ_hash``,
    ``mu_ry``, ``smearing_family``, ``smearing_width_ry`` and
    ``n_electrons`` (which is stamped under the key ``occ_nelec`` — the
    attribute and the stamp are deliberately spelled differently and that
    is the only place it is written down).

    Rank-local and serial.  ``src`` is a path (this call's ``_h5`` owns the
    handle) or an open h5py group (the caller owns it); the read is
    hard-wired ``mode='r'``, so passing the PATH while holding the file
    open ``'a'`` yourself puts a second h5py handle on it.
    """
    stamps = read_occupation_stamps(src)
    if stamps is None:
        raise ValueError(
            f"{where}: no occupation stamps present. A metallic reuse "
            "needs a store written with its occupation state "
            "(mpa_occ_hash ...); regenerate the fit store with the "
            "current build.")
    want = _occ_stamp_values(occupation_state)
    hard = ("occ_hash", "smearing_family", "smearing_width_ry", "occ_nelec")
    bad = [k for k in hard if stamps[k] != want[k]]
    if bad:
        detail = ", ".join(
            f"{k}: store {stamps[k]!r} != run {want[k]!r}" for k in bad)
        raise ValueError(
            f"{where}: occupation stamps disagree with the run's "
            f"occupation state ({detail}; store mu_ry={stamps['mu_ry']!r}, "
            f"run mu_ry={want['mu_ry']!r}). The fit was made from "
            "different occupations — regenerate it.")


#: Bytes per complex128 element.  Named because it appears in the budget
#: arithmetic, and a budget whose constants are anonymous is a budget
#: nobody can check against a message.
COMPLEX128_BYTES = 16

#: "the caller did not read this yet", distinct from a legitimate
#: ``None`` (a full-BZ store genuinely has no unfold tables).
_UNREAD = object()

_QS_CACHE: list = []


def _qs():
    """The ``symmetry_maps`` DOOR, imported lazily and once.

    THE DOOR, NOT THE SUBMODULE.  Everything this module needs from the
    q_irr format layer — :class:`~symmetry_maps.QirrDest`,
    :func:`~symmetry_maps.qirr_attr_str`,
    :data:`~symmetry_maps.QIRR_VERSION_ATTR`,
    :data:`~symmetry_maps.QIRR_TABLE_SUFFIX`,
    :func:`~symmetry_maps.validate_qirr_tables`,
    :func:`~symmetry_maps.qirr_generator_commit`,
    :data:`~symmetry_maps.QIRR_RANK_BY_VERSION` and the read/stamp
    entry points — is a TOP-LEVEL name on ``symmetry_maps``.  It was not
    always: this module was written while the q_irr checkpoint was still
    landing, when the format's plumbing was private to
    ``symmetry_maps.qirr_store``, and it reached into that submodule for
    thirty-four of them.  ``tests/test_layering.py``'s door rule counted
    that reach, correctly — a consumer that imports a service's submodule
    is a consumer that stops the service being replaceable — and the
    checkpoint answered it by PUBLISHING the plumbing rather than by
    letting a second store copy it.  The door's own docstring gives the
    reason in the format layer's words.

    STILL LAZY, for the reason that outlived the other one.
    ``symmetry_maps`` is a jax-importing package and ``file_io`` is
    imported by tools that want neither jax nor a device; paying the
    dependency at first use rather than at collection is what keeps this
    module dependency-light, which is half of what makes it MOVABLE (see
    the module docstring).  The staging argument that it might sit in a
    tree without the checkpoint is retired — the checkpoint is landed,
    and this module now depends on names only it publishes.
    """
    if _QS_CACHE:
        return _QS_CACHE[0]
    try:
        import symmetry_maps
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "file_io.mpa_store needs the symmetry_maps service — the "
            "q_irr restart format layer.  The frequency-resolved layout "
            "is that format with a leading ω axis and it shares its "
            "table record, digest, validation and provenance stamp "
            "rather than restating them.  Install/branch onto a tree "
            "that carries it."
        ) from exc
    _QS_CACHE.append(symmetry_maps)
    return symmetry_maps


@contextlib.contextmanager
def _h5(target, mode, *, where=None):
    """THE serial-h5py door onto a store the FFI transport also drives.

    Every h5py open in this module goes through here, with no exception,
    because every file this module writes is ALSO written by ``SlabIO``
    through a different HDF5 library instance (audit A1; sandbox
    claims/0110).  What the door adds over a bare ``QirrDest``:

    * it DECLARES the open to :mod:`file_io.hdf5_owner`, which refuses by
      name if the FFI currently holds a live handle on the same path and
      either side can write — the undefined case, and the one whose
      symptom is a native rank-0 segfault rather than an exception;
    * it FLUSHES before close on a write mode, so the bytes are durable
      before the next collective open reads the superblock;
    * it closes in a ``finally`` — the registry must not be left believing
      a handle is live after an exception, or it would refuse every later
      legitimate open on the path.

    ``target`` is a path (this door owns the handle) or an already-open
    h5py File/Group (the caller owns it; the door still declares it, since
    a live foreign handle is exactly what the registry needs to know
    about).  ``where`` defaults to the calling function's name.

    THE FLUSH IS OWNED-ONLY.  A caller-supplied File/Group is never
    flushed by this door, on any mode — the caller owns the handle and
    therefore owns its durability.  Only the path form gets the flush.

    THE ONE OPEN THAT USED TO BYPASS IT was :func:`read_w_tables`, which
    forwarded straight to ``qirr_store.read_tables`` and let the format
    layer open h5py undeclared — on every rank, in production.  It routes
    through here since 2026-08-15 (audit §E.3 item 2), which is what makes
    the first sentence above a machine-enforced invariant rather than a
    convention; ``test_the_table_read_is_declared_to_the_registry`` is the
    cell that would notice it drifting back.
    """
    from .hdf5_owner import STACK_H5PY, is_write_mode, note_close, note_open

    if where is None:
        try:
            where = "mpa_store." + sys._getframe(2).f_code.co_name
        except Exception:                                   # pragma: no cover
            where = "mpa_store"
    owned = isinstance(target, (str, bytes, os.PathLike))
    if owned:
        path, reg_mode = os.fspath(target), mode
    else:
        handle = getattr(target, "file", None)
        path = getattr(handle, "filename", None)
        # The caller's real mode, not ours: ``QirrDest`` ignores ``mode``
        # on an open group, so claiming "a" for a handle the caller opened
        # "r" would invent a writer.
        reg_mode = getattr(handle, "mode", "r") or "r"
    token = (None if path is None
             else note_open(path, STACK_H5PY, reg_mode, where=where))
    try:
        with _qs().QirrDest(target, mode) as grp:
            try:
                yield grp
            finally:
                if owned and is_write_mode(mode):
                    grp.file.flush()
    finally:
        if token is not None:
            note_close(path, token)


# ---------------------------------------------------------------------------
# The ω grid and its provenance
# ---------------------------------------------------------------------------

def _canonical_sampling(sampling):
    """Validate the protocol record and return it in stamping order.

    Returns a plain dict; there is no record class here on purpose
    (AGENTS.md CONVENTIONS: procedural on plain arrays, no new API
    layers for what a function on numpy arrays does).
    """
    if not isinstance(sampling, dict):
        raise TypeError(
            f"mpa_store: sampling= must be a dict carrying "
            f"{list(_SAMPLING_REQUIRED)}; got "
            f"{type(sampling).__name__}")
    missing = [k for k in _SAMPLING_REQUIRED if k not in sampling]
    if missing:
        raise ValueError(
            f"mpa_store: the sampling protocol record is missing "
            f"{missing}.  The ω grid alone does not say which protocol "
            f"produced it, and a fit whose partition α or pole count "
            f"nobody recorded cannot be reproduced or extended — nested "
            f"partitions are the whole reason growing n_p adds samples "
            f"instead of moving them.")
    varpi = np.ascontiguousarray(sampling["varpi"], dtype=np.float64)
    if varpi.ndim != 1 or varpi.size < 1:
        raise ValueError(
            f"mpa_store: varpi must be the 1-D list of sampling lines in "
            f"Hartree (the double-parallel protocol ships two, ϖ₁ = 0.1 "
            f"and ϖ₂ = 1); got shape {varpi.shape}.")
    if np.any(varpi < 0.0):
        raise ValueError(
            f"mpa_store: varpi carries a negative line offset "
            f"{varpi.tolist()}.  The sampling lines are at +iϖ in the "
            f"upper half plane; a negative one is the wrong branch and "
            f"the fit's forced time ordering would fight it.")
    n_p = int(sampling["n_p"])
    if n_p < 1:
        raise ValueError(f"mpa_store: n_p must be >= 1; got {n_p}")
    alpha = int(sampling["alpha"])
    if alpha < 1:
        raise ValueError(
            f"mpa_store: the partition α must be >= 1 (1 for insulators "
            f"and Na, 2 for Al and Cu); got {alpha}")
    out = {
        "protocol": str(sampling.get("protocol", "double_parallel")),
        "varpi": varpi,
        "n_p": n_p,
        "alpha": alpha,
        "omega_max": float(sampling["omega_max"]),
    }
    extra = {k: v for k, v in sampling.items()
             if k not in _SAMPLING_ORDER}
    return out, extra


def omega_grid_digest(omega, omega_line, sampling):
    """``'sha256:<hex>'`` over the ω grid AND the protocol that made it.

    The grid and the protocol are hashed TOGETHER because either one
    alone is an incomplete identity: two runs can sample the same 2·n_p
    points from different α partitions when n_p is small, and a fit
    extended from a nested partition must know it is extending the same
    chain.  Names go in beside the bytes so two float arrays of the same
    shape cannot be swapped without the digest moving.
    """
    can, _ = _canonical_sampling(sampling)
    w = np.ascontiguousarray(omega, dtype=np.complex128)
    line = np.ascontiguousarray(omega_line, dtype=np.int32)
    h = hashlib.sha256()
    h.update(b"omega")
    h.update(str(w.shape).encode("utf-8"))
    h.update(w.tobytes())
    h.update(b"omega_line")
    h.update(line.tobytes())
    for key in _SAMPLING_ORDER:
        val = can[key]
        h.update(key.encode("utf-8"))
        if isinstance(val, np.ndarray):
            h.update(val.tobytes())
        elif isinstance(val, str):
            h.update(val.encode("utf-8"))
        elif isinstance(val, int):
            h.update(np.int64(val).tobytes())
        else:
            h.update(np.float64(val).tobytes())
    return "sha256:" + h.hexdigest()


def _normalise_grid(omega, omega_line, n_omega):
    w = np.ascontiguousarray(omega, dtype=np.complex128)
    if w.ndim != 1:
        raise ValueError(
            f"mpa_store: omega must be the 1-D list of sampling points; "
            f"got shape {w.shape}")
    if int(w.shape[0]) != int(n_omega):
        raise ValueError(
            f"mpa_store: the ω grid has {int(w.shape[0])} points but the "
            f"tensor's leading axis is {int(n_omega)}.  The leading axis "
            f"IS the frequency axis; a disagreement here means the file "
            f"cannot say which ω a slab was evaluated at, which is the "
            f"one thing the fit stage reads it for.")
    if omega_line is None:
        line = np.zeros(w.shape[0], dtype=np.int32)
    else:
        line = np.ascontiguousarray(omega_line, dtype=np.int32)
    if line.shape != w.shape:
        raise ValueError(
            f"mpa_store: omega_line is {line.shape} but omega is "
            f"{w.shape}; one line label per sampling point.")
    if line.size and (int(line.min()) < 0):
        raise ValueError(
            f"mpa_store: omega_line carries a negative index "
            f"{int(line.min())}; it indexes into varpi.")
    return w, line


# ---------------------------------------------------------------------------
# Write: allocate, fill slab by slab, stamp
# ---------------------------------------------------------------------------

def allocate_w_omega(
    dest,
    name,
    *,
    n_omega,
    n_q_on_disk,
    n_mu,
    tables,
    omega,
    sampling,
    omega_line=None,
    closure_verdict=None,
    dtype=None,
    provenance=None,
    energy_unit="Ha",
    mode="a",
):
    """Create the (n_omega, n_q, N_μ, N_μ) dataset with NO slab ready.

    THE ALLOCATE-THEN-FILL SPLIT IS THE PRODUCER'S SHAPE, not a
    convenience.  The fit's sampling grid is 2·n_p frequencies and the
    screening solve produces them one line-batched sweep at a time; a
    writer that demanded the whole tensor at once would demand exactly
    the memory the owner's constraint says is unavailable.  So the file
    is allocated at full extent, every slab's ledger bit is FALSE, and
    :func:`write_w_slab` flips them one at a time.

    Every ledger bit starts False and that is the load-bearing default.
    An allocated-but-unwritten slab reads back as zeros of exactly the
    right shape — the all-zero-screening hazard, now once per frequency
    — so :func:`read_w_slab` refuses on the ledger and not on a
    heuristic about the data.

    Parameters
    ----------
    dest
        An open ``h5py.File``/``Group``, or a path opened in ``mode``.
    name
        Dataset name, e.g. ``"W_qmunu_omega"``.
    n_omega, n_q_on_disk, n_mu
        Extents.  ``n_mu`` is the LOGICAL centroid count: the μ pad is
        device-count-dependent and never reaches disk.
    tables
        ``symmetry_maps.qirr_store.QirrTables``, already at the logical
        μ extent.  Written into ``<name>__qirr`` by the shared stamp.
    omega
        ``(n_omega,)`` complex — the sampling points z_i in Hartree.
    sampling
        Protocol record; see :data:`_SAMPLING_REQUIRED`.
    omega_line
        ``(n_omega,)`` int — which ``varpi`` line each point sits on.
        Defaults to all-zero (a single-line grid).
    closure_verdict
        ``CentroidClosureVerdict``.  Required, and it REFUSES on a
        non-closed centroid set: a wedge stored against a set with no
        permutation α is silently unrecoverable, per frequency.
    """
    qs = _qs()
    shape, dtype = _w_storage_geometry(
        n_omega, n_q_on_disk, n_mu, dtype, closure_verdict,
        where=f"allocate_w_omega({name!r})")
    n_omega, n_q_on_disk, n_mu = shape[:3]
    with _h5(dest, mode) as grp:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, shape=shape, dtype=dtype)
        return stamp_w_omega(
            grp, name, tables=tables, omega=omega, sampling=sampling,
            omega_line=omega_line, closure_verdict=closure_verdict,
            data_ready=np.zeros(n_omega, dtype=bool),
            provenance=provenance, energy_unit=energy_unit)


def _w_storage_geometry(n_omega, n_q_on_disk, n_mu, dtype,
                        closure_verdict, *, where):
    """Validate one W-frequency allocation before any bytes are created."""
    n_omega = int(n_omega)
    n_q_on_disk = int(n_q_on_disk)
    n_mu = int(n_mu)
    if min(n_omega, n_q_on_disk, n_mu) < 1:
        raise ValueError(
            f"mpa_store: extents must be positive; got n_omega="
            f"{n_omega}, n_q_on_disk={n_q_on_disk}, n_mu={n_mu}")
    dtype = np.dtype(np.complex128 if dtype is None else dtype)
    shape = (n_omega, n_q_on_disk, n_mu, n_mu)
    # THE CLOSURE REFUSAL RUNS BEFORE ANY BYTE IS ALLOCATED.  The stamp
    # below refuses too — it is the same call — but by then a full-size
    # dataset exists, and a refused allocation that leaves a
    # correctly-shaped file behind is precisely the shape of the
    # all-zero-screening hazard this format spends a ledger to avoid.
    if closure_verdict is None:
        raise ValueError(
            f"{where}: closure_verdict= is required.  A wedge "
            "stored against a centroid set that is not orbit-closed has "
            "no permutation α and is unrecoverable — at EVERY "
            "frequency, so the ω axis multiplies the damage rather than "
            "diluting it.  Take one from "
            "symmetry_maps.verify_centroid_orbit_closure.")
    closure_verdict.raise_if_not_closed(f"{where} refuses q_irr storage")
    return shape, dtype


def allocate_w_omega_collective(
    dest,
    name,
    *,
    mesh_xy,
    n_omega,
    n_q_on_disk,
    n_mu,
    tables,
    omega,
    sampling,
    omega_line=None,
    closure_verdict=None,
    dtype=None,
    provenance=None,
    n_rmu_logical=None,
    energy_unit="Ha",
    mode="a",
):
    """Collectively allocate W(z); serial HDF5 writes metadata only.

    The large rank-4 dataset is created through :class:`SlabIO` on every
    process.  After that collective handle closes, rank zero stamps the small
    q-wedge tables, frequency grid, and readiness ledger, then all processes
    synchronize before any slab write begins.

    COLLECTIVE over ``mesh_xy``: every rank calls it, in the same order.

    ``dest`` MUST BE A PATH.  Unlike the serial :func:`allocate_w_omega`,
    which takes an open group or a path, this one hands ``dest`` to
    ``SlabIO``, which ``str()``s it — an h5py Group would be stringified
    into a filename.

    ``omega`` is ``(n_omega,)`` complex, ``omega_line`` ``(n_omega,)``
    int32, ``tables`` a ``QirrTables`` at the LOGICAL μ extent, and
    ``energy_unit`` one of :data:`FIT_ENERGY_UNITS`; the serial twin
    documents each in full.

    RETURNS ``None``, ON EVERY RANK.  It used to hand back rank 0's
    :func:`read_w_header` dict and ``None`` everywhere else — a
    rank-dependent return type, i.e. a mesh divergence waiting for the
    first caller that branches on it (audit §E.3 item 3).  Uniform
    ``None`` rather than a broadcast because nothing consumes it: both
    production call sites (``gw/mpa/model.py``) discard it, and a rank
    that wants the header can call :func:`read_w_header` — one more h5py
    open, which a collective allocator must not pay on every rank for a
    value nobody asked for.

    The open sequence is FFI ``'w'`` → close → rank-0 h5py ``'a'`` →
    barrier: sequential cross-stack alternation with a write on each side,
    which :mod:`file_io.hdf5_owner` counts and which
    ``LORRAX_HDF5_ONE_OWNER=strict`` refuses.
    """
    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    shape, dtype = _w_storage_geometry(
        n_omega, n_q_on_disk, n_mu, dtype, closure_verdict,
        where=f"allocate_w_omega_collective({name!r})")
    with SlabIO(dest, mode=mode, mesh=mesh_xy) as io:
        io.create_dataset(name, shape=shape, dtype=dtype)

    if process_rank() == 0:
        stamp_w_omega(
            dest, name, tables=tables, omega=omega, sampling=sampling,
            omega_line=omega_line, closure_verdict=closure_verdict,
            data_ready=np.zeros(shape[0], dtype=bool),
            n_rmu_logical=n_rmu_logical, provenance=provenance,
            energy_unit=energy_unit)
    barrier("mpa_w_omega_allocated")


def stamp_w_omega(
    dest,
    name,
    *,
    tables,
    omega,
    sampling,
    omega_line=None,
    closure_verdict,
    data_ready=None,
    n_rmu_logical=None,
    provenance=None,
    energy_unit="Ha",
    mode="a",
):
    """Make an ALREADY-WRITTEN 4-D dataset a version-2 W(ω) tensor.

    THE STAMP IS ``qirr_store.stamp_qirr_tensor`` PLUS THE ω GROUP, and
    the split is exactly the same one that function exists for: the GW
    producer does not write its restart tensors with h5py, it writes
    them through ``file_io.slab_io.SlabIO`` where every rank contributes
    its own (μ, ν) hyperslab and no rank ever holds the whole array.  A
    format function that insisted on creating the dataset would force
    that write back through one process.

    THE ONE PLACE THIS CANNOT DELEGATE is the rank.  The landed stamp
    asserts ``ds.ndim == 3``, correctly, because that is what version 1
    is.  A 4-D dataset therefore cannot be handed to it at all, so the
    q_irr attrs are written here against a temporary 3-D VIEW of the
    layout — one frequency slab's worth of shape — and the version attr
    is then overwritten with 2 and the frequency attrs added.  Doing it
    in that order means every v1 attr on a v2 file is written by the v1
    stamp and not by a copy of it, which is the property that keeps the
    two formats from drifting apart attr by attr.
    """
    qs = _qs()
    can = tables.canonical()
    if n_rmu_logical is not None:
        can = can.logical(int(n_rmu_logical)).canonical()

    with _h5(dest, mode) as grp:
        if name not in grp:
            raise KeyError(
                f"mpa_store: {name!r} is not in this file.  "
                f"stamp_w_omega stamps a dataset the caller has already "
                f"written (the SlabIO path); use allocate_w_omega to "
                f"create one.")
        ds = grp[name]
        if ds.ndim != 4:
            raise ValueError(
                f"mpa_store: {name!r} is {ds.shape} (rank {ds.ndim}); "
                f"the frequency-resolved layout is rank 4, "
                f"(n_omega, n_q, N_μ, N_μ).  A rank-3 tensor is a "
                f"version-1 q_irr tensor and belongs to "
                f"qirr_store.stamp_qirr_tensor — stamping it here would "
                f"claim a frequency axis it does not have.")
        if int(ds.shape[2]) != int(ds.shape[3]):
            raise ValueError(
                f"mpa_store: {name!r} has μ extents {ds.shape[2]} x "
                f"{ds.shape[3]}; W_q(μ,ν) is square in the ISDF basis.")
        n_omega = int(ds.shape[0])
        n_q_on_disk = int(ds.shape[1])
        n_mu = int(ds.shape[3])

        w, line = _normalise_grid(omega, omega_line, n_omega)
        san, extra = _canonical_sampling(sampling)
        grid_hash = omega_grid_digest(w, line, san)

        if data_ready is None:
            ready = np.zeros(n_omega, dtype=bool)
        else:
            ready = np.ascontiguousarray(data_ready, dtype=bool)
        if ready.shape != (n_omega,):
            raise ValueError(
                f"mpa_store: the data_ready ledger is {ready.shape} but "
                f"the tensor has {n_omega} frequency slabs.  One bit per "
                f"slab: a ledger that cannot address every slab cannot "
                f"say which of them are data.")

        # THE V1 STAMP, ON A 3-D VIEW.  ``stamp_qirr_tensor`` writes the
        # tables, the digest, the closure record and the provenance —
        # everything the two versions share — and it insists on rank 3,
        # which a v2 dataset is not.  So the shared attrs are taken from
        # a scratch 3-D dataset of ONE SLAB'S shape (zero-filled, never
        # read) and copied across.  The alternative is a second
        # implementation of the stamp, which is a second claim about
        # what the file says, differing on the day one of them gains an
        # attr.
        scratch = name + "__v1stamp_scratch"
        if scratch in grp:
            del grp[scratch]
        grp.create_dataset(scratch, shape=(n_q_on_disk, n_mu, n_mu),
                           dtype=ds.dtype)
        try:
            qs.stamp_qirr_tensor(
                grp, scratch, tables=can,
                closure_verdict=closure_verdict,
                provenance=provenance,
                data_ready=bool(ready.all()))
            src = grp[scratch]
            for key, val in src.attrs.items():
                ds.attrs[key] = val
            tgrp_name = name + qs.QIRR_TABLE_SUFFIX
            if tgrp_name in grp:
                del grp[tgrp_name]
            grp.copy(grp[scratch + qs.QIRR_TABLE_SUFFIX], tgrp_name)
        finally:
            del grp[scratch]
            stale = scratch + qs.QIRR_TABLE_SUFFIX
            if stale in grp:
                del grp[stale]

        # THE VERSION BUMP AND THE AXIS.  Written AFTER the v1 stamp so
        # it overwrites rather than races it.
        ds.attrs[qs.QIRR_VERSION_ATTR] = np.int64(QIRR_FORMAT_VERSION_FREQ)
        ds.attrs[_FREQ_ATTR] = _FREQ_ATTR_VALUE
        ds.attrs["mpa_n_omega"] = np.int64(n_omega)
        unit = str(energy_unit)
        if unit not in FIT_ENERGY_UNITS:
            raise ValueError(
                f"mpa_store: energy_unit must be one of "
                f"{tuple(FIT_ENERGY_UNITS)}, got {energy_unit!r}")
        ds.attrs["mpa_omega_units"] = unit
        # THE SAMPLING ATTRS, in :data:`_SAMPLING_ORDER` — the one list
        # of sampling keys.  ``_canonical_sampling`` already fixed the
        # types (str, float64 array, int, int, float); ints and floats
        # are stamped as fixed-width scalars.
        for key in _SAMPLING_ORDER:
            val = san[key]
            if isinstance(val, int):
                val = np.int64(val)
            elif isinstance(val, float):
                val = np.float64(val)
            ds.attrs["mpa_" + key] = val
        ds.attrs["mpa_grid_hash"] = grid_hash
        ds.attrs["mpa_writer"] = "file_io.mpa_store"
        for key, val in extra.items():
            ds.attrs["mpa_prov_" + str(key)] = val

        mgrp_name = name + MPA_GROUP_SUFFIX
        if mgrp_name in grp:
            del grp[mgrp_name]
        mgrp = grp.create_group(mgrp_name)
        mgrp.create_dataset("omega", data=w)
        mgrp.create_dataset("omega_line", data=line)
        mgrp.create_dataset("data_ready", data=ready)
        mgrp.attrs["grid_hash"] = grid_hash
        return read_w_header(grp, name)


def write_w_slab(dest, name, i_omega, W_q_munu, *, ready=True, mode="a"):
    """Write frequency slab ``i_omega`` and (by default) stamp it ready.

    ``ready=False`` writes the BYTES WITHOUT the ledger bit, which is
    not a curiosity: it is the state a crashed or preempted producer
    leaves behind, and the state the readiness refusal exists to catch.
    It is also how the red twin is constructed without hand-forging a
    file.

    Parameters
    ----------
    W_q_munu
        ``(n_q_on_disk, N_μ, N_μ)`` — the PRE-UNFOLD wedge at this ω,
        at the LOGICAL μ extent.  The same array shape a version-1 q_irr
        tensor holds in its entirety, which is the removability claim
        stated as a signature.
    """
    qs = _qs()
    X = np.asarray(W_q_munu)
    with _h5(dest, mode) as grp:
        ds, mgrp = _open_w(grp, name)
        i = int(i_omega)
        n_omega = int(ds.shape[0])
        if not 0 <= i < n_omega:
            raise IndexError(
                f"mpa_store: frequency index {i} is outside [0, "
                f"{n_omega}) for {name!r}.")
        if X.shape != tuple(int(s) for s in ds.shape[1:]):
            raise ValueError(
                f"mpa_store: slab {i} is {X.shape} but {name!r} holds "
                f"{tuple(int(s) for s in ds.shape[1:])} per frequency.  "
                f"The wedge and the μ extent are the same at every ω — "
                f"the symmetry operation acts on (q, μ, ν) and does not "
                f"touch ω — so a slab of a different shape is not this "
                f"tensor's slab.")
        ds[i] = X
        return _mark_w_slab_ready(ds, mgrp, i, ready)


def _mark_w_slab_ready(ds, mgrp, i_omega, ready):
    """Commit one slab only after its data writer has closed."""
    i = int(i_omega)
    n_omega = int(ds.shape[0])
    if not 0 <= i < n_omega:
        raise IndexError(
            f"mpa_store: frequency index {i} is outside [0, {n_omega}) "
            f"for {ds.name!r}.")
    led = mgrp["data_ready"][()]
    led[i] = bool(ready)
    mgrp["data_ready"][...] = led
    ds.attrs["qirr_data_ready"] = bool(led.all())
    return int(led.sum())


def _require_layout(arr, mesh_xy, spec, where):
    """Refuse unless ``arr`` lies on ``NamedSharding(mesh_xy, spec)``.

    Layout EQUIVALENCE, never spec equality: JAX canonicalizes specs (a
    rank-3 array's trailing replicated entries are dropped, for one), so
    two unequal specs can name ONE placement — and on a one-device mesh
    every placement is the same bytes.  ``where`` is the complete
    refusal message; the caller writes it because only the caller knows
    what the array is and what its producer should have done.
    """
    from jax.sharding import NamedSharding

    expected = NamedSharding(mesh_xy, spec)
    mesh_size = int(np.prod(tuple(int(v) for v in mesh_xy.shape.values())))
    sharding = getattr(arr, "sharding", None)
    if not (isinstance(sharding, NamedSharding)
            and sharding.mesh == mesh_xy
            and (sharding.is_equivalent_to(expected, np.ndim(arr))
                 or mesh_size == 1)):
        raise ValueError(where)


def write_w_slab_collective(
    dest,
    name,
    i_omega,
    W_q_munu,
    *,
    mesh_xy,
    global_shape,
    ready=True,
):
    """Write one native sharded W(z) slab and then commit readiness.

    ``W_q_munu`` remains on its ``P(None, 'x', 'y')`` layout — REQUIRED,
    and enforced by ``_require_layout``.  It is rank 3,
    ``(n_q, n_mu_padded, n_mu_padded)``; SlabIO inserts the singleton
    frequency axis, clips only the device-dependent mu padding against
    ``global_shape`` (which must be the rank-4
    ``(n_omega, n_q, n_mu, n_mu)``), and closes collectively before rank
    zero updates the small readiness ledger.  A failed data write therefore
    leaves the slab explicitly unready.

    COLLECTIVE over ``mesh_xy``.  ``dest`` must be a PATH (SlabIO).

    RETURNS ``None``, ON EVERY RANK — it used to be the new ready count
    on rank 0 and ``None`` elsewhere, the same rank-divergent hazard
    :func:`allocate_w_omega_collective` carried (audit §E.3 item 3).  The
    count is not lost, it is just not returned per slab: it lives in the
    readiness ledger this call commits, and :func:`read_w_header` reports
    it as ``n_ready`` / ``data_ready`` for a rank that asks.
    """
    from jax.sharding import PartitionSpec as P

    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    shape = tuple(int(n) for n in global_shape)
    if len(shape) != 4:
        raise ValueError(
            "write_w_slab_collective: global_shape must be "
            "(n_omega,n_q,n_mu,n_mu)")
    _require_layout(
        W_q_munu, mesh_xy, P(None, "x", "y"),
        "write_w_slab_collective requires W on P(None,'x','y'); "
        "the producer must return the native SlabIO layout rather "
        "than resharding a bulk tensor at the writer seam")
    W4 = W_q_munu[None, ...]
    with SlabIO(dest, mode="a", mesh=mesh_xy) as io:
        io.write_slab(
            name, W4, offset=(int(i_omega), 0, 0, 0),
            global_shape=shape)
    del W4

    if process_rank() == 0:
        with _h5(dest, "a") as grp:
            ds, mgrp = _open_w(grp, name)
            _mark_w_slab_ready(ds, mgrp, i_omega, ready)
    barrier("mpa_w_slab_ready")


def read_w_slab_collective(
    src,
    name,
    i_omega,
    *,
    mesh_xy,
    require_ready=True,
):
    """Read one frequency slab directly into ``P(None,'x','y')``.

    This is the inverse of :func:`write_w_slab_collective`: the file keeps
    the logical centroid extent, SlabIO pads only the two distributed axes,
    and no rank materializes the complete ``(q,mu,nu)`` slab.  The routine
    is valid for any MPA frequency tensor with this layout (in particular
    both ``chi(z)`` and ``Wc(z)``); the historical ``W`` in its name denotes
    the on-disk format, not an extra transport.

    COLLECTIVE over ``mesh_xy``: every rank calls it, in the same order,
    for the same ``i_omega``.  ``src`` must be a PATH (SlabIO).

    RETURNS ``(slab, header)``.  ``slab`` is
    ``(n_q_on_disk, n_mu_padded, n_mu_padded)`` complex128 — the μ extent
    is ``mesh_divisible_shape``'s round-up, NOT ``header['n_mu']``.  The
    4-D read is issued at ``P(None, None, 'x', 'y')`` and the returned 3-D
    array therefore carries ``P(None, 'x', 'y')``; that is inferred from the
    leading index, not asserted on the way out.  ``header`` is
    :func:`read_w_header`'s dict, read with h5py on every rank BEFORE the
    collective handle opens (read-only on both stacks, which the one-owner
    registry allows and counts).
    """
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO, mesh_divisible_shape

    header = read_w_header(src, name)
    i = int(i_omega)
    if not 0 <= i < header["n_omega"]:
        raise IndexError(
            f"read_w_slab_collective: frequency index {i} is outside "
            f"[0,{header['n_omega']}) for {name!r}")
    if require_ready and not bool(header["data_ready"][i]):
        raise ValueError(
            f"read_w_slab_collective: {name!r} slab {i} is allocated but "
            "not ready")

    logical = (1, header["n_q_on_disk"], header["n_mu"], header["n_mu"])
    spec = P(None, None, "x", "y")
    shape = mesh_divisible_shape(logical, mesh_xy, spec)
    with SlabIO(src, mode="r", mesh=mesh_xy) as io:
        slab = io.read_slab(
            name, shape=shape, offset=(i, 0, 0, 0),
            valid_shape=logical, partition_spec=spec)
    return slab[0], header


# ---------------------------------------------------------------------------
# Read: the header, the widened discriminator, the slab, the columns
# ---------------------------------------------------------------------------

def _open_w(grp, name):
    """``(dataset, mpa_group)`` for a stamped W(ω) tensor, or refuse."""
    if name not in grp:
        raise KeyError(f"mpa_store: {name!r} is not in this file")
    mgrp_name = name + MPA_GROUP_SUFFIX
    if mgrp_name not in grp:
        raise ValueError(
            f"mpa_store: {name!r} carries no {mgrp_name!r} group, so the "
            f"file cannot say which frequencies its slabs were evaluated "
            f"at nor which of them are data.  A frequency-resolved "
            f"tensor whose ω grid lives anywhere but beside it is a "
            f"tensor that silently decays when the sampling protocol is "
            f"regenerated.")
    return grp[name], grp[mgrp_name]


def read_w_header(src, name, *, mode="r"):
    """Everything the file CLAIMS about ``name``, reading no tensor data.

    Returns a plain dict with these keys, every one of which some reader
    below indexes by name: ``format_version``, ``freq_axis``, ``n_omega``,
    ``omega`` ``(n_omega,)`` complex128, ``omega_line`` ``(n_omega,)``
    int32, ``omega_units``, ``sampling``, ``grid_hash``, ``data_ready``
    ``(n_omega,)`` bool, ``n_ready``, ``q_storage``, ``n_q_on_disk``,
    ``n_q_full``, ``n_mu``, ``n_rmu_logical``, ``centroid_hash``,
    ``table_hash``, ``closure_verdict``, ``provenance``.

    Rank-local and serial, but called on every rank from three collective
    functions — a collective caller must invoke it uniformly.  ``src`` is a
    path (this call's ``_h5`` owns the handle) or an already-open group (the
    caller owns it, and ``mode`` is then IGNORED by ``QirrDest``: the
    parameter looks live and is not).

    Every cross-check the format owns runs here,
    so a caller that got a header back has already been told the file is
    self-consistent, and every reader below calls this first rather than
    repeating the checks — one implementation of "what does this file
    say", because a second one is how a reader ends up disagreeing with
    the format about what it is holding.
    """
    qs = _qs()
    with _h5(src, mode) as grp:
        ds, mgrp = _open_w(grp, name)
        version = _refuse_unless_rank_matches_version(ds, name)
        if version != QIRR_FORMAT_VERSION_FREQ:
            raise ValueError(
                f"mpa_store: {name!r} is format version {version}; the "
                f"frequency-resolved readers are version "
                f"{QIRR_FORMAT_VERSION_FREQ}.  Use "
                f"qirr_store.read_tensor for a version-1 tensor.")

        # THE PARTIAL-STAMP REFUSAL, version 2's half.  The rank check
        # above settles which format this is; this settles whether the
        # format's own record is whole.  Named rather than left to a
        # KeyError deep in the read, because "which attr is missing" is
        # the question a half-written file raises and a traceback
        # through ``ds.attrs[...]`` answers it one attr at a time.
        absent = [a for a in _MPA_OWNED_ATTRS if a not in ds.attrs]
        if absent:
            raise ValueError(
                f"mpa_store: {name!r} is a version "
                f"{QIRR_FORMAT_VERSION_FREQ} tensor missing {absent}.  "
                f"A half-stamped file is refused rather than read: the "
                f"missing half is the sampling protocol, which is what "
                f"says what the ω values MEAN, and a fit against "
                f"abscissae nobody can characterise is a fit nobody can "
                f"reproduce or extend.")

        n_omega = int(ds.shape[0])
        stamped_n = int(ds.attrs["mpa_n_omega"])
        if stamped_n != n_omega:
            raise ValueError(
                f"mpa_store: {name!r} stamps mpa_n_omega={stamped_n} but "
                f"its leading axis is {n_omega}.  The SHAPE is the "
                f"primary discriminant and the attr is its cross-check, "
                f"so a disagreement is a refusal and not a preference.")

        omega = mgrp["omega"][()]
        line = mgrp["omega_line"][()]
        ready = np.asarray(mgrp["data_ready"][()], dtype=bool)
        for label, arr in (("omega", omega), ("omega_line", line),
                           ("data_ready", ready)):
            if int(np.asarray(arr).shape[0]) != n_omega:
                raise ValueError(
                    f"mpa_store: {name!r} has {n_omega} frequency slabs "
                    f"but its {label} is length "
                    f"{int(np.asarray(arr).shape[0])}.  Each of these is "
                    f"one entry per slab; a short one cannot address "
                    f"every slab and a long one addresses slabs that do "
                    f"not exist.")

        # THE SAMPLING ATTRS, through :data:`_SAMPLING_ORDER` and back
        # through ``_canonical_sampling`` — the same coercions the stamp
        # used, run by the one function that owns them.  (The digest
        # check below already re-canonicalises, so this adds no check
        # that did not run before; it only runs one call earlier.)
        sampling, _ = _canonical_sampling({
            key: (qs.qirr_attr_str(ds, "mpa_" + key) if key == "protocol"
                  else ds.attrs["mpa_" + key])
            for key in _SAMPLING_ORDER})
        recomputed = omega_grid_digest(omega, line, sampling)
        stamped_hash = qs.qirr_attr_str(ds, "mpa_grid_hash")
        if stamped_hash != recomputed:
            raise ValueError(
                f"mpa_store: {name!r} ω-grid hash mismatch.  Stamped "
                f"{stamped_hash}, the grid and protocol on disk hash to "
                f"{recomputed}.  The sampling points are not the ones "
                f"this tensor was evaluated at, so every pole fitted "
                f"from it would be fitted against the wrong abscissae.")

        scalar_ready = ds.attrs.get("qirr_data_ready", None)
        if scalar_ready is not None and bool(scalar_ready) != bool(
                ready.all()):
            raise ValueError(
                f"mpa_store: {name!r} stamps qirr_data_ready="
                f"{bool(scalar_ready)} but its per-frequency ledger has "
                f"{int(ready.sum())} of {n_omega} slabs ready.  The "
                f"scalar is the CONSERVATIVE summary any version-1 "
                f"reader will honour, so it must be all(ledger); a "
                f"disagreement is a file claiming readiness it cannot "
                f"support.")

        # The q_irr half: tables, digest, shape-vs-attr — the landed
        # checks, run against the PER-FREQUENCY extents.
        tables = qs.read_tables(grp, name)
        can = tables.canonical()
        if can.digest() != qs.qirr_attr_str(ds, "qirr_table_hash"):
            raise ValueError(
                f"mpa_store: {name!r} table hash mismatch.  The unfold "
                f"tables are not the ones this tensor was written "
                f"against, so every q it reconstructs — at every ω — "
                f"would be a permutation of the wrong centroids.")
        n_q_on_disk = int(ds.shape[1])
        n_mu = int(ds.shape[3])
        shape_says = qs.validate_qirr_tables(can, n_q_on_disk, n_mu)
        attr_says = qs.qirr_attr_str(ds, "q_storage")
        if attr_says != shape_says:
            raise ValueError(
                f"mpa_store: {name!r} shape says q_storage="
                f"{shape_says!r} ({n_q_on_disk} q rows per frequency "
                f"against {can.n_q_full} full-BZ rows in the tables) but "
                f"the attr says {attr_says!r}.  The SHAPE is the primary "
                f"discriminant and the attr is its cross-check, so a "
                f"disagreement is a refusal.")

        prov = {k[len("prov_"):]: v for k, v in ds.attrs.items()
                if str(k).startswith("prov_")}
        for key in ("qirr_generator_commit", "qirr_written_utc",
                    "qirr_writer", "mpa_writer"):
            if key in ds.attrs:
                prov[key] = qs.qirr_attr_str(ds, key)
        return {
            "format_version": version,
            "freq_axis": qs.qirr_attr_str(ds, _FREQ_ATTR),
            "n_omega": n_omega,
            "omega": omega,
            "omega_line": line,
            "omega_units": qs.qirr_attr_str(ds, "mpa_omega_units"),
            "sampling": sampling,
            "grid_hash": recomputed,
            "data_ready": ready,
            "n_ready": int(ready.sum()),
            "q_storage": shape_says,
            "n_q_on_disk": n_q_on_disk,
            "n_q_full": can.n_q_full,
            "n_mu": n_mu,
            "n_rmu_logical": int(ds.attrs["qirr_n_rmu_logical"]),
            "centroid_hash": qs.qirr_attr_str(ds, "qirr_centroid_hash"),
            "table_hash": can.digest(),
            "closure_verdict": qs.qirr_attr_str(ds, "qirr_closure_verdict"),
            "provenance": prov,
        }


def _refuse_unless_rank_matches_version(ds, name):
    """THE DISCRIMINANT, for the versions THIS module owns.

    Rank and version must agree or the file is refused, because a
    version-1 reader takes ``ds.shape[0]`` as the q extent and
    ``ds.shape[-1]`` as the μ extent.  Hand it a
    ``(n_omega, n_q_ibz, N_μ, N_μ)`` dataset and both of those
    expressions still evaluate — ``shape[-1]`` is genuinely N_μ, and
    ``shape[0]`` is n_omega, which the table validation compares against
    the number of IBZ rows.  When ``n_omega == n_q_ibz`` that comparison
    PASSES.  So do the q_storage cross-check, the table digest, the
    n_q_full stamp and the readiness flag.  Nothing refuses, and the
    caller receives a 4-D array it believes is 3-D with the frequency
    axis relabelled as q.  Si 4³ reduces 64 q to 8 and an n_p = 4 fit
    samples 8 frequencies, so the coincidence is one deck away.

    THE VERSION-1 HALF OF THAT CHECK IS NO LONGER HERE, and its removal
    is the point rather than a simplification.  This function used to
    enforce rank 3 under a version-1 stamp, and the since-deleted
    dispatcher registered the reason as a follow-up in as
    many words: a wrapper protects the callers who use it and nobody
    else, and the hazard is worst precisely for a caller who does not
    know the new layout exists.  ``qirr_store.read_tensor`` now runs that
    refusal itself, above ``read_tables`` and before any extent is
    believed, so a consumer that never heard of the frequency axis is
    protected by the version-1 reader it was already calling.  Repeating
    it here would be a second, weaker copy of a check that has found its
    home — and the copy would be the one to go stale.

    What stays is what only this module can know.  Version 2 is not a
    version ``qirr_store`` has heard of, so its rank-4 requirement is
    ours to state and ours to enforce; and the ``mpa_freq_axis`` attr is
    ours in both directions, which is why the presence cross-check below
    runs on a version-1 file too.  A v1 file carrying the frequency attr
    is a half-stamp, and the missing half is exactly what would say
    whether the shape means what it looks like.
    """
    qs = _qs()
    if qs.QIRR_VERSION_ATTR not in ds.attrs:
        raise ValueError(
            f"mpa_store: {name!r} carries no {qs.QIRR_VERSION_ATTR!r}.  "
            f"'No attrs' is read as q_storage='full' for backward "
            f"compatibility by qirr_store.read_tensor, which is the "
            f"reader for that case; the frequency-resolved layout is "
            f"never legacy.")
    version = int(ds.attrs[qs.QIRR_VERSION_ATTR])
    if version not in QIRR_FORMAT_VERSIONS_READABLE:
        raise ValueError(
            f"mpa_store: {name!r} is format version {version}; this "
            f"reader knows {list(QIRR_FORMAT_VERSIONS_READABLE)}.  "
            f"Refusing rather than guessing.")
    # THE RANK, FOR VERSION 2 ONLY.  Version 1's rank is
    # ``qirr_store.read_tensor``'s refusal now — it runs it before it
    # believes any extent, so the caller this module dispatches to has
    # already made the check by the time it returns.  See the docstring.
    want = _rank_by_version()[version]
    if version == QIRR_FORMAT_VERSION_FREQ and int(ds.ndim) != want:
        raise ValueError(
            f"mpa_store: {name!r} stamps qirr_format_version={version}, "
            f"which is rank {want}, but the dataset is {ds.shape} — rank "
            f"{int(ds.ndim)}.  RANK IS THE DISCRIMINANT and the version "
            f"attr is its cross-check, in that order.  Version "
            f"{QIRR_FORMAT_VERSION_FREQ} is the frequency-resolved layout "
            f"and its leading axis is ω; a tensor stamped for it that is "
            f"not rank {want} has either lost that axis or never had it, "
            f"and reading it would hand the caller a q axis relabelled as "
            f"frequency.  That is silent corruption, so it is refused on "
            f"a property of the bytes rather than of an attr.")
    has_freq_attr = _FREQ_ATTR in ds.attrs
    if has_freq_attr != (version == QIRR_FORMAT_VERSION_FREQ):
        raise ValueError(
            f"mpa_store: {name!r} is version {version} but "
            f"{_FREQ_ATTR!r} is "
            f"{'present' if has_freq_attr else 'absent'}.  The attr "
            f"marks the leading frequency axis and belongs to version "
            f"{QIRR_FORMAT_VERSION_FREQ} exactly; a half-stamp is "
            f"refused rather than read, because the missing half is "
            f"what would say whether the shape means what it looks "
            f"like.")
    if has_freq_attr and qs.qirr_attr_str(ds, _FREQ_ATTR) != _FREQ_ATTR_VALUE:
        raise ValueError(
            f"mpa_store: {name!r} stamps {_FREQ_ATTR}="
            f"{qs.qirr_attr_str(ds, _FREQ_ATTR)!r}; this format's frequency "
            f"axis is {_FREQ_ATTR_VALUE!r} — axis 0 — and nothing else "
            f"has been defined.")
    return version


def read_w_slab(
    src,
    name,
    i_omega,
    *,
    q=None,
    unfold=False,
    mesh_xy=None,
    n_mu_padded=None,
    require_ready=True,
    mode="r",
):
    """One frequency slab: ``(n_q, N_μ, N_μ)``, or one q of it.

    THE REMOVABILITY CLAIM, AS A FUNCTION.  What comes back for
    ``unfold=False`` and ``n_mu_padded=None`` is bit-identical to what
    ``qirr_store.read_tensor`` returns from a version-1 file written
    from this slab — same bytes, same wedge, same tables.  That is the
    whole content of "the leading dimension is removable later": the
    axis is a container, not a change of meaning, and dropping it is a
    slice rather than a migration.  ``test_the_leading_axis_is_
    removable`` asserts it attr-for-attr.

    Parameters
    ----------
    q
        Optional q index into the stored wedge.  ``None`` returns every
        stored q at this ω.
    unfold
        Unfold the wedge to the full BZ AT THIS FREQUENCY.  The tables
        are ω-independent, so this is ``unfold_isdf_operator`` on the
        slab — the same call, the same arguments, one frequency at a
        time.  Needs ``mesh_xy``.
    n_mu_padded
        Re-apply a μ pad of the READER's own width.  The file stores the
        LOGICAL extent, so a consumer that wants the padded in-memory
        layout asks for it here rather than finding the writer's pad and
        hoping it matches.
    require_ready
        Refuse when this slab's ledger bit is False.  Default True.
    """
    qs = _qs()
    header = read_w_header(src, name, mode=mode)
    i = int(i_omega)
    n_omega = header["n_omega"]
    if not 0 <= i < n_omega:
        raise IndexError(
            f"mpa_store: frequency index {i} is outside [0, {n_omega}) "
            f"for {name!r}.")
    if require_ready and not bool(header["data_ready"][i]):
        raise ValueError(
            f"mpa_store: {name!r} frequency slab {i} (ω = "
            f"{header['omega'][i]}) is PRESENT AND CORRECTLY SHAPED but "
            f"its data_ready bit is False — it is allocated space, not "
            f"data.  {header['n_ready']} of {n_omega} slabs are ready.  "
            f"Reading it would hand the fit a slab of zeros that passes "
            f"every shape check, which is the mechanism behind the "
            f"all-zero-screening incident: a plausible excitonic "
            f"spectrum out of a W that was never written.  A "
            f"frequency-resolved file reaches this state routinely — "
            f"the producer fills ω one line-batched sweep at a time — "
            f"so the ledger is per slab and not per file.  Pass "
            f"require_ready=False to inspect the placeholder "
            f"deliberately.")

    with _h5(src, mode) as grp:
        ds = grp[name]
        raw = ds[i] if q is None else ds[i, int(q)]
    raw = np.asarray(raw)

    # THE TABLES ARE READ ONLY WHEN THEY ARE NEEDED, which is the
    # unfold and the re-pad.  The production per-slab read is neither —
    # a consumer walking ω takes the wedge as stored — and opening the
    # table group on every one of those would be a second file open per
    # frequency for arrays nobody looks at.  Their DIGEST was already
    # checked by ``read_w_header`` above, so this is a saved read and
    # not a skipped check.
    if not unfold and n_mu_padded is None:
        return raw, header

    tables = read_w_tables(src, name, mode=mode)
    can = tables.canonical()
    if n_mu_padded is not None and int(n_mu_padded) != int(can.n_mu):
        pad = int(n_mu_padded) - int(can.n_mu)
        if pad < 0:
            raise ValueError(
                f"mpa_store: {name!r} stores {can.n_mu} logical "
                f"centroids and the caller asked to pad DOWN to "
                f"{n_mu_padded}.  The pad only ever grows the extent; a "
                f"smaller request means the caller and the file "
                f"disagree about the centroid set.")
        widths = [(0, 0)] * (raw.ndim - 2) + [(0, pad), (0, pad)]
        raw = np.pad(raw, widths)
        can = can.padded(int(n_mu_padded))

    if not unfold or header["q_storage"] == "full":
        return raw, header
    if q is not None:
        raise ValueError(
            f"mpa_store: {name!r} cannot unfold a single stored q "
            f"(q={q}).  The unfold gathers every full-BZ row from its "
            f"IBZ parent, so it needs the whole wedge at this ω; ask "
            f"for q=None and index the result.")
    if mesh_xy is None:
        raise ValueError(
            f"mpa_store: {name!r} is stored on the q wedge "
            f"({header['n_q_on_disk']} of {header['n_q_full']} q) and "
            f"unfolding slab {i} needs a mesh; pass mesh_xy= or "
            f"unfold=False to take the wedge.")
    import jax.numpy as jnp
    # THROUGH THE SERVICE'S DOOR, not past it: the top-level package,
    # never ``symmetry_maps.maps``.  Reaching a submodule is what stops
    # a service being replaceable, and ``test_layering`` enforces it.
    from symmetry_maps import unfold_isdf_operator
    # Test seam only — the bulk to-device transfer below has zero
    # production callers; production unfolds sharded in _finish_pole_read.
    full = unfold_isdf_operator(
        jnp.asarray(raw),
        irr_idx=can.irr_idx_q,
        sym_idx=can.sym_idx_q,
        sym_perm=can.sym_perm,
        L_table=can.L_table,
        q_irr_frac=can.q_irr_frac,
        mesh_xy=mesh_xy,
        n_sym_spatial=int(can.n_sym_spatial),
    )
    return full, header


def read_w_tables(src, name, *, mode="r"):
    """The stored unfold tables — ``qirr_store.read_tables``, unchanged.

    Re-exported rather than reimplemented, and named here so a caller
    reading a v2 file does not have to know which module owns the table
    group.  They are ω-INDEPENDENT: one set for the whole frequency
    axis, because the symmetry operation acts on (q, μ, ν).

    THROUGH :func:`_h5`, and that is the whole content of this wrapper.
    ``qirr_store.read_tables`` opens h5py itself, so forwarding ``src``
    to it was an h5py open the ownership registry could not see — the one
    blind spot in this module's one-owner invariant, and not a rare one:
    it runs on EVERY RANK in production (``gw/mpa/fit_driver.py``'s
    unfold-table read).  The door takes the open and hands the already-open
    group on, so the format layer still owns the reading and the registry
    still owns the counting (audit §E.3 item 2).
    """
    with _h5(src, mode) as grp:
        return _qs().read_tables(grp, name)


# ---------------------------------------------------------------------------
# The column budget, and the 1-D-sharded column read
# ---------------------------------------------------------------------------

def one_tile_bytes(n_mu, itemsize=COMPLEX128_BYTES):
    """Bytes in ONE (N_μ, N_μ) tile — the unit the constraint is in.

    The owner's memory constraint is stated in tiles: a small number of
    W_q(μ,ν) copies fit at once.  Everything the fit stage holds is
    priced against this, so it has a name.
    """
    return int(n_mu) * int(n_mu) * int(itemsize)


def choose_column_budget(n_mu, n_omega, tile_bytes=None,
                         itemsize=COMPLEX128_BYTES):
    """How many ν columns may be read ACROSS ALL ω for one tile's cost.

    THE ARITHMETIC, and it is deliberately a closed form rather than a
    heuristic.  A column block spanning every frequency costs

        n_omega * N_mu * n_cols * itemsize        bytes

    (one full row axis, ``n_cols`` columns, all of ω) and the budget is
    one (N_μ, N_μ) tile,

        N_mu * N_mu * itemsize                    bytes

    so

        n_cols = min(N_mu,
                     max(1, floor(tile_bytes / (n_omega * N_mu * itemsize))))

    which for the default budget collapses to ``n_cols = N_mu //
    n_omega`` — the frequency axis is paid for out of the column count,
    one for one.  At the Si production scale, N_μ = 480 and n_omega = 16
    (an n_p = 8 fit's 2·n_p samples): the tile is 480·480·16 =
    3 686 400 B (3.52 MiB), the per-column cost across ω is 16·480·16 =
    122 880 B, and the budget is exactly **30 columns** — 16·480·30·16 =
    3 686 400 B, the tile to the byte.

    BOTH CLAMPS ARE IN THE FORMULA ABOVE, and both bite in practice.  The
    LOWER one — a budget of zero columns is not a budget, it is a refusal
    to make progress, and the honest failure for a grid so long that one
    column busts a tile is to hand back 1 and let the caller see the cost
    in :func:`describe_column_cost`.  The UPPER one bites whenever
    ``tile_bytes`` is generous: measured 2026-08-15,
    ``choose_column_budget(480, 16)`` is 30 as the worked example says,
    but ``choose_column_budget(480, 16, 100*2**20)`` returns **480** — the
    whole row extent — where the unclamped ratio would give 853.  There
    are only ``n_mu`` columns to read.  The clamp used to be absent from
    the published form, so a caller that sized an ``n_cols_buffer`` off it
    and passed a large budget was refused by
    :func:`read_w_columns_collective` (audit §E.3 item 5).

    Parameters
    ----------
    n_mu
        LOGICAL centroid count — the row extent of the block.
    n_omega
        Frequencies read at once.  This is the whole grid: the point of
        the leading axis is that a fit reads all of ω for a few columns,
        never all columns for a few ω.
    tile_bytes
        Budget.  Defaults to :func:`one_tile_bytes`.  Pass a larger one
        to spend more deliberately; the reader will say so in its
        refusal either way.
    """
    n_mu = int(n_mu)
    n_omega = int(n_omega)
    if n_mu < 1 or n_omega < 1:
        raise ValueError(
            f"choose_column_budget: n_mu and n_omega must be positive; "
            f"got n_mu={n_mu}, n_omega={n_omega}")
    budget = one_tile_bytes(n_mu, itemsize) if tile_bytes is None \
        else int(tile_bytes)
    if budget < 1:
        raise ValueError(
            f"choose_column_budget: tile_bytes must be positive; got "
            f"{budget}")
    per_col = n_omega * n_mu * int(itemsize)
    return max(1, min(n_mu, budget // per_col))


def describe_column_cost(n_mu, n_omega, n_cols, tile_bytes=None,
                         itemsize=COMPLEX128_BYTES):
    """The budget arithmetic as a sentence, for refusals and for logs.

    Separate from the refusal so the same numbers can be printed by a
    driver that is deciding rather than failing — a message a caller can
    only see by triggering an exception is a message that gets read once.

    THE ARITHMETIC IS THE ONE THAT RAN.  The budget half prints the tile
    product only when the budget IS one tile; a caller-supplied
    ``tile_bytes`` prints as the number it is, because
    ``n_mu*n_mu*itemsize = <a different number>`` is a false equation and
    a false equation in a refusal costs more than the refusal saves.  The
    ``choose_column_budget(...)`` call it ends with is ECHOED IN FULL, so
    a reader who types it gets the number that was just printed rather
    than the default-budget one (audit §E.3 item 4).
    """
    n_mu = int(n_mu)
    n_omega = int(n_omega)
    n_cols = int(n_cols)
    itemsize = int(itemsize)
    budget = one_tile_bytes(n_mu, itemsize) if tile_bytes is None \
        else int(tile_bytes)
    cost = n_omega * n_mu * n_cols * itemsize
    allowed = choose_column_budget(n_mu, n_omega, tile_bytes, itemsize)
    budget_terms = (f"one (N_mu, N_mu) tile, {n_mu}*{n_mu}*{itemsize} B = "
                    if tile_bytes is None else "")
    return (
        f"{n_cols} columns at n_omega={n_omega}, N_mu={n_mu} costs "
        f"{n_omega}*{n_mu}*{n_cols}*{itemsize} B = {cost} B "
        f"({cost / 2 ** 20:.2f} MiB) against a budget of "
        f"{budget_terms}{budget} B "
        f"({budget / 2 ** 20:.2f} MiB) — a ratio of "
        f"{cost / budget:.3f}x.  choose_column_budget({n_mu}, "
        f"{n_omega}, tile_bytes={budget}, itemsize={itemsize}) allows "
        f"{allowed}.")


def normalise_columns(mu_cols, n_mu):
    """Columns as a sorted, unique, in-range int64 array, or refuse.

    Sorted and unique because the read below hands them to HDF5 as a
    point selection, which requires increasing order, and because a
    duplicated column is a column the fit would solve twice and write
    twice — the second write racing the first in the staged store.
    """
    cols = np.atleast_1d(np.asarray(mu_cols))
    if cols.dtype == bool:
        raise TypeError(
            "mpa_store: mu_cols must be indices, not a boolean mask; a "
            "mask hides the column COUNT, which is the quantity the "
            "budget is about.")
    cols = cols.astype(np.int64, copy=False)
    if cols.ndim != 1:
        raise ValueError(
            f"mpa_store: mu_cols must be 1-D; got shape {cols.shape}")
    if cols.size == 0:
        raise ValueError("mpa_store: mu_cols is empty")
    uniq, counts = np.unique(cols, return_counts=True)
    if uniq.size != cols.size:
        # The REPEATED ones, not the first four distinct ones: the label
        # says "repeats a column", so the numbers beside it have to be the
        # columns that repeat or the message sends the reader to innocent
        # indices (audit §E.3 item 11).
        dup = uniq[counts > 1].tolist()
        raise ValueError(
            f"mpa_store: mu_cols repeats a column "
            f"({cols.size} given, {uniq.size} distinct, e.g. "
            f"{dup[:4]}).  A repeated column is fitted twice and "
            f"written twice into the staged store, where the second "
            f"write silently overwrites the first.")
    if int(uniq[0]) < 0 or int(uniq[-1]) >= int(n_mu):
        raise IndexError(
            f"mpa_store: mu_cols spans [{int(uniq[0])}, "
            f"{int(uniq[-1])}] but the tensor has {int(n_mu)} logical "
            f"columns.")
    return uniq


def _refuse_two_dim_sharding(spec, where):
    """A row-axis-only spec, or refuse by name.

    THE BLOCK IS 1-D SHARDED, ROW AXIS ONLY, and this is where that is
    enforced rather than documented.  The fit is elementwise in (μ, ν):
    every column's poles are solved independently, so a second split on
    the column axis buys no parallelism the column loop does not
    already have, while making each rank's column count a function of
    the mesh shape — and the column count is exactly the quantity
    :func:`choose_column_budget` sized against a tile.  A 2-D sharding
    turns a budget the caller computed into a budget the mesh computed.

    ``spec`` is a ``PartitionSpec``, a plain tuple of the same shape, or
    a ``NamedSharding`` (its ``.spec`` is taken).  ``None`` means
    unsharded and is always allowed.
    """
    if spec is None:
        return None
    spec = getattr(spec, "spec", spec)
    parts = tuple(spec)
    if len(parts) != 3:
        raise ValueError(
            f"{where}: the returned block is (n_omega, N_mu_rows, "
            f"n_cols) — rank 3 — so its sharding spec must have three "
            f"entries; got {parts!r}.")
    named = [i for i, p in enumerate(parts) if p is not None]
    if named == [1]:
        return parts
    raise ValueError(
        f"{where}: the column block is 1-D SHARDED ON THE ROW AXIS "
        f"ONLY (axis 1), never 2-D; got {parts!r}, which names "
        f"{[('omega', 'row', 'col')[i] for i in named]}.  The fit is "
        f"elementwise in (μ, ν), so splitting the column axis as well "
        f"buys no parallelism the column loop does not already have "
        f"while making each rank's column count a function of the mesh "
        f"shape — and the column count is the quantity the tile budget "
        f"is computed against.  Shard the rows, loop the columns.")


def _column_span(cols):
    """``(lo, hi, sel)`` for a normalised column set.

    ``[lo, hi)`` is the covering span; ``sel`` is ``None`` when the set
    is contiguous (one HDF5 hyperslab) and the explicit point-selection
    list when it is not.
    """
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    if (hi - lo) == int(cols.size):
        return lo, hi, None
    return lo, hi, cols.tolist()


def _validate_column_request(header, q, mu_cols, tile_bytes, where,
                             *, require_ready=True):
    """The refusal set both all-frequency column readers share.

    Returns ``(iq, cols, budget)`` — the checked q index, the
    normalised columns and the tile-budget column count — or refuses:
    over budget, frequency slabs missing, or q out of range, in that
    order, each naming its arithmetic.
    """
    n_mu = int(header["n_mu"])
    n_omega = int(header["n_omega"])
    cols = normalise_columns(mu_cols, n_mu)
    budget = choose_column_budget(n_mu, n_omega, tile_bytes)
    if int(cols.size) > budget:
        raise ValueError(
            f"{where}: refusing {int(cols.size)} columns.  " +
            describe_column_cost(n_mu, n_omega, int(cols.size),
                                 tile_bytes) +
            f"  A block spanning all {n_omega} frequencies is priced to "
            f"be ONE of the small number of W_q(μ,ν) copies that fit at "
            f"once; pass tile_bytes= to raise the budget deliberately, "
            f"or loop the columns in blocks of {budget}.")
    if require_ready and int(header["n_ready"]) != n_omega:
        missing = np.flatnonzero(
            ~np.asarray(header["data_ready"], dtype=bool))
        raise ValueError(
            f"{where}: {len(missing)} of {n_omega} "
            f"frequency slabs are not ready (indices "
            f"{missing[:8].tolist()}{'...' if len(missing) > 8 else ''})."
            f"  A column block spans the WHOLE frequency axis, so a fit "
            f"run on the ready half of the grid returns poles that are "
            f"wrong rather than absent — the unwritten slabs read as "
            f"zeros and the Padé solve happily fits them.  Fill the "
            f"grid, or pass require_ready=False to inspect it.")
    iq = int(q)
    if not 0 <= iq < int(header["n_q_on_disk"]):
        raise IndexError(
            f"{where}: q={iq} is outside [0, "
            f"{int(header['n_q_on_disk'])}); the tensor is stored on "
            f"the {header['q_storage']} q axis.")
    return iq, cols, budget


def read_w_columns(
    src,
    name,
    q,
    mu_cols,
    *,
    tile_bytes=None,
    n_mu_padded=None,
    out_spec=None,
    require_ready=True,
    mode="r",
):
    """A few ν columns of W_q, ACROSS ALL FREQUENCIES.

    Returns ``(n_omega, N_μ_rows, len(mu_cols))`` complex — the shape
    the per-element plasmon-pole fit consumes.  This is the read the
    leading frequency axis exists for: the fit needs all of ω for one
    (μ, ν) element and never needs all of (μ, ν) for one ω, so the
    frequency axis is the OUTER one on disk and the innermost one in the
    solve.

    THE BUDGET REFUSES BY NAME.  ``len(mu_cols)`` is checked against
    :func:`choose_column_budget` and a request that busts it raises with
    the full arithmetic — the per-column cost, the total, the tile it is
    measured against, the ratio, and the count that would have fit.
    Silently truncating or silently allowing would each defeat the
    constraint the number encodes: a small number of W_q(μ,ν) copies fit
    at once, and this block is priced to be one of them.

    THE SHARDING IS 1-D ON THE ROW AXIS.  ``out_spec`` is checked, not
    applied — the read itself is host-side h5py and the placement is the
    caller's — but a 2-D spec is refused here rather than downstream,
    because by the time it is downstream the column count is no longer
    the number the budget was computed for.  See
    :func:`_refuse_two_dim_sharding`.

    REQUIRES EVERY SLAB.  The block spans the whole frequency axis, so
    every ω must be ready; a partially filled file refuses and names how
    many slabs are missing.  That is stricter than :func:`read_w_slab`
    on purpose — a fit run on the ready half of a grid produces poles
    that are wrong rather than absent.
    """
    qs = _qs()
    header = read_w_header(src, name, mode=mode)
    _refuse_two_dim_sharding(out_spec, f"read_w_columns({name!r})")
    n_mu = header["n_mu"]
    iq, cols, _ = _validate_column_request(
        header, q, mu_cols, tile_bytes, f"read_w_columns({name!r})",
        require_ready=require_ready)

    # ONE HYPERSLAB, NOT ONE PER FREQUENCY.  A contiguous run becomes a
    # slice (HDF5 reads it as a single hyperslab); anything else is a
    # point selection on the LAST axis only, which h5py supports and
    # which keeps the row axis whole — the axis the caller shards.
    lo, hi, sel = _column_span(cols)
    with _h5(src, mode) as grp:
        block = grp[name][:, iq, :, slice(lo, hi) if sel is None else sel]
    block = np.asarray(block)

    if n_mu_padded is not None and int(n_mu_padded) != n_mu:
        pad = int(n_mu_padded) - n_mu
        if pad < 0:
            raise ValueError(
                f"read_w_columns({name!r}): the file stores {n_mu} "
                f"logical centroids and the caller asked to pad the row "
                f"axis DOWN to {n_mu_padded}.  The pad only ever grows "
                f"the extent.")
        # ROWS ONLY.  The columns are a selection the caller chose, not
        # an axis with a pad; padding them would invent centroids the
        # caller did not ask for and shift every index in ``mu_cols``.
        block = np.pad(block, ((0, 0), (0, pad), (0, 0)))
    return block


def read_w_columns_collective(
    src,
    name,
    q,
    mu_cols,
    *,
    mesh_xy,
    n_cols_buffer,
    tile_bytes=None,
    header=None,
):
    """Collectively read one all-frequency column tile, sharded on rows.

    The returned array is ``(n_omega, 1, n_mu_padded, n_cols_buffer)``
    with ``P(None, None, ('x', 'y'), None)``.  The singleton is the stored
    q axis; retaining it makes the SlabIO offset and the fit-store write
    geometry identical.  A short final column block is zero-filled to the
    fixed buffer width and ``valid_shape`` prevents those zeros from reading
    bytes belonging to the next block.
    """
    from jax.sharding import PartitionSpec as P

    from file_io.slab_io import SlabIO, mesh_divisible_shape

    hdr = read_w_header(src, name) if header is None else header
    n_mu = int(hdr["n_mu"])
    n_omega = int(hdr["n_omega"])
    iq, cols, budget = _validate_column_request(
        hdr, q, mu_cols, tile_bytes,
        f"read_w_columns_collective({name!r})")
    lo, hi, sel = _column_span(cols)
    if sel is not None:
        raise ValueError(
            "read_w_columns_collective requires one contiguous column "
            "range; the production fit schedule emits contiguous blocks")
    width = int(n_cols_buffer)
    if width < int(cols.size) or width > budget:
        raise ValueError(
            f"read_w_columns_collective: n_cols_buffer={width}, actual "
            f"width={int(cols.size)}, budget={budget}; require "
            "actual <= buffer <= budget")

    spec = P(None, None, ("x", "y"), None)
    shape = mesh_divisible_shape(
        (n_omega, 1, n_mu, width), mesh_xy, spec)
    with SlabIO(src, mode="r", mesh=mesh_xy) as io:
        return io.read_slab(
            name, shape=shape, offset=(0, iq, 0, lo),
            valid_shape=(n_omega, 1, n_mu, int(cols.size)),
            partition_spec=spec)


# ---------------------------------------------------------------------------
# The staged B/Ω fit store
# ---------------------------------------------------------------------------

def _initialise_fit_metadata(
    grp, *, n_q, n_mu, n_p, energy_unit, grid_hash, table_hash,
    centroid_hash, unfold_tables, provenance, diagnostic_keys=None,
    preserve_payload=False, occupation_state=None,
):
    """Rank-zero-only small metadata half of fit-store allocation."""
    qs = _qs()
    reset = [MPA_FIT_SUFFIX, MPA_HEAD_SUFFIX]
    if not preserve_payload:
        reset = ["Omega_p", "B_p"] + [
            k for k in grp if str(k).startswith("fit_")] + reset
    for key in reset:
        if key in grp:
            del grp[key]
    led = grp.create_group(MPA_FIT_SUFFIX)
    led.create_dataset("blocks_done", data=np.zeros((n_q, n_mu), dtype=bool))
    led.create_dataset("block_journal", shape=(0, 3), maxshape=(None, 3),
                       dtype=np.int64)
    for key in REQUIRED_DIAGNOSTICS:
        led.create_dataset("block_" + key + "_max", shape=(0,),
                           maxshape=(None,), dtype=np.float64)
    if diagnostic_keys is not None:
        led.attrs["diagnostic_keys"] = ",".join(sorted(diagnostic_keys))

    grp.attrs["mpa_fit_format_version"] = np.int64(MPA_FIT_FORMAT_VERSION)
    if energy_unit is not None:
        grp.attrs[FIT_ENERGY_UNIT_ATTR] = str(energy_unit)
    grp.attrs["mpa_fit_n_p"] = np.int64(n_p)
    grp.attrs["mpa_fit_n_q"] = np.int64(n_q)
    grp.attrs["mpa_fit_n_mu_logical"] = np.int64(n_mu)
    grp.attrs["mpa_fit_complete"] = False
    grp.attrs["mpa_fit_writer"] = "file_io.mpa_store"
    grp.attrs["mpa_fit_generator_commit"] = qs.qirr_generator_commit()
    grp.attrs["mpa_fit_allocated_utc"] = _utc_now()
    for label, val in (("grid_hash", grid_hash), ("table_hash", table_hash),
                       ("centroid_hash", centroid_hash)):
        if val is not None:
            grp.attrs["mpa_fit_w_" + label] = str(val)
    for key, val in (provenance or {}).items():
        grp.attrs["prov_" + str(key)] = val
    if occupation_state is not None:
        stamp_occupation_provenance(grp, occupation_state)
    if unfold_tables is not None:
        stamp_fit_unfold_tables(grp, unfold_tables)

def allocate_fit_store(
    dest,
    *,
    n_q,
    n_mu,
    n_p,
    energy_unit=None,
    grid_hash=None,
    table_hash=None,
    centroid_hash=None,
    unfold_tables=None,
    dtype=None,
    provenance=None,
    occupation_state=None,
    mode="a",
):
    """Create the staged B_q / Ω_q store with an EMPTY completion ledger.

    WHY STAGED AT ALL.  The fit is per element and reads sub-tiles: the
    driver walks column blocks of ``choose_column_budget`` width, solves
    each block's Padé-in-z² systems, and moves on.  Holding every
    block's poles until the last one finished would hold ``2·n_p``
    tensors of (N_μ, N_μ) — the fit's output is LARGER than its input
    when n_p > 1 — so results go to disk as they complete and the file
    is the working set.

    WHICH MEANS THE FILE IS INCOMPLETE FOR MOST OF ITS LIFE, and that is
    the state the ledger exists to make legible.  ``blocks_done`` is one
    bool per (q, column); ``block_journal`` is the append-only record of
    which column RANGE of which q was written when, with that block's
    condition and backward error beside it.  A reader refuses an
    unfinalized file unless it asks for partial ANNOUNCED — see
    :func:`read_fit_block`.

    Parameters
    ----------
    n_p
        Poles per element.  Si is 8 (scan 6–12), hBN and TiO₂ 10–11, Al
        and Na 8, Cu 12 (see the authoritative MPA theory chapter).
    grid_hash, table_hash, centroid_hash
        The W(ω) file's stamps, carried here so the Σ stage can assert
        that these poles came from that screening on that centroid set.
        Optional only because a synthetic fit has no such file.
    """
    n_q = int(n_q)
    n_mu = int(n_mu)
    n_p = int(n_p)
    if min(n_q, n_mu, n_p) < 1:
        raise ValueError(
            f"allocate_fit_store: extents must be positive; got n_q="
            f"{n_q}, n_mu={n_mu}, n_p={n_p}")
    dtype = np.complex128 if dtype is None else dtype
    if energy_unit is not None and str(energy_unit) not in FIT_ENERGY_UNITS:
        raise ValueError(
            f"allocate_fit_store: energy_unit must be one of "
            f"{tuple(FIT_ENERGY_UNITS)}, got {energy_unit!r}")
    with _h5(dest, mode) as grp:
        _initialise_fit_metadata(
            grp, n_q=n_q, n_mu=n_mu, n_p=n_p, energy_unit=energy_unit,
            grid_hash=grid_hash, table_hash=table_hash,
            centroid_hash=centroid_hash, unfold_tables=unfold_tables,
            provenance=provenance, occupation_state=occupation_state)
        grp.create_dataset("Omega_p", shape=(n_p, n_q, n_mu, n_mu),
                           dtype=dtype)
        grp.create_dataset("B_p", shape=(n_p, n_q, n_mu, n_mu),
                           dtype=dtype)
        for key in REQUIRED_DIAGNOSTICS:
            grp.create_dataset("fit_" + key, shape=(n_q, n_mu, n_mu),
                               dtype=np.float64)

        return fit_completion_ledger(grp)


def allocate_fit_store_collective(
    dest, *, mesh_xy, n_q, n_mu, n_p, diagnostic_keys,
    energy_unit=None, grid_hash=None, table_hash=None, centroid_hash=None,
    unfold_tables=None, dtype=None, provenance=None, occupation_state=None,
    mode="w",
):
    """Allocate a fit store without any rank owning a pole tensor.

    SlabIO first replaces the inode and collectively creates the pole and
    diagnostic datasets.  This ordering is required on Lustre: the stripe
    layout is fixed when the inode is created, so a preliminary serial-h5py
    metadata file would pin the multi-gigabyte fit store to one OST.  After
    SlabIO closes, rank zero stamps only the small ledger/tables and a barrier
    makes the complete store visible before any block is written.

    COLLECTIVE over ``mesh_xy``.  ``dest`` must be a PATH (SlabIO), and
    ``mode`` is REFUSED unless ``'w'`` — this call owns the inode.

    ``diagnostic_keys`` is required and must be a superset of
    :data:`REQUIRED_DIAGNOSTICS`; one ``fit_<key>`` dataset of
    ``(n_q, n_mu, n_mu)`` float64 is created per key, so the set is fixed
    at allocation and cannot grow later.  ``unfold_tables``,
    ``occupation_state`` and the three identity hashes are stamped as the
    serial twin documents them.

    RETURNS :func:`fit_completion_ledger`, UNIFORMLY — every rank reads it
    back after the barrier, so this return does not depend on the rank
    (which is now true of every collective in this module).

    Open sequence: FFI ``'w'`` → barrier → rank-0 h5py ``'a'`` → barrier →
    all-rank h5py ``'r'``.  Cross-stack alternation with writes on both
    sides; ``LORRAX_HDF5_ONE_OWNER=strict`` refuses it.
    """
    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    n_q, n_mu, n_p = int(n_q), int(n_mu), int(n_p)
    if min(n_q, n_mu, n_p) < 1:
        raise ValueError("allocate_fit_store_collective: extents must be positive")
    if energy_unit is not None and str(energy_unit) not in FIT_ENERGY_UNITS:
        raise ValueError(
            f"allocate_fit_store_collective: unsupported energy unit "
            f"{energy_unit!r}")
    if mode != "w":
        raise ValueError(
            "allocate_fit_store_collective requires mode='w': allocation "
            "must replace the inode so SlabIO can apply its rank-aware "
            "Lustre stripe policy")
    keys = tuple(sorted(str(k) for k in diagnostic_keys))
    if not set(REQUIRED_DIAGNOSTICS).issubset(keys):
        raise ValueError(
            "allocate_fit_store_collective requires "
            + " and ".join(REQUIRED_DIAGNOSTICS) + " diagnostics")
    dtype = np.complex128 if dtype is None else dtype
    with SlabIO(dest, mode="w", mesh=mesh_xy) as io:
        io.create_dataset("Omega_p", shape=(n_p, n_q, n_mu, n_mu),
                          dtype=dtype)
        io.create_dataset("B_p", shape=(n_p, n_q, n_mu, n_mu), dtype=dtype)
        for key in keys:
            io.create_dataset("fit_" + key, shape=(n_q, n_mu, n_mu),
                              dtype=np.float64)
    barrier("mpa_fit_payload_allocated")
    if process_rank() == 0:
        with _h5(dest, "a") as grp:
            _initialise_fit_metadata(
                grp, n_q=n_q, n_mu=n_mu, n_p=n_p,
                energy_unit=energy_unit, grid_hash=grid_hash,
                table_hash=table_hash, centroid_hash=centroid_hash,
                unfold_tables=unfold_tables, provenance=provenance,
                diagnostic_keys=keys, preserve_payload=True,
                occupation_state=occupation_state)
    barrier("mpa_fit_metadata_allocated")
    return fit_completion_ledger(dest)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0).isoformat()


def _open_fit(grp):
    if MPA_FIT_SUFFIX not in grp:
        raise ValueError(
            f"mpa_store: this group carries no {MPA_FIT_SUFFIX!r} "
            f"ledger, so it cannot say which column ranges of which q "
            f"have been fitted.  A staged store without its ledger is a "
            f"tensor of poles indistinguishable from a tensor of zeros.")
    version = int(grp.attrs.get("mpa_fit_format_version", -1))
    if version != MPA_FIT_FORMAT_VERSION:
        raise ValueError(
            f"mpa_store: fit store is format version {version}; this "
            f"reader is version {MPA_FIT_FORMAT_VERSION}.  Refusing "
            f"rather than guessing.")
    return grp[MPA_FIT_SUFFIX]


def _append(dset, values):
    n = int(dset.shape[0])
    arr = np.asarray(values)
    dset.resize(n + arr.shape[0], axis=0)
    dset[n:] = arr


def _commit_fit_block(led, iq, cols, lo, hi, cond_max, berr_max):
    """Advance the completion ledger for one block whose bytes landed.

    THE JOURNAL RECORDS THE SPAN, ``blocks_done`` RECORDS THE TRUTH.
    ``fit_schedule`` only ever emits contiguous blocks, so for a normal
    walk the two agree exactly; a caller that hands a scattered
    selection gets a span WIDER than its column count, which is why the
    ledger and not the journal is what :func:`finalize_fit_store` and
    :func:`read_fit_block` refuse on.  A sentinel in the journal would
    have made "which columns" a question with two answers.
    """
    done = np.asarray(led["blocks_done"][()], dtype=bool)
    done[iq, cols] = True
    led["blocks_done"][...] = done
    _append(led["block_journal"],
            np.array([[iq, lo, hi]], dtype=np.int64))
    for key, val in zip(REQUIRED_DIAGNOSTICS, (cond_max, berr_max)):
        _append(led["block_" + key + "_max"], np.array([float(val)]))


def _unit_scale(source_unit, to_unit, where):
    """The multiplier taking energies from ``source_unit`` to ``to_unit``.

    Both units must be declared in :data:`FIT_ENERGY_UNITS`; an
    undeclared unit is refused rather than passed through at 1.0.
    """
    if source_unit not in FIT_ENERGY_UNITS or to_unit not in FIT_ENERGY_UNITS:
        raise ValueError(
            f"{where}: both stored and requested energy units must be "
            f"declared; stored={source_unit!r}, requested={to_unit!r}")
    return FIT_ENERGY_UNITS[source_unit] / FIT_ENERGY_UNITS[to_unit]


def write_head_fit(
    dest,
    sample_z,
    sample_Wc,
    Omega_p,
    B_p,
    *,
    energy_unit,
    fit_condition,
    fit_backward_error,
    fit_max_abs_residual,
    model,
    occupation_state=None,
    mode="a",
):
    """Write the complete scalar q->0 MPA fit and stamp readiness last.

    The head is intentionally not expanded to ``(q,mu,mu)``: its samples and
    poles are one tiny frequency axis, owned by the same file as the body fit.
    ``sample_Wc`` stays in Coulomb-head atomic units; ``sample_z``, ``Omega_p``
    and ``B_p`` use ``energy_unit`` (the pole residue has one energy factor).
    """
    if str(energy_unit) not in FIT_ENERGY_UNITS:
        raise ValueError(
            f"write_head_fit: energy_unit must be one of "
            f"{tuple(FIT_ENERGY_UNITS)}, got {energy_unit!r}")
    z = np.ascontiguousarray(sample_z, dtype=np.complex128).reshape(-1)
    wc = np.ascontiguousarray(sample_Wc, dtype=np.complex128).reshape(-1)
    poles = np.ascontiguousarray(Omega_p, dtype=np.complex128).reshape(-1)
    residues = np.ascontiguousarray(B_p, dtype=np.complex128).reshape(-1)
    if z.size < 1 or poles.size < 1:
        raise ValueError("write_head_fit: sample and pole axes must be nonempty")
    if z.shape != wc.shape:
        raise ValueError("write_head_fit: sample_z and sample_Wc shapes differ")
    if poles.shape != residues.shape:
        raise ValueError("write_head_fit: Omega_p and B_p shapes differ")
    for name, arr in (("sample_z", z), ("sample_Wc", wc),
                      ("Omega_p", poles), ("B_p", residues)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"write_head_fit: {name} contains non-finite values")
    diag = {
        "fit_condition": float(fit_condition),
        "fit_backward_error": float(fit_backward_error),
        "fit_max_abs_residual": float(fit_max_abs_residual),
    }
    if any(not np.isfinite(v) or v < 0.0 for v in diag.values()):
        raise ValueError("write_head_fit: diagnostics must be finite and nonnegative")

    qs = _qs()
    with _h5(dest, mode) as grp:
        _open_fit(grp)
        if MPA_HEAD_SUFFIX in grp:
            del grp[MPA_HEAD_SUFFIX]
        head = grp.create_group(MPA_HEAD_SUFFIX)
        head.attrs["ready"] = False
        head.create_dataset("sample_z", data=z)
        head.create_dataset("sample_Wc", data=wc)
        head.create_dataset("Omega_p", data=poles)
        head.create_dataset("B_p", data=residues)
        head.attrs["format_version"] = np.int64(1)
        head.attrs["model"] = str(model)
        head.attrs["frequency_unit"] = str(energy_unit)
        head.attrs["Wc_unit"] = "a.u."
        head.attrs["residue_unit"] = f"{energy_unit}*a.u."
        for key, value in diag.items():
            head.attrs[key] = value
        if occupation_state is not None:
            stamp_occupation_provenance(head, occupation_state)
        head.attrs["ready"] = True


def read_head_fit(src, *, to_unit=None, mode="r"):
    """Read the complete scalar q->0 MPA fit; refuse absent/partial data."""
    qs = _qs()
    with _h5(src, mode) as grp:
        _open_fit(grp)
        if MPA_HEAD_SUFFIX not in grp:
            raise ValueError("read_head_fit: fit store carries no scalar head")
        head = grp[MPA_HEAD_SUFFIX]
        if int(head.attrs.get("format_version", -1)) != 1:
            raise ValueError("read_head_fit: unsupported scalar-head format")
        if not bool(head.attrs.get("ready", False)):
            raise ValueError("read_head_fit: scalar head is NOT READY")
        source_unit = qs.qirr_attr_str(head, "frequency_unit")
        z = np.asarray(head["sample_z"][()])
        wc = np.asarray(head["sample_Wc"][()])
        poles = np.asarray(head["Omega_p"][()])
        residues = np.asarray(head["B_p"][()])
        diagnostics = {
            key: float(head.attrs[key])
            for key in ("fit_condition", "fit_backward_error",
                        "fit_max_abs_residual")
        }
        units = {
            "frequency": source_unit,
            "Wc": qs.qirr_attr_str(head, "Wc_unit"),
            "residue": qs.qirr_attr_str(head, "residue_unit"),
        }
        model = qs.qirr_attr_str(head, "model")
        if model not in _HEAD_FIT_MODELS:
            raise ValueError(
                f"read_head_fit: got scalar-head model {model!r}; want "
                f"one of {_HEAD_FIT_MODELS} — the only fitting protocols "
                f"this reader knows how to interpret, and a pole set "
                f"whose protocol nobody can name cannot be consumed "
                f"correctly.  Fix: refit the head with a known model, or "
                f"teach _HEAD_FIT_MODELS the new one alongside its "
                f"consumer.")
    if to_unit is not None:
        scale = _unit_scale(source_unit, to_unit, "head read")
        z, poles, residues = z * scale, poles * scale, residues * scale
        units["frequency"] = str(to_unit)
        units["residue"] = f"{to_unit}*a.u."
    return {
        "sample_z": z,
        "sample_Wc": wc,
        "Omega_p": poles,
        "B_p": residues,
        "units": units,
        "diagnostics": diagnostics,
        "model": model,
        "ready": True,
    }


def write_head_fit_collective(
    dest,
    sample_z,
    sample_Wc,
    Omega_p,
    B_p,
    *,
    mesh_xy,
    energy_unit,
    fit_condition,
    fit_backward_error,
    fit_max_abs_residual,
    grid_hash,
    fit_provenance,
    model="multipole",
    occupation_state=None,
):
    """Collectively publish one tiny scalar head fit through SlabIO.

    COLLECTIVE over ``mesh_xy`` (four barriers).  ``dest`` must be a PATH.

    SHAPES AND UNITS.  All four arrays are flattened to 1-D complex128.
    ``sample_z``/``sample_Wc`` share an extent — 2·n_p in production — and
    ``Omega_p``/``B_p`` share a DIFFERENT one, n_p; the two pairs are not
    required to agree with each other.  Units follow the serial
    :func:`write_head_fit`: ``sample_Wc`` stays in Coulomb-head atomic
    units, while ``sample_z``, ``Omega_p`` and ``B_p`` use ``energy_unit``
    (the pole residue carries one energy factor).

    REPLICATED INPUT, RANK-0 BYTES.  The vectors are published through
    ``SlabIO.write_attr``, which queues and lets only rank 0's copy land at
    close — so every rank must supply identical values, and a rank-dependent
    array here is silently discarded on all ranks but one.

    ORDERING PRECONDITION: the BODY fit must already be finalized and its
    ``w_grid_hash`` must equal ``grid_hash``; both are refused here.  A head
    cannot be attached to an incomplete store.

    FORMAT VERSION 2, AND IT IS NOT THE SERIAL PAIR'S.
    :func:`write_head_fit` stamps 1 and :func:`read_head_fit` refuses
    anything else; this writer stamps 2 and
    :func:`read_head_fit_collective` refuses anything else.  The two pairs
    CANNOT read each other's heads.  Four HDF5 opens per call: all-rank
    h5py ``'r'``, rank-0 h5py ``'a'``, FFI ``'a'``, rank-0 h5py ``'a'``.
    """
    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    z = np.ascontiguousarray(sample_z, dtype=np.complex128).reshape(-1)
    wc = np.ascontiguousarray(sample_Wc, dtype=np.complex128).reshape(-1)
    poles = np.ascontiguousarray(Omega_p, dtype=np.complex128).reshape(-1)
    residues = np.ascontiguousarray(B_p, dtype=np.complex128).reshape(-1)
    if str(energy_unit) not in FIT_ENERGY_UNITS:
        raise ValueError("collective scalar head has an unsupported energy unit")
    if str(model) not in _HEAD_FIT_MODELS:
        raise ValueError(
            f"write_head_fit_collective: scalar-head model {model!r} is "
            f"not one of {_HEAD_FIT_MODELS}; refuse to publish a fitting "
            "protocol that the matching reader cannot consume")
    if (z.size < 1 or poles.size < 1 or z.shape != wc.shape
            or poles.shape != residues.shape):
        raise ValueError("collective scalar head has inconsistent vector extents")
    for name, arr in (
        ("sample_z", z), ("sample_Wc", wc),
        ("Omega_p", poles), ("B_p", residues),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"collective scalar head {name} is not finite")
    diagnostics = {
        "fit_condition": float(fit_condition),
        "fit_backward_error": float(fit_backward_error),
        "fit_max_abs_residual": float(fit_max_abs_residual),
    }
    if any(not np.isfinite(v) or v < 0.0 for v in diagnostics.values()):
        raise ValueError("collective scalar-head diagnostics are invalid")
    ledger = fit_completion_ledger(dest)
    if not ledger["complete"]:
        raise ValueError("scalar head may only be attached to a finalized body fit")
    if str(ledger["w_grid_hash"]) != str(grid_hash):
        raise ValueError(
            "scalar-head grid hash does not match the fitted body grid")

    # ``fit_completion_ledger`` is an all-rank serial-h5py read.  Rank 0
    # must not open the same inode for write until every peer has closed
    # that reader: otherwise a fast rank 0 sets HDF5's superblock write-open
    # flag while a slower peer is still entering its read, which that peer
    # refuses as "file is already open for write".  The following barrier
    # is therefore a reader->single-writer ownership transfer, not optional
    # synchronization around the later parallel payload write.
    barrier("mpa_head_ledger_readers_closed")
    if process_rank() == 0:
        with _h5(dest, "a") as grp:
            _open_fit(grp)
            if MPA_HEAD_SUFFIX in grp:
                del grp[MPA_HEAD_SUFFIX]
            head = grp.create_group(MPA_HEAD_SUFFIX)
            head.attrs["ready"] = False
            head.attrs["format_version"] = np.int64(2)
            head.attrs["model"] = str(model)
            head.attrs["frequency_unit"] = str(energy_unit)
            head.attrs["Wc_unit"] = "a.u."
            head.attrs["residue_unit"] = f"{energy_unit}*a.u."
            head.attrs["mpa_grid_hash"] = str(grid_hash)
            for key, value in diagnostics.items():
                head.attrs[key] = value
            for key, value in sorted(dict(fit_provenance).items()):
                head.attrs["fit_" + str(key)] = value
            if occupation_state is not None:
                stamp_occupation_provenance(head, occupation_state)
    barrier("mpa_head_metadata_allocated")
    with SlabIO(dest, mode="a", mesh=mesh_xy) as io:
        prefix = MPA_HEAD_SUFFIX + "/"
        io.write_attr(prefix + "sample_z", z)
        io.write_attr(prefix + "sample_Wc", wc)
        io.write_attr(prefix + "Omega_p", poles)
        io.write_attr(prefix + "B_p", residues)
    barrier("mpa_head_payload_written")
    if process_rank() == 0:
        with _h5(dest, "a") as grp:
            grp[MPA_HEAD_SUFFIX].attrs["ready"] = True
    barrier("mpa_head_committed")


def read_head_fit_collective(src, *, mesh_xy, to_unit=None):
    """Collectively read and certify the scalar head fit through SlabIO.

    COLLECTIVE over ``mesh_xy``.  ``src`` must be a PATH.  Every open on
    this path is read-only (h5py ``'r'`` twice, then FFI ``'r'``), which is
    the cross-stack concurrency the one-owner registry allows.

    WHAT IS READ, stated because this is the call that produced failure
    signature S3 (``docs/architecture/slab_io.md#s3``): the four
    ``mpa_head/{sample_z,sample_Wc,Omega_p,B_p}`` vectors, WHOLE, with
    ``partition_spec=P(None)`` and **no offset and no shape** — so the
    extent comes from the dataset and the offset that reaches the FFI is
    zero.  ``sample_z``/``sample_Wc`` are 2·n_p long, ``Omega_p``/``B_p``
    are n_p.  A nonzero ``offset_base`` in a refusal from this call is not
    an arithmetic mistake in this function; it is the marshal.

    RETURNS HOST NUMPY, not sharded arrays, despite taking ``mesh_xy``:
    the four vectors come back ``np.complex128`` via ``as_numpy=True``,
    alongside ``units``, ``diagnostics``, ``provenance``, ``model``,
    ``occupation_stamps`` and ``ready``.

    "CERTIFY" MEANS: :func:`validate_fit_store` on the body, head
    ``format_version == 2``, ``ready``, head-vs-body ``mpa_grid_hash``,
    a known head model, and observed ``condition`` / ``backward_error``
    against the stamped ``*_max_allowed``.  Those two thresholds DEFAULT TO
    INFINITY when absent, so an unstamped head certifies vacuously.
    """
    from jax.sharding import PartitionSpec as P

    from file_io.slab_io import SlabIO

    ledger = validate_fit_store(src)
    with _h5(src, "r") as grp:
        _open_fit(grp)
        if MPA_HEAD_SUFFIX not in grp:
            raise ValueError("MPA fit store carries no scalar head")
        head = grp[MPA_HEAD_SUFFIX]
        if int(head.attrs.get("format_version", -1)) != 2:
            raise ValueError("collective scalar-head reader requires format version 2")
        if not bool(head.attrs.get("ready", False)):
            raise ValueError("scalar MPA head is NOT READY")
        source_unit = _qs().qirr_attr_str(head, "frequency_unit")
        model = _qs().qirr_attr_str(head, "model")
        grid_hash = _qs().qirr_attr_str(head, "mpa_grid_hash")
        occupation_stamps = None
        if ("mpa_" + _OCC_STAMP_ORDER[0]) in head.attrs:
            occupation_stamps = {
                "occ_hash": _qs().qirr_attr_str(head, "mpa_occ_hash"),
                "mu_ry": float(head.attrs["mpa_mu_ry"]),
            }
        diagnostics = {
            key: float(head.attrs[key])
            for key in (
                "fit_condition", "fit_backward_error",
                "fit_max_abs_residual")
        }
        provenance = {
            str(key)[len("fit_"):]: head.attrs[key]
            for key in head.attrs if str(key).startswith("fit_")
            and str(key) not in diagnostics
        }
    if str(grid_hash) != str(ledger["w_grid_hash"]):
        raise ValueError("scalar-head/body MPA grid hashes differ")
    if str(model) not in _HEAD_FIT_MODELS:
        raise ValueError(
            f"read_head_fit_collective: stored head model {model!r} is not "
            f"one of {_HEAD_FIT_MODELS}; a consumer must not silently "
            "interpret an unknown fitting protocol")
    condition_limit = float(provenance.get(
        "condition_max_allowed", np.inf))
    backward_limit = float(provenance.get(
        "backward_error_max_allowed", np.inf))
    if diagnostics["fit_condition"] > condition_limit:
        raise ValueError("scalar-head MPA fit exceeds its condition gate")
    if diagnostics["fit_backward_error"] > backward_limit:
        raise ValueError("scalar-head MPA fit exceeds its backward-error gate")

    prefix = MPA_HEAD_SUFFIX + "/"
    with SlabIO(src, mode="r", mesh=mesh_xy) as io:
        z = io.read_slab(
            prefix + "sample_z", partition_spec=P(None), as_numpy=True)
        wc = io.read_slab(
            prefix + "sample_Wc", partition_spec=P(None), as_numpy=True)
        poles = io.read_slab(
            prefix + "Omega_p", partition_spec=P(None), as_numpy=True)
        residues = io.read_slab(
            prefix + "B_p", partition_spec=P(None), as_numpy=True)
    z = np.asarray(z, dtype=np.complex128)
    wc = np.asarray(wc, dtype=np.complex128)
    poles = np.asarray(poles, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    if z.shape != wc.shape or poles.shape != residues.shape:
        raise ValueError("collective scalar-head payload has inconsistent shapes")
    if not all(np.all(np.isfinite(x)) for x in (z, wc, poles, residues)):
        raise ValueError("collective scalar-head payload is not finite")
    if to_unit is not None:
        # THE SHARED HELPER, not a second copy of the same policy: it
        # refuses an undeclared unit by NAME and prints both spellings,
        # where the inline version said only that the conversion was
        # "unsupported" (audit §E.3 item 12).
        scale = _unit_scale(source_unit, to_unit, "read_head_fit_collective")
        z, poles, residues = z * scale, poles * scale, residues * scale
        source_unit = str(to_unit)
    return {
        "sample_z": z,
        "sample_Wc": wc,
        "Omega_p": poles,
        "B_p": residues,
        "units": {
            "frequency": source_unit,
            "Wc": "a.u.",
            "residue": f"{source_unit}*a.u.",
        },
        "diagnostics": diagnostics,
        "provenance": provenance,
        "model": model,
        "occupation_stamps": occupation_stamps,
        "ready": True,
    }


def write_fit_block(
    dest,
    q,
    mu_cols,
    Omega_p_block,
    B_p_block,
    diag_block,
    *,
    mode="a",
):
    """Append one column block's poles and residues as the fit completes.

    Parameters
    ----------
    q
        Index into the stored q axis — the SAME axis the W(ω) file
        stores, wedge or full.  The fit does not unfold: poles fitted on
        the wedge unfold like W does, per q, and doing it here would
        store ``n_q_full`` copies of a tensor the symmetry says is
        ``n_q_ibz`` of them.
    mu_cols
        The ν columns this block covers, as handed to
        :func:`read_w_columns`.
    Omega_p_block, B_p_block
        ``(n_p, N_μ_rows, len(mu_cols))`` complex — the poles
        Ω_p = a_p − iΓ_p and their residues B_p, per (μ, ν) element.
    diag_block
        Dict with ``"condition"`` and ``"backward_error"``, each
        ``(N_μ_rows, len(mu_cols))`` float.  REQUIRED, not optional:
        the Σ stage's certification refuses poles whose fit did not meet
        its gates, and a pole whose conditioning nobody recorded cannot
        be refused later — it can only be trusted.  Extra keys are
        stored beside them under their own names.

    Returns the ledger dict, so a driver can log progress without a
    second open.
    """
    qs = _qs()
    Om = np.asarray(Omega_p_block)
    Bp = np.asarray(B_p_block)
    with _h5(dest, mode) as grp:
        led = _open_fit(grp)
        n_p = int(grp.attrs["mpa_fit_n_p"])
        n_q = int(grp.attrs["mpa_fit_n_q"])
        n_mu = int(grp.attrs["mpa_fit_n_mu_logical"])
        if bool(grp.attrs.get("mpa_fit_complete", False)):
            raise ValueError(
                "write_fit_block: this store is FINALIZED.  Appending "
                "to a finalized file would make its completion stamp a "
                "claim about a state that no longer exists; re-open the "
                "fit by allocating a new store.")
        iq = int(q)
        if not 0 <= iq < n_q:
            raise IndexError(
                f"write_fit_block: q={iq} is outside [0, {n_q})")
        cols = normalise_columns(mu_cols, n_mu)
        want = (n_p, n_mu, int(cols.size))
        for label, arr in (("Omega_p_block", Om), ("B_p_block", Bp)):
            if arr.shape != want:
                raise ValueError(
                    f"write_fit_block: {label} is {arr.shape}, expected "
                    f"{want} = (n_p, N_μ_rows, len(mu_cols)).  The row "
                    f"axis is WHOLE — the block is 1-D sharded on rows "
                    f"and gathered before it is written — and the pole "
                    f"axis leads because the Σ stage consumes W(τ) = "
                    f"Σ_p B_p e^{{−iΩ_p τ}} with p outermost.")
        diag = _canonical_diagnostics(diag_block, n_mu, int(cols.size))

        # ONE DIAGNOSTIC SET PER STORE.  The first block fixes which
        # quantities this fit measured; a later block that measured a
        # different set would leave the odd-one-out's array full of
        # zeros wherever the other blocks wrote — and a zero condition
        # number is a PERFECTLY conditioned solve, so the Σ stage's
        # certification would pass exactly the elements nobody measured.
        keys = ",".join(sorted(diag))
        stamped = qs.qirr_attr_str(led, "diagnostic_keys")
        if stamped is None:
            led.attrs["diagnostic_keys"] = keys
        elif stamped != keys:
            raise ValueError(
                f"write_fit_block: this block reports diagnostics "
                f"[{keys}] but the store's earlier blocks reported "
                f"[{stamped}].  A quantity measured for some blocks and "
                f"not others reads back as ZERO for the rest, and a "
                f"zero condition number is a perfectly conditioned "
                f"solve — the certification would pass precisely the "
                f"elements nobody measured.")

        done = led["blocks_done"][()]
        already = cols[done[iq, cols]]
        if already.size:
            raise ValueError(
                f"write_fit_block: q={iq} columns {already[:8].tolist()}"
                f"{'...' if already.size > 8 else ''} are already "
                f"fitted.  Rewriting a fitted block would replace poles "
                f"the ledger already certified as complete, and the "
                f"journal would carry two entries for one column with "
                f"no rule for which one the diagnostics belong to.")

        lo, hi, sel = _column_span(cols)
        sel = slice(lo, hi) if sel is None else sel
        grp["Omega_p"][:, iq, :, sel] = Om
        grp["B_p"][:, iq, :, sel] = Bp
        for key in REQUIRED_DIAGNOSTICS:
            grp["fit_" + key][iq, :, sel] = diag[key]
        for key, arr in diag.items():
            if key in REQUIRED_DIAGNOSTICS:
                continue
            name = "fit_" + key
            if name not in grp:
                grp.create_dataset(name, shape=(n_q, n_mu, n_mu),
                                   dtype=np.float64)
            grp[name][iq, :, sel] = arr

        _commit_fit_block(led, iq, cols, lo, hi,
                          np.max(diag["condition"]),
                          np.max(diag["backward_error"]))
        return fit_completion_ledger(grp)


def write_fit_block_collective(
    dest,
    q,
    mu_cols,
    Omega_p_block,
    B_p_block,
    diag_block,
    *,
    mesh_xy,
    block_condition_max,
    block_backward_error_max,
    diagnostics_finite,
):
    """Collectively write one native row-sharded fit block.

    Pole and diagnostic bytes never leave their owning devices.  Only after
    all collective writes have closed does rank zero advance the small
    completion ledger; a barrier publishes that commit to every rank.

    COLLECTIVE over ``mesh_xy``.  ``dest`` must be a PATH.  Returns the
    updated ledger, uniformly on every rank.

    SHAPES AND SPECS, all enforced.  ``Omega_p_block`` and ``B_p_block`` are
    ``(n_p, 1, n_mu_padded, n_cols_buffer)`` on
    ``P(None, None, ('x', 'y'), None)``; each ``diag_block[key]`` is
    ``(1, n_rows, width)`` on ``P(None, ('x', 'y'))`` (a two-entry spec for
    a rank-3 array, relying on JAX canonicalization).  ``mu_cols`` must be
    ONE CONTIGUOUS RANGE.

    ``block_condition_max``, ``block_backward_error_max`` and
    ``diagnostics_finite`` are host scalars that MUST BE IDENTICAL ON EVERY
    RANK — only rank 0's are committed.  The driver satisfies that with a
    ``lax.pmax``/``lax.pmin`` reduction before the call; nothing here checks
    it.

    THE COST, measured 2026-08-15 and disclosed because it is the dominant
    term in the store's HDF5 traffic: **five opens per block, four of them
    h5py** — the ledger read, a ``diagnostic_keys`` attr read, the FFI
    write, the rank-0 commit, and the ledger read again.  At the R6 deck's
    464 blocks per iteration that is ~1900 h5py and ~464 FFI opens on one
    file per iteration, and it is the h5py→FFI→h5py alternation
    :mod:`file_io.hdf5_owner` counts.  **This function therefore cannot run
    under ``LORRAX_HDF5_ONE_OWNER=strict``**, which is the target state
    audit A1 names; removing three of the four h5py opens is on the
    follow-up list.
    """
    from jax.sharding import PartitionSpec as P

    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    ledger = fit_completion_ledger(dest)
    if ledger["complete"]:
        raise ValueError("write_fit_block_collective: store is finalized")
    iq = int(q)
    if not 0 <= iq < ledger["n_q"]:
        raise IndexError(
            f"write_fit_block_collective: q={iq} outside [0,{ledger['n_q']})")
    cols = normalise_columns(mu_cols, ledger["n_mu"])
    lo, hi, sel = _column_span(cols)
    if sel is not None:
        raise ValueError("write_fit_block_collective requires contiguous columns")
    already = cols[np.asarray(ledger["blocks_done"][iq, cols], dtype=bool)]
    if already.size:
        raise ValueError(
            f"write_fit_block_collective: q={iq} columns "
            f"{already[:8].tolist()} are already fitted")
    if not bool(diagnostics_finite):
        raise ValueError(
            "write_fit_block_collective: fit diagnostics contain non-finite "
            "values; refusing to certify this block")

    pole_spec = P(None, None, ("x", "y"), None)
    # JAX canonicalizes trailing replicated entries away on rank-3 arrays.
    diag_spec = P(None, ("x", "y"))

    Om, Bp = Omega_p_block, B_p_block
    for arr in (Om, Bp):
        _require_layout(
            arr, mesh_xy, pole_spec,
            "write_fit_block_collective requires Omega_p and B_p on "
            "P(None,None,('x','y'),None)")
    n_p, one, n_rows, width = map(int, Om.shape)
    if tuple(Bp.shape) != tuple(Om.shape) or one != 1:
        raise ValueError(
            "write_fit_block_collective: pole blocks must agree and have "
            "shape (n_p,1,n_mu_padded,n_cols_buffer)")
    if n_p != ledger["n_p"] or n_rows < ledger["n_mu"] \
            or width < int(cols.size):
        raise ValueError(
            f"write_fit_block_collective: pole block {tuple(Om.shape)} is "
            f"incompatible with n_p={ledger['n_p']}, n_mu={ledger['n_mu']}, "
            f"columns={int(cols.size)}")
    keys = tuple(sorted(diag_block))
    stamped = None
    with _h5(dest, "r") as grp:
        stamped = _qs().qirr_attr_str(_open_fit(grp), "diagnostic_keys")
    if stamped != ",".join(keys):
        raise ValueError(
            f"write_fit_block_collective: diagnostics {keys} do not match "
            f"allocated keys {stamped!r}")
    for key, arr in diag_block.items():
        if tuple(arr.shape) != (1, n_rows, width):
            raise ValueError(
                f"write_fit_block_collective: diagnostic {key!r} has "
                f"shape {tuple(arr.shape)}, expected {(1, n_rows, width)}")
        _require_layout(
            arr, mesh_xy, diag_spec,
            f"write_fit_block_collective: diagnostic {key!r} must be "
            "on P(None,('x','y'),None)")

    with SlabIO(dest, mode="a", mesh=mesh_xy) as io:
        valid_pole = (n_p, 1, ledger["n_mu"], int(cols.size))
        valid_diag = (1, ledger["n_mu"], int(cols.size))
        global_pole = (n_p, ledger["n_q"], ledger["n_mu"],
                       ledger["n_mu"])
        global_diag = (ledger["n_q"], ledger["n_mu"], ledger["n_mu"])
        io.write_slab("Omega_p", Om, offset=(0, iq, 0, lo),
                      global_shape=global_pole, valid_shape=valid_pole)
        io.write_slab("B_p", Bp, offset=(0, iq, 0, lo),
                      global_shape=global_pole, valid_shape=valid_pole)
        for key in keys:
            io.write_slab("fit_" + key, diag_block[key],
                          offset=(iq, 0, lo), global_shape=global_diag,
                          valid_shape=valid_diag)

    if process_rank() == 0:
        with _h5(dest, "a") as grp:
            _commit_fit_block(_open_fit(grp), iq, cols, lo, hi,
                              block_condition_max,
                              block_backward_error_max)
    barrier(f"mpa_fit_block_{iq}_{lo}_{hi}_committed")
    return fit_completion_ledger(dest)


def _canonical_diagnostics(diag_block, n_rows, n_cols):
    """The per-block fit diagnostics, validated and float64.

    Condition number and backward error are the two the Σ stage's
    certification is stated in the MPA theory chapter: "condition numbers
    and backward error, diagonal/off-diagonal and norm-resolved
    distributions"), so they are REQUIRED and everything else is extra.
    """
    if not isinstance(diag_block, dict):
        raise TypeError(
            f"write_fit_block: diag_block must be a dict with "
            f"'condition' and 'backward_error'; got "
            f"{type(diag_block).__name__}")
    missing = [k for k in REQUIRED_DIAGNOSTICS if k not in diag_block]
    if missing:
        raise ValueError(
            f"write_fit_block: diag_block is missing {missing}.  The Σ "
            f"stage refuses poles that fail certification, and a pole "
            f"whose conditioning and backward error nobody recorded "
            f"cannot be refused later — only trusted.  These are not "
            f"optional telemetry; they are the evidence the refusal "
            f"runs on.")
    out = {}
    for key, val in diag_block.items():
        arr = np.ascontiguousarray(val, dtype=np.float64)
        if arr.shape != (n_rows, n_cols):
            raise ValueError(
                f"write_fit_block: diagnostic {key!r} is {arr.shape}, "
                f"expected {(n_rows, n_cols)} — one value per (μ, ν) "
                f"element of the block.  The fit is ELEMENTWISE in "
                f"ISDF, so a per-block scalar would hide exactly the "
                f"elements the certification is looking for.")
        if not np.all(np.isfinite(arr)):
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            raise ValueError(
                f"write_fit_block: diagnostic {key!r} carries {bad} "
                f"non-finite entries.  A NaN condition number is a "
                f"solve that failed, and writing it as data would let "
                f"the Σ stage's threshold comparison pass it silently "
                f"(NaN > tol is False).")
        out[key] = arr
    return out


def fit_completion_ledger(src, *, mode="r"):
    """Which column ranges of which q are fitted — a plain dict.

    ``blocks_done`` is the authority — ``(n_q, n_mu)`` bool, indexed
    ``blocks_done[iq, cols]``.  ``journal`` is the append-only record of the
    order the blocks arrived in: ``(n_blocks, 3)`` int64 of
    ``[iq, lo, hi]`` and NOTHING ELSE.  The per-block worst condition and
    backward error are SEPARATE 1-D keys, ``block_condition_max`` and
    ``block_backward_error_max``, with the scalar reductions under
    ``condition_max`` / ``backward_error_max``; this docstring used to put
    them inside ``journal``, and a caller indexing ``journal[:, 3]`` for a
    condition number gets an IndexError.  The two structures are not
    redundant: the ledger answers "is this column done", the journal
    answers "in what order", and the Σ stage's certification reads the
    maxima.

    Rank-local and serial, but invoked on every rank by three collective
    functions — a collective caller must invoke it uniformly.  ``src`` is a
    path or an already-open group; this is the door in this module most
    often passed both ways.
    """
    qs = _qs()
    with _h5(src, mode) as grp:
        led = _open_fit(grp)
        done = np.asarray(led["blocks_done"][()], dtype=bool)
        journal = np.asarray(led["block_journal"][()], dtype=np.int64)
        maxima = {
            key: np.asarray(led["block_" + key + "_max"][()],
                            dtype=np.float64)
            for key in REQUIRED_DIAGNOSTICS
        }
        certification = {
            str(key)[len("mpa_cert_"):]: grp.attrs[key]
            for key in grp.attrs if str(key).startswith("mpa_cert_")
        }
        provenance = {
            str(key)[len("prov_"):]: grp.attrs[key]
            for key in grp.attrs if str(key).startswith("prov_")
        }
        out = {
            "format_version": int(grp.attrs["mpa_fit_format_version"]),
            "n_p": int(grp.attrs["mpa_fit_n_p"]),
            "n_q": int(grp.attrs["mpa_fit_n_q"]),
            "n_q_full": int(grp.attrs.get(
                "mpa_fit_n_q_full", grp.attrs["mpa_fit_n_q"])),
            "q_storage": qs.qirr_attr_str(
                grp, "mpa_fit_q_storage") or "full",
            "table_hash": qs.qirr_attr_str(grp, "mpa_fit_table_hash"),
            "w_grid_hash": qs.qirr_attr_str(grp, "mpa_fit_w_grid_hash"),
            "w_table_hash": qs.qirr_attr_str(grp, "mpa_fit_w_table_hash"),
            "w_centroid_hash": qs.qirr_attr_str(
                grp, "mpa_fit_w_centroid_hash"),
            "energy_unit": qs.qirr_attr_str(grp, FIT_ENERGY_UNIT_ATTR),
            "n_mu": int(grp.attrs["mpa_fit_n_mu_logical"]),
            "complete": bool(grp.attrs.get("mpa_fit_complete", False)),
            "blocks_done": done,
            "n_done": int(done.sum()),
            "n_total": int(done.size),
            "journal": journal,
            "finalized_utc": qs.qirr_attr_str(grp, "mpa_fit_finalized_utc"),
            "certification": certification,
            "provenance": provenance,
        }
        for key, vals in maxima.items():
            out["block_" + key + "_max"] = vals
            out[key + "_max"] = float(vals.max()) if vals.size else None
        return out


def validate_fit_store(src, *, expected_identity=None,
                       expected_screening_diagrams=None, mode="r"):
    """Validate the finalized fit contract before Sigma reads pole bytes.

    ``expected_identity`` may name ``w_grid_hash``, ``w_table_hash`` and
    ``w_centroid_hash`` from the screening object currently in use.  The
    fit's own declared ``*_max_allowed`` certification thresholds are always
    enforced against its observed maxima.

    ``expected_screening_diagrams`` is the run's ``screening_diagrams``
    value.  RPA poles and ladder-corrected poles are the same shape, pass
    the same certification and read back equally plausibly, so the only
    thing separating them is the stamp the writer left — which makes this
    the load-time half of QUALITY_PATTERNS #10.  A store with NO stamp is
    refused rather than assumed RPA: "written before the axis existed" and
    "written by the RPA path" are different facts, and silently reading the
    first as the second is how a ladder fit gets consumed as an RPA one.

    IT VALIDATES THE LEDGER, NOT THE BYTES.  Every check is on ledger
    attributes: COMPLETE, the energy unit, the optional identity hashes,
    the PRESENCE of the ``mpa_cert_*`` thresholds, and observed-vs-allowed
    maxima.  ``Omega_p`` and ``B_p`` are never opened — not their shape,
    not their dtype, not their finiteness.  Read the summary line as "the
    contract Σ relies on", not "the poles were checked".

    Returns the :func:`fit_completion_ledger` dict, which callers use for
    ``n_p``.  Rank-local and serial; ``src`` is a path or an open group.
    """
    ledger = fit_completion_ledger(src, mode=mode)
    if expected_screening_diagrams is not None:
        want = str(getattr(expected_screening_diagrams, "value",
                           expected_screening_diagrams))
        got = ledger["provenance"].get("screening_diagrams")
        if got is None:
            raise ValueError(
                f"MPA fit store carries no screening_diagrams stamp, so it "
                f"cannot say whether its poles came from the RPA W or the "
                f"ladder-corrected W; this run is {want!r}.  Regenerate the "
                f"fit with a writer that stamps it (gw.mpa.model."
                f"build_mpa_fit does).")
        if str(got) != want:
            raise ValueError(
                f"MPA fit provenance mismatch: the store was built with "
                f"screening_diagrams = {str(got)!r} and this run is "
                f"{want!r}.  The two produce different W and therefore "
                f"different poles; reusing one for the other is the "
                f"changed-band-window class of silent reuse.")
    if not ledger["complete"]:
        raise ValueError("MPA Sigma requires a finalized pole fit store")
    if ledger["energy_unit"] not in FIT_ENERGY_UNITS:
        raise ValueError("MPA fit store does not declare a supported unit")
    for key, want in (expected_identity or {}).items():
        if key not in ("w_grid_hash", "w_table_hash", "w_centroid_hash"):
            raise KeyError(f"unknown MPA fit identity field {key!r}")
        got = ledger[key]
        if got is None or str(got) != str(want):
            raise ValueError(
                f"MPA fit identity mismatch for {key}: got {got!r}, "
                f"expected {want!r}")
    missing = [key + "_max_allowed" for key in REQUIRED_DIAGNOSTICS
               if key + "_max_allowed" not in ledger["certification"]]
    if missing:
        raise ValueError(
            "MPA Sigma requires certified pole fits; the store is missing "
            + ", ".join(missing))
    for metric_key in REQUIRED_DIAGNOSTICS:
        key = metric_key + "_max_allowed"
        allowed = ledger["certification"][key]
        if not np.isfinite(float(allowed)) or float(allowed) <= 0.0:
            raise ValueError(
                f"MPA fit has invalid stored certification {key}="
                f"{allowed!r}")
        metric = metric_key + "_max"
        got = ledger[metric]
        if got is None or float(got) > float(allowed):
            raise ValueError(
                f"MPA fit failed its stored certification: {metric}="
                f"{got!r} exceeds {allowed!r}")
    return ledger


def _require_certification(certification, dest):
    """A certification :func:`validate_fit_store` will accept, or refuse.

    THE PRESENCE AND THE VALUE, because the validator checks both and a
    finalize is the last moment either can be fixed: a threshold that is
    absent, non-numeric, non-finite or non-positive fails
    ``validate_fit_store`` exactly as hard as no certification at all, and
    a finalized store cannot be written to or finalized again.
    """
    cert = dict(certification or {})
    missing, bad = [], []
    for key in REQUIRED_DIAGNOSTICS:
        name = key + "_max_allowed"
        if name not in cert:
            missing.append(name)
            continue
        try:
            val = float(cert[name])
        except (TypeError, ValueError):
            bad.append(f"{name}={cert[name]!r}")
            continue
        if not np.isfinite(val) or val <= 0.0:
            bad.append(f"{name}={cert[name]!r}")
    if not missing and not bad:
        return
    faults = "; ".join(
        ([f"missing {', '.join(missing)}"] if missing else [])
        + ([f"unusable {', '.join(bad)}"] if bad else []))
    raise ValueError(
        "finalize_fit_store: refusing to finalize a store Σ could never "
        "read\n"
        f"  file  : {dest}\n"
        f"  got   : certification={certification!r} — {faults}.\n"
        f"  want  : a finite, positive "
        + " and ".join(k + "_max_allowed" for k in REQUIRED_DIAGNOSTICS)
        + ", the thresholds validate_fit_store enforces against the "
        "observed maxima.  Finalizing without them stamps a store that "
        "the validator refuses ('MPA Sigma requires certified pole "
        "fits') and that NOTHING can repair: a second finalize is "
        "refused and every writer refuses a finalized store.\n"
        "  fix   : pass certification={"
        + ", ".join(f"'{k}_max_allowed': <threshold>"
                    for k in REQUIRED_DIAGNOSTICS)
        + "}.  The fit driver's own values are the solver-consistency "
        "pair 1/rcond and sqrt(eps) (gw/mpa/fit_driver.py); a stage with "
        "no material tolerance of its own should reuse them rather than "
        "leave the store uncertified.")


def finalize_fit_store(dest, *, certification=None, mode="a"):
    """Stamp the store COMPLETE — once, and only when it is.

    Refuses a store with an unfitted (q, column) and NAMES the gaps,
    because "which columns are missing" is the only question a driver
    that crashed halfway actually has.  Refuses a second finalize for
    the same reason a second stamp is refused everywhere in this
    format: two claims about what the file says, differing on the day
    one of them is made against a different state.

    ``certification`` is stamped as ``mpa_cert_*`` — the thresholds the
    Σ stage should hold these poles to.  The OBSERVED maxima are stamped
    regardless (``mpa_fit_condition_max``, ``mpa_fit_backward_error_max``,
    ``mpa_fit_n_blocks``).

    ``certification`` IS REQUIRED, and the keyword keeps its ``None``
    default only so the omission fails by name instead of as a
    ``TypeError``.  A ``certification`` that would not satisfy
    :func:`validate_fit_store` — absent, empty, or missing/garbage in any
    ``<key>_max_allowed`` for :data:`REQUIRED_DIAGNOSTICS` — is REFUSED
    HERE, before the store is opened, because the resulting file would be
    unusable AND unrepairable: the validator refuses an uncertified store
    ("MPA Sigma requires certified pole fits"), a second
    :func:`finalize_fit_store` is refused, and every writer refuses a
    finalized store, so there is no way back through this API.  A store
    that cannot be validated must not be finalizable; the refusal is that
    sentence in code.  (Before 2026-08-15 this call accepted the omission
    and bricked the store for Σ — audit §E.3 item 1.)  The observed-maxima
    attrs are read by no code in ``src/`` or ``services/``; only tests
    assert them, which is why they are not a fallback.

    RANK-0 ONLY in production, and that is a real precondition, not a
    convention: the driver guards the call with ``process_rank() == 0`` and
    barriers after it.  Called on every rank this is a concurrent
    multi-process h5py write plus an "already finalized" race.  ``dest`` is
    a path or an open group; the default ``mode='a'`` is a WRITE open, so
    calling it while a SlabIO handle is live on the same path is exactly
    the case :mod:`file_io.hdf5_owner` refuses.  Returns the ledger.
    """
    _require_certification(certification, dest)
    qs = _qs()
    with _h5(dest, mode) as grp:
        led = _open_fit(grp)
        if bool(grp.attrs.get("mpa_fit_complete", False)):
            raise ValueError(
                f"finalize_fit_store: this store was already finalized "
                f"at {qs.qirr_attr_str(grp, 'mpa_fit_finalized_utc')}.  A "
                f"second finalize would stamp completeness against a "
                f"state the first one did not measure; if blocks were "
                f"written since, they were written to a file that "
                f"already claimed to be done, which is the bug and not "
                f"the fix.")
        done = np.asarray(led["blocks_done"][()], dtype=bool)
        if not done.all():
            gaps = []
            for iq in range(done.shape[0]):
                miss = np.flatnonzero(~done[iq])
                if miss.size:
                    gaps.append(f"q={iq}: columns "
                                f"{_ranges(miss)}")
            raise ValueError(
                f"finalize_fit_store: {int((~done).sum())} of "
                f"{int(done.size)} (q, column) pairs are unfitted, so "
                f"the store is not complete.  " +
                "; ".join(gaps[:6]) +
                (" ..." if len(gaps) > 6 else "") +
                "  Stamping it complete would tell the Σ stage that "
                "zeros are poles.")
        grp.attrs["mpa_fit_complete"] = True
        grp.attrs["mpa_fit_finalized_utc"] = _utc_now()
        for key in REQUIRED_DIAGNOSTICS:
            vals = np.asarray(led["block_" + key + "_max"][()])
            grp.attrs["mpa_fit_" + key + "_max"] = np.float64(
                vals.max() if vals.size else 0.0)
        grp.attrs["mpa_fit_n_blocks"] = np.int64(
            int(led["block_journal"].shape[0]))
        for key, val in (certification or {}).items():
            grp.attrs["mpa_cert_" + str(key)] = val
        return fit_completion_ledger(grp)


def _ranges(idx):
    """``[0 1 2 5 6]`` -> ``'0-2,5-6'`` — gaps a human can act on."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return ""
    cuts = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[0], cuts + 1])
    stops = np.concatenate([cuts, [idx.size - 1]])
    parts = [f"{int(idx[a])}" if idx[a] == idx[b]
             else f"{int(idx[a])}-{int(idx[b])}"
             for a, b in zip(starts, stops)]
    return ",".join(parts[:8]) + ("..." if len(parts) > 8 else "")


def _refuse_unfinalized(grp, ledger, allow_partial, where):
    if ledger["complete"] or allow_partial:
        return
    raise ValueError(
        f"{where}: this fit store is NOT FINALIZED — {ledger['n_done']} "
        f"of {ledger['n_total']} (q, column) pairs are fitted.  An "
        f"unfitted column reads back as zeros, and a zero pole is not "
        f"an absent pole: B_p = 0 at Ω_p = 0 contributes nothing to "
        f"W(τ) = Σ_p B_p e^{{−iΩ_p τ}} and therefore looks exactly like "
        f"a converged fit of a screening channel that is genuinely "
        f"dark.  Pass allow_partial=True to read the staged state "
        f"deliberately; the ledger comes back beside the arrays so the "
        f"caller can say which of it is real.")


def read_fit_block(src, q, mu_cols, *, allow_partial=False, mode="r"):
    """One column block's ``(Omega_p, B_p, diagnostics, ledger)``.

    Refuses an unfinalized store unless ``allow_partial=True``, and
    when partial, refuses the specific columns that are not fitted —
    "the file is incomplete" and "the columns you asked for are
    incomplete" are different facts and a driver resuming a crashed fit
    needs the second one.
    """
    qs = _qs()
    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial,
                            f"read_fit_block(q={q})")
        iq = int(q)
        if not 0 <= iq < ledger["n_q"]:
            raise IndexError(
                f"read_fit_block: q={iq} is outside [0, "
                f"{ledger['n_q']})")
        cols = normalise_columns(mu_cols, ledger["n_mu"])
        undone = cols[~ledger["blocks_done"][iq, cols]]
        if undone.size:
            raise ValueError(
                f"read_fit_block: q={iq} columns "
                f"{_ranges(undone)} are not fitted.  They read back as "
                f"zeros, which is a converged-looking dark channel and "
                f"not an absent one, so the refusal is on the LEDGER "
                f"and never on the data.")
        lo, hi, sel = _column_span(cols)
        sel = slice(lo, hi) if sel is None else sel
        Om = np.asarray(grp["Omega_p"][:, iq, :, sel])
        Bp = np.asarray(grp["B_p"][:, iq, :, sel])
        required_ds = tuple("fit_" + k for k in REQUIRED_DIAGNOSTICS)
        diag = {k: np.asarray(grp["fit_" + k][iq, :, sel])
                for k in REQUIRED_DIAGNOSTICS}
        for key in grp:
            if str(key).startswith("fit_") and key not in required_ds:
                diag[str(key)[len("fit_"):]] = np.asarray(
                    grp[key][iq, :, sel])
        return Om, Bp, diag, ledger


def read_fit_tensors(src, *, allow_partial=False, mode="r"):
    """The whole ``(Omega_p, B_p, diagnostics, ledger)``.

    For tests and offline inspection.  The Σ stage does NOT read this — it
    streams contiguous pole ranges through :class:`PoleReader` (opened by
    :func:`open_pole_reader`) and never holds the whole tensor.  This used
    to name :func:`read_poles`, which the Σ stage stopped using when
    ``PoleReader`` landed.

    ``Omega_p`` and ``B_p`` come back ``(n_p, n_q, n_mu, n_mu)``
    complex128 and ``diagnostics`` as ``{key: (n_q, n_mu, n_mu) float64}``
    — the whole tensor, on this rank, which is the cost the first
    paragraph is warning about.  Serial; ``src`` is a path or an open
    group.  Same finalize refusal as :func:`read_fit_block`.
    """
    qs = _qs()
    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial,
                            "read_fit_tensors")
        Om = np.asarray(grp["Omega_p"][()])
        Bp = np.asarray(grp["B_p"][()])
        diag = {str(k)[len("fit_"):]: np.asarray(grp[k][()])
                for k in grp if str(k).startswith("fit_")}
        return Om, Bp, diag, ledger


def stamp_fit_unfold_tables(dest, tables, *, mode="a"):
    """Store the W wedge's existing q-unfold tables beside its fitted poles."""
    qs = _qs()
    can = tables.canonical()
    with _h5(dest, mode) as grp:
        ledger = fit_completion_ledger(grp)
        storage = qs.validate_qirr_tables(
            can, int(ledger["n_q"]), int(ledger["n_mu"]))
        digest = can.digest()
        stamped = qs.qirr_attr_str(grp, "mpa_fit_table_hash")
        if stamped is not None and stamped != digest:
            raise ValueError(
                "stamp_fit_unfold_tables: the store already carries a "
                "different q-unfold table")
        name = FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX
        if name in grp:
            del grp[name]
        tgrp = grp.create_group(name)
        for key in ("irr_idx_q", "sym_idx_q", "q_irr_frac", "sym_perm",
                    "L_table"):
            tgrp.create_dataset(key, data=getattr(can, key))
        tgrp.attrs["n_sym_spatial"] = np.int64(can.n_sym_spatial)
        tgrp.attrs["table_hash"] = digest
        grp.attrs["mpa_fit_table_hash"] = digest
        grp.attrs["mpa_fit_q_storage"] = storage
        grp.attrs["mpa_fit_n_q_full"] = np.int64(can.n_q_full)
    return can


def read_fit_unfold_tables(src, *, mode="r"):
    """Return the fit store's q-unfold tables, or ``None`` for full BZ."""
    qs = _qs()
    with _h5(src, mode) as grp:
        if FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX not in grp:
            return None
        return qs.read_tables(grp, FIT_TABLE_OWNER, mode=mode)


def unfold_pole_field(Omega_p, B_p, tables, *, mesh_xy):
    """Unfold one pole wedge without conjugating its frequency dependence."""
    import jax.numpy as jnp

    from symmetry_maps import unfold_isdf_operator

    can = tables.canonical()
    if (Omega_p.ndim != 3 or tuple(B_p.shape) != tuple(Omega_p.shape)
            or Omega_p.shape[-1] != Omega_p.shape[-2]):
        raise ValueError("pole slabs must share one square (q,mu,mu) shape")
    if int(can.n_mu) > int(Omega_p.shape[-1]):
        raise ValueError("pole slab is smaller than its centroid tables")
    if int(can.n_mu) != int(Omega_p.shape[-1]):
        can = can.padded(int(Omega_p.shape[-1]))
    kwargs = dict(
        irr_idx=can.irr_idx_q, sym_idx=can.sym_idx_q,
        sym_perm=can.sym_perm, q_irr_frac=can.q_irr_frac,
        mesh_xy=mesh_xy, n_sym_spatial=int(can.n_sym_spatial),
        trs_rule="pair_transpose")
    zeros = np.zeros_like(np.asarray(can.L_table))
    Omega_full = unfold_isdf_operator(
        jnp.asarray(Omega_p), L_table=zeros, **kwargs)
    B_full = unfold_isdf_operator(
        jnp.asarray(B_p), L_table=can.L_table, **kwargs)
    return Omega_full, B_full


def _finish_pole_read(
    src, Omega, Bp, ledger, *, mesh_xy, unfold, return_sharded, to_unit,
    tables=_UNREAD,
):
    """Apply the one unit/unfold policy shared by pole readers.

    ``tables`` is the unfold table set when the caller already read it
    (:class:`PoleReader` does, once per iteration, BEFORE it opens its
    collective handle); the sentinel means "read it from ``src`` now",
    which is the one-shot :func:`read_poles` path.
    """
    if to_unit is not None:
        scale = _unit_scale(ledger["energy_unit"], to_unit, "pole read")
        Omega, Bp = Omega * scale, Bp * scale

    if unfold and ledger["q_storage"] == "ibz":
        import jax.numpy as jnp

        if mesh_xy is None:
            raise ValueError("a wedge pole unfold requires mesh_xy")
        if tables is _UNREAD:
            tables = read_fit_unfold_tables(src)
        if tables is None:
            raise ValueError("wedge fit has no unfold tables")
        if Omega.ndim == 3:
            Omega, Bp = unfold_pole_field(Omega, Bp, tables, mesh_xy=mesh_xy)
        else:
            Omega, Bp = map(
                jnp.stack,
                zip(*(unfold_pole_field(Omega[p], Bp[p], tables,
                                        mesh_xy=mesh_xy)
                      for p in range(int(Omega.shape[0])))))
    if return_sharded and mesh_xy is None:
        raise ValueError("return_sharded requires mesh_xy")
    return Omega, Bp


def _pole_read_shape(ledger, mesh_xy):
    """Global read shape for one pole range's trailing (q, μ, ν) axes.

    Deliberately ``runtime.padding.padded_mu_extent`` and NOT
    ``slab_io.mesh_divisible_shape``: the unfold's all_to_all needs the
    μ extent divisible by the device PRODUCT, not merely by each mesh
    axis.
    """
    from runtime.padding import padded_mu_extent

    n_pad = int(padded_mu_extent(ledger["n_mu"], mesh_xy))
    return ledger["n_q"], n_pad, n_pad


def _pole_range(ledger, pole_slice, where):
    """Normalise ``pole_slice`` against the ledger's pole count."""
    if pole_slice is None:
        lo, hi = 0, ledger["n_p"]
    elif isinstance(pole_slice, (int, np.integer)):
        lo, hi = int(pole_slice), int(pole_slice) + 1
    else:
        lo, hi, step = pole_slice.indices(ledger["n_p"])
        if step != 1:
            raise ValueError(f"{where} requires a contiguous pole slice")
    if not (0 <= lo < hi <= ledger["n_p"]):
        raise IndexError(
            f"{where}: pole range [{lo},{hi}) is outside "
            f"[0,{ledger['n_p']})")
    return int(lo), int(hi)


class PoleReader:
    """ONE collective handle serving every pole batch of one iteration.

    WHY THIS EXISTS (audit A1 fix 2).  The Σ stage walks the pole axis in
    batches so no complete pole tensor ever exists on host or device, and
    it walks it TWICE per iteration — once for the census that plans the
    windows, once for the spatial executor.  Called through
    :func:`read_poles`, each batch opened and closed its own h5py handle
    (ledger), its own collective ``SlabIO``, and a THIRD h5py handle for
    the unfold tables — and the third one landed *between* the two, so the
    per-batch sequence alternated h5py → FFI → h5py on one file, through
    two independent HDF5 library instances, once per batch.

    This reader collapses that to: read the ledger and the unfold tables
    with h5py FIRST (two opens, or one when the store is not wedge-packed),
    CLOSE h5py, then hold ONE ``SlabIO`` open for every batch of the
    iteration.  Two properties, and the second is the one that matters more
    than the arithmetic:

    * the churn drops from ``3·n_batches`` opens per walk to ``2 + 1``;
    * **no h5py open happens while the collective handle is live**, so
      the alternation the two libraries cannot survive does not occur at
      all inside a Σ stage.

    HOW MUCH OF THAT IS MACHINE-ENFORCED, precisely — because the sentence
    here used to claim all of it.  This reader opens ``SlabIO`` READ-ONLY,
    and :mod:`file_io.hdf5_owner` refuses a cross-stack overlap only when
    one side can WRITE.  So a stray h5py **write** open on this path while
    the reader is alive refuses by name; a stray h5py **read** open is
    ALLOWED BY DESIGN and merely counted.  The no-h5py-while-live property
    above is upheld by this class's own ordering, and by the registry only
    for writers.

    The handle is released in a ``finally`` — use it as a context manager,
    or call :meth:`close` from one.  A refusal raised mid-walk (an
    uncertified fit, a bad pole range) must still release the collective
    handle on every rank, because the next collective call on this mesh
    would otherwise rendezvous against a file this rank never closed.
    """

    def __init__(self, src, *, mesh_xy, allow_partial=False, mode="r"):
        if mesh_xy is None:
            raise ValueError(
                "PoleReader requires mesh_xy: it exists to hold ONE "
                "collective handle open across an iteration's pole "
                "batches, and a mesh-less read has no such handle.  Use "
                "read_poles(src, pole_slice=...) for the host path.")
        self.src = src
        self.mesh_xy = mesh_xy
        # h5py FIRST, and completely, and closed — before any collective
        # handle exists.  Both reads are small and neither is repeated.
        with _h5(src, mode) as grp:
            self.ledger = fit_completion_ledger(grp)
            _refuse_unfinalized(grp, self.ledger, allow_partial, "PoleReader")
        self.tables = (read_fit_unfold_tables(src)
                       if self.ledger["q_storage"] == "ibz" else None)
        self.n_poles = int(self.ledger["n_p"])
        from file_io.slab_io import SlabIO
        self._io = SlabIO(src, mode="r", mesh=mesh_xy)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        io, self._io = getattr(self, "_io", None), None
        if io is not None:
            io.close()

    def read(self, pole_slice=None, *, unfold=False, return_sharded=False,
             to_unit=None):
        """One contiguous pole range, through the handle already open."""
        if self._io is None:
            raise ValueError(
                "PoleReader.read after close(): the collective handle this "
                "reader owns is gone.  Open a new reader rather than "
                "reopening this one — a reader whose handle can be revived "
                "is a reader whose lifetime nobody can read off the code.")
        from jax.sharding import PartitionSpec as P

        lo, hi = _pole_range(self.ledger, pole_slice, "PoleReader.read")
        shape = (hi - lo, *_pole_read_shape(self.ledger, self.mesh_xy))
        Omega = self._io.read_slab(
            "Omega_p", shape=shape, offset=(lo, 0, 0, 0),
            partition_spec=P(None, None, "x", "y"))
        Bp = self._io.read_slab(
            "B_p", shape=shape, offset=(lo, 0, 0, 0),
            partition_spec=P(None, None, "x", "y"))
        return _finish_pole_read(
            self.src, Omega, Bp, self.ledger, mesh_xy=self.mesh_xy,
            unfold=unfold, return_sharded=return_sharded, to_unit=to_unit,
            tables=self.tables)


def open_pole_reader(src, *, mesh_xy, allow_partial=False, mode="r"):
    """A :class:`PoleReader` for one iteration's pole walk.

    COLLECTIVE: this opens a ``SlabIO`` handle, so every rank must call it,
    in the same order, and every rank must close it — use ``with``, or a
    ``close()`` in a ``finally``.  ``src`` must be a PATH.  While the
    reader lives, do not open this path with h5py (see :class:`PoleReader`
    for exactly how much of that the registry enforces).
    """
    return PoleReader(src, mesh_xy=mesh_xy, allow_partial=allow_partial,
                      mode=mode)


def read_poles(
    src,
    *,
    pole_slice=None,
    mesh_xy=None,
    unfold=False,
    return_sharded=False,
    to_unit=None,
    allow_partial=False,
    mode="r",
):
    """Read one contiguous pole range with two collective SlabIO reads.

    The leading pole axis is always retained.  ``pole_slice=None`` reads it
    completely; an integer reads a length-one range.  A mesh read is always
    sharded — pass ``return_sharded=True`` with ``mesh_xy``; there is no
    gather path.

    THE SUMMARY LINE DESCRIBES ONE OF TWO BRANCHES, and the contract
    changes with ``mesh_xy``:

    * ``mesh_xy`` given — COLLECTIVE, two ``SlabIO`` reads through
      :func:`open_pole_reader`; ``src`` must be a PATH; returns two sharded
      ``jax.Array``s of ``(n_poles, n_q, n_mu_padded, n_mu_padded)`` on
      ``P(None, None, 'x', 'y')``, where the pad is
      ``runtime.padding.padded_mu_extent``, not ``mesh_divisible_shape``.
    * ``mesh_xy=None`` — SERIAL, rank-local, no SlabIO and nothing
      collective; ``src`` may be a path or an open h5py group; returns two
      HOST numpy arrays at the LOGICAL ``(n_poles, n_q, n_mu, n_mu)``.

    Returns the 2-tuple ``(Omega_p, B_p)`` either way.  This is NOT the
    sole pole reader — :func:`read_fit_block`, :func:`read_fit_tensors` and
    :meth:`PoleReader.read` also read these datasets, and the production Σ
    reader is the last of those; this function has no ``src`` caller.

    ONE range per call, so a caller reading SEVERAL ranges of one file —
    every Σ stage does — opens and closes the store once per range.  Use
    :func:`open_pole_reader` there instead: it holds one collective handle
    across the whole walk and does its h5py reads before that handle
    exists (audit A1).  This function is the single-range door, and is
    that reader with a lifetime of one call.
    """
    if mesh_xy is not None and not return_sharded:
        raise ValueError(
            "read_poles: got mesh_xy with return_sharded=False; want "
            "return_sharded=True on every mesh read — the collective "
            "read lands sharded and every mesh caller consumes that "
            "layout, so no gather path exists.  Fix: pass "
            "return_sharded=True, or drop mesh_xy for a host-side read.")
    if mesh_xy is not None:
        with open_pole_reader(src, mesh_xy=mesh_xy,
                              allow_partial=allow_partial, mode=mode) as rd:
            return rd.read(pole_slice, unfold=unfold,
                           return_sharded=return_sharded, to_unit=to_unit)

    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial, "read_poles")
        lo, hi = _pole_range(ledger, pole_slice, "read_poles")
        Omega = np.asarray(grp["Omega_p"][lo:hi])
        Bp = np.asarray(grp["B_p"][lo:hi])
    return _finish_pole_read(
        src, Omega, Bp, ledger, mesh_xy=None, unfold=unfold,
        return_sharded=return_sharded, to_unit=to_unit)
