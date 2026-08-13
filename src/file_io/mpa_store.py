"""Frequency-resolved W restart tensors, and the staged B/Ω fit store.

STAGING LOCATION.  This module is deliberately dependency-light and
MOVABLE.  It lives in ``src/file_io/`` because that is where the restart
format layer lives today, but the multipole-W work is a staging area:
when the MPA stage settles, both this module and ``gw/mpa/tiling.py``
are expected to move as a pair (most likely into a ``mpa`` service
alongside ``symmetry_maps``).  Nothing here imports from ``gw``, and the
``symmetry_maps`` DOOR — not its submodules — is imported LAZILY through
:func:`_qs`, so the move costs an import line rather than a redesign and
importing ``file_io`` still costs no jax.

WHAT THIS FORMAT IS.  The multipole-W fit needs W_c evaluated on the
double-parallel sampling grid — ~2·n_p complex frequencies on two lines
ϖ₁ and ϖ₂ (MPA_THEORY_PLAN §B) — before it can fit anything, and the
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
:func:`read_w_slab` is that read, and the removability claim has a test.

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
million.  So: the writer stamps 2, and :func:`read_qirr_tensor` is the
WIDENED reader that accepts {1, 2} and discriminates on the RANK before
it looks at anything else.  See that function for the full argument.

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
:func:`read_w_columns` is that read and :func:`choose_column_budget` is
its arithmetic; the budget is sized so the returned block costs about
what ONE (N_μ, N_μ) tile costs, which is the unit the owner's
constraint is stated in.  The block is 1-D SHARDED ON THE ROW AXIS
ONLY — never 2-D — because the fit is elementwise in (μ, ν) and a
second split on the column axis buys nothing while making every rank's
column count a function of the mesh shape.

Testing note: everything below is exercised host-side with plain h5py
at LOGICAL extents.  The phdf5 FFI is not built on WSL, so the format
is tested at its seams the way the symmetry lane tested the q_irr
format; the ``SlabIO`` write path (where each rank contributes its own
(μ, ν) hyperslab and no rank holds the whole array) gets its Perlmutter
leg when this is integrated, and :func:`stamp_w_omega` exists for
exactly that split — the producer writes the bytes with the machinery
it already has and this stamps them.
"""

from __future__ import annotations

import datetime
import hashlib

import numpy as np

#: The frequency-resolved layout.  Registered ALONGSIDE version 1 rather
#: than replacing it: a v1 file is still a v1 file and still reads, and
#: :data:`QIRR_FORMAT_VERSIONS_READABLE` is the set the widened reader
#: accepts.  The value lives here rather than in ``qirr_store`` only
#: while this module is staged outside the format layer; when it moves,
#: it moves next to ``QIRR_FORMAT_VERSION``.
QIRR_FORMAT_VERSION_FREQ = 2

#: Every version :func:`read_qirr_tensor` will read.  A reader that
#: reads an unknown version best-effort returns wrong numbers on the day
#: the layout changes; a reader that accepts a KNOWN version without
#: checking the rank returns wrong numbers on the day the layout gains
#: an axis, which is this day.
QIRR_FORMAT_VERSIONS_READABLE = (1, 2)

#: Rank of the stored dataset THIS format adds — version 2 is
#: ``(n_omega, n_q, N_μ, N_μ)``.  The rank is the discriminant, not a
#: consistency nicety; see :func:`read_qirr_tensor`.
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
#: and a v2 file's attrs must equal a v1 file's attr for attr.  Written
#: out as a list rather than matched by an ``mpa_`` prefix so that an
#: attr added later to one format and not the other fails the comparison
#: instead of being swallowed by a ``startswith``.  (``mpa_writer`` is
#: deliberately absent: it says BY WHAT, not WHAT, and belongs with the
#: timestamps the comparison already exempts.)
_MPA_OWNED_ATTRS = (
    _FREQ_ATTR, "mpa_n_omega", "mpa_omega_units", "mpa_protocol",
    "mpa_varpi", "mpa_n_p", "mpa_alpha", "mpa_omega_max", "mpa_grid_hash",
)

#: Bytes per complex128 element.  Named because it appears in the budget
#: arithmetic, and a budget whose constants are anonymous is a budget
#: nobody can check against a message.
COMPLEX128_BYTES = 16

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
    with qs.QirrDest(dest, mode) as grp:
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
    """
    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    shape, dtype = _w_storage_geometry(
        n_omega, n_q_on_disk, n_mu, dtype, closure_verdict,
        where=f"allocate_w_omega_collective({name!r})")
    with SlabIO(dest, mode=mode, mesh=mesh_xy) as io:
        io.create_dataset(name, shape=shape, dtype=dtype)

    header = None
    if process_rank() == 0:
        header = stamp_w_omega(
            dest, name, tables=tables, omega=omega, sampling=sampling,
            omega_line=omega_line, closure_verdict=closure_verdict,
            data_ready=np.zeros(shape[0], dtype=bool),
            n_rmu_logical=n_rmu_logical, provenance=provenance,
            energy_unit=energy_unit)
    barrier("mpa_w_omega_allocated")
    return header


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

    with qs.QirrDest(dest, mode) as grp:
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
        ds.attrs["mpa_protocol"] = san["protocol"]
        ds.attrs["mpa_varpi"] = san["varpi"]
        ds.attrs["mpa_n_p"] = np.int64(san["n_p"])
        ds.attrs["mpa_alpha"] = np.int64(san["alpha"])
        ds.attrs["mpa_omega_max"] = np.float64(san["omega_max"])
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
    with qs.QirrDest(dest, mode) as grp:
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

    ``W_q_munu`` remains on its ``P(None, 'x', 'y')`` layout.  SlabIO
    inserts the singleton frequency axis, clips only the device-dependent
    mu padding against ``global_shape``, and closes collectively before rank
    zero updates the small readiness ledger.  A failed data write therefore
    leaves the slab explicitly unready.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common.collectives import barrier, process_rank
    from file_io.slab_io import SlabIO

    shape = tuple(int(n) for n in global_shape)
    if len(shape) != 4:
        raise ValueError(
            "write_w_slab_collective: global_shape must be "
            "(n_omega,n_q,n_mu,n_mu)")
    expected = NamedSharding(mesh_xy, P(None, "x", "y"))
    if getattr(W_q_munu, "sharding", None) != expected:
        raise ValueError(
            "write_w_slab_collective requires W on P(None,'x','y'); "
            "the producer must return the native SlabIO layout rather "
            "than resharding a bulk tensor at the writer seam")
    W4 = W_q_munu[None, ...]
    with SlabIO(dest, mode="a", mesh=mesh_xy) as io:
        io.write_slab(
            name, W4, offset=(int(i_omega), 0, 0, 0),
            global_shape=shape)
    del W4

    n_ready = None
    if process_rank() == 0:
        with _qs().QirrDest(dest, "a") as grp:
            ds, mgrp = _open_w(grp, name)
            n_ready = _mark_w_slab_ready(ds, mgrp, i_omega, ready)
    barrier("mpa_w_slab_ready")
    return n_ready


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

    Returns a plain dict.  Every cross-check the format owns runs here,
    so a caller that got a header back has already been told the file is
    self-consistent, and every reader below calls this first rather than
    repeating the checks — one implementation of "what does this file
    say", because a second one is how a reader ends up disagreeing with
    the format about what it is holding.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        ds, mgrp = _open_w(grp, name)
        version = _refuse_unless_rank_matches_version(ds, name)
        if version != QIRR_FORMAT_VERSION_FREQ:
            raise ValueError(
                f"mpa_store: {name!r} is format version {version}; the "
                f"frequency-resolved readers are version "
                f"{QIRR_FORMAT_VERSION_FREQ}.  Use "
                f"qirr_store.read_tensor for a version-1 tensor, or "
                f"read_qirr_tensor to dispatch on the version.")

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

        sampling = {
            "protocol": qs.qirr_attr_str(ds, "mpa_protocol"),
            "varpi": np.asarray(ds.attrs["mpa_varpi"], dtype=np.float64),
            "n_p": int(ds.attrs["mpa_n_p"]),
            "alpha": int(ds.attrs["mpa_alpha"]),
            "omega_max": float(ds.attrs["mpa_omega_max"]),
        }
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
    enforce rank 3 under a version-1 stamp, and the docstring of
    :func:`read_qirr_tensor` registered the reason as a follow-up in as
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


def read_qirr_tensor(src, name, *, mode="r", **kw):
    """THE WIDENED READER: version 1 or 2, dispatched on the RANK.

    ``qirr_store.read_tensor`` is the version-1 reader, and it refuses
    any other version AND any rank but 3 under its own stamp — the hole
    the frequency axis opened is closed there, at the reader every
    unsuspecting consumer already calls, rather than here.  See
    :func:`_refuse_unless_rank_matches_version` for the mechanism and for
    why a version stamp alone does not close it.

    So this is the reader a caller who may be handed EITHER layout should
    ask, and what it adds is the dispatch plus version 2's own checks,
    run before the tables are opened:

    * version 1, rank 3 -> ``qirr_store.read_tensor``, untouched.  Every
      keyword goes straight through, that reader makes its own rank
      refusal, and the bytes that come back are the bytes that came back
      before this module existed.
    * version 2, rank 4 -> :func:`read_w_omega`, which returns the whole
      frequency-resolved tensor.  Callers that want one slab or a few
      columns should ask for those directly; this path exists so a
      generic consumer is never SILENTLY wrong, not because reading all
      of ω at once is a good idea.
    * anything else -> refuse, naming the rank and the version.

    A file with no version attr at all is legacy full-BZ and is
    delegated to ``qirr_store.read_tensor`` unchanged, which is where
    the no-attr-means-full rule lives.

    THAT FOLLOW-UP IS DISCHARGED.  This docstring used to register one:
    the version-1 rank check belonged INSIDE ``qirr_store.read_tensor``
    rather than in a wrapper, because a wrapper protects the callers who
    use it and nobody else, and it sat here only because the symmetry
    checkpoint carrying that reader was still landing.  The checkpoint
    landed with the refusal in ``read_tensor``, and this function is now
    what the note asked it to become: the DISPATCHER, plus the two
    checks that are genuinely this format's own — version 2's rank, and
    the ``mpa_freq_axis`` cross-check in both directions.  See
    :func:`_refuse_unless_rank_matches_version`.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        if name not in grp:
            raise KeyError(f"mpa_store: {name!r} is not in this file")
        ds = grp[name]
        if qs.QIRR_VERSION_ATTR not in ds.attrs:
            return qs.read_tensor(grp, name, **kw)
        version = _refuse_unless_rank_matches_version(ds, name)
        if version == QIRR_FORMAT_VERSION_FREQ:
            return read_w_omega(grp, name, **kw)
        return qs.read_tensor(grp, name, **kw)


def read_w_omega(src, name, *, require_ready=True, mode="r", **kw):
    """The WHOLE (n_omega, n_q, N_μ, N_μ) tensor, slab by slab.

    Present for the widened dispatcher and for tests, and it says so:
    the owner's constraint is that all of ω does NOT fit in memory, so a
    production consumer wants :func:`read_w_slab` or
    :func:`read_w_columns`.  Reading everything is the thing the format
    exists to make unnecessary.
    """
    header = read_w_header(src, name, mode=mode)
    slabs = []
    for i in range(header["n_omega"]):
        arr, _ = read_w_slab(src, name, i, require_ready=require_ready,
                             mode=mode, **kw)
        slabs.append(np.asarray(arr))
    return np.stack(slabs, axis=0), header


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

    with qs.QirrDest(src, mode) as grp:
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
    """
    return _qs().read_tables(src, name, mode=mode)


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

        n_cols = floor(tile_bytes / (n_omega * N_mu * itemsize))

    which for the default budget collapses to ``n_cols = N_mu //
    n_omega`` — the frequency axis is paid for out of the column count,
    one for one.  At the Si production scale, N_μ = 480 and n_omega = 16
    (an n_p = 8 fit's 2·n_p samples): the tile is 480·480·16 =
    3 686 400 B (3.52 MiB), the per-column cost across ω is 16·480·16 =
    122 880 B, and the budget is exactly **30 columns** — 16·480·30·16 =
    3 686 400 B, the tile to the byte.

    Clamped to at least 1: a budget of zero columns is not a budget, it
    is a refusal to make progress, and the honest failure for a grid so
    long that one column busts a tile is to hand back 1 and let the
    caller see the cost in :func:`describe_column_cost`.

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
    """
    n_mu = int(n_mu)
    n_omega = int(n_omega)
    n_cols = int(n_cols)
    budget = one_tile_bytes(n_mu, itemsize) if tile_bytes is None \
        else int(tile_bytes)
    cost = n_omega * n_mu * n_cols * int(itemsize)
    allowed = choose_column_budget(n_mu, n_omega, tile_bytes, itemsize)
    return (
        f"{n_cols} columns at n_omega={n_omega}, N_mu={n_mu} costs "
        f"{n_omega}*{n_mu}*{n_cols}*{int(itemsize)} B = {cost} B "
        f"({cost / 2 ** 20:.2f} MiB) against a budget of "
        f"{'one (N_mu, N_mu) tile, ' if tile_bytes is None else ''}"
        f"{n_mu}*{n_mu}*{int(itemsize)} B = {budget} B "
        f"({budget / 2 ** 20:.2f} MiB) — a ratio of "
        f"{cost / budget:.3f}x.  choose_column_budget({n_mu}, "
        f"{n_omega}) allows {allowed}.")


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
    uniq = np.unique(cols)
    if uniq.size != cols.size:
        dup = sorted(set(cols.tolist()))
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
    n_omega = header["n_omega"]
    cols = normalise_columns(mu_cols, n_mu)
    budget = choose_column_budget(n_mu, n_omega, tile_bytes)
    if int(cols.size) > budget:
        raise ValueError(
            f"read_w_columns({name!r}): refusing "
            f"{int(cols.size)} columns.  " +
            describe_column_cost(n_mu, n_omega, int(cols.size),
                                 tile_bytes) +
            f"  A block spanning all {n_omega} frequencies is priced to "
            f"be ONE of the small number of W_q(μ,ν) copies that fit at "
            f"once; pass tile_bytes= to raise the budget deliberately, "
            f"or loop the columns in blocks of {budget}.")

    if require_ready and header["n_ready"] != n_omega:
        missing = np.flatnonzero(~header["data_ready"])
        raise ValueError(
            f"read_w_columns({name!r}): {len(missing)} of {n_omega} "
            f"frequency slabs are not ready (indices "
            f"{missing[:8].tolist()}{'...' if len(missing) > 8 else ''})."
            f"  A column block spans the WHOLE frequency axis, so a fit "
            f"run on the ready half of the grid returns poles that are "
            f"wrong rather than absent — the unwritten slabs read as "
            f"zeros and the Padé solve happily fits them.  Fill the "
            f"grid, or pass require_ready=False to inspect it.")

    iq = int(q)
    if not 0 <= iq < header["n_q_on_disk"]:
        raise IndexError(
            f"read_w_columns({name!r}): q={iq} is outside [0, "
            f"{header['n_q_on_disk']}); the tensor is stored on the "
            f"{header['q_storage']} q axis.")

    # ONE HYPERSLAB, NOT ONE PER FREQUENCY.  A contiguous run becomes a
    # slice (HDF5 reads it as a single hyperslab); anything else is a
    # point selection on the LAST axis only, which h5py supports and
    # which keeps the row axis whole — the axis the caller shards.
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    contiguous = (hi - lo) == int(cols.size)
    with qs.QirrDest(src, mode) as grp:
        ds = grp[name]
        if contiguous:
            block = ds[:, iq, :, lo:hi]
        else:
            block = ds[:, iq, :, cols.tolist()]
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


# ---------------------------------------------------------------------------
# The staged B/Ω fit store
# ---------------------------------------------------------------------------

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
        and Na 8, Cu 12 (MPA_THEORY_PLAN §B).
    grid_hash, table_hash, centroid_hash
        The W(ω) file's stamps, carried here so the Σ stage can assert
        that these poles came from that screening on that centroid set.
        Optional only because a synthetic fit has no such file.
    """
    qs = _qs()
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
    with qs.QirrDest(dest, mode) as grp:
        # EVERY ``fit_*`` GOES, not just the two required ones.  A
        # re-allocation that left an earlier run's extra diagnostic
        # behind would leave a full-size array of ITS numbers indexed by
        # THIS run's ledger, and the Σ stage would certify the new poles
        # against the old evidence.
        for key in ["Omega_p", "B_p", MPA_FIT_SUFFIX, MPA_HEAD_SUFFIX] + [
                k for k in grp if str(k).startswith("fit_")]:
            if key in grp:
                del grp[key]
        grp.create_dataset("Omega_p", shape=(n_p, n_q, n_mu, n_mu),
                           dtype=dtype)
        grp.create_dataset("B_p", shape=(n_p, n_q, n_mu, n_mu),
                           dtype=dtype)
        grp.create_dataset("fit_condition", shape=(n_q, n_mu, n_mu),
                           dtype=np.float64)
        grp.create_dataset("fit_backward_error", shape=(n_q, n_mu, n_mu),
                           dtype=np.float64)

        led = grp.create_group(MPA_FIT_SUFFIX)
        led.create_dataset("blocks_done", data=np.zeros((n_q, n_mu),
                                                        dtype=bool))
        led.create_dataset("block_journal",
                           shape=(0, 3), maxshape=(None, 3),
                           dtype=np.int64)
        for key in ("block_condition_max", "block_backward_error_max"):
            led.create_dataset(key, shape=(0,), maxshape=(None,),
                               dtype=np.float64)

        grp.attrs["mpa_fit_format_version"] = np.int64(
            MPA_FIT_FORMAT_VERSION)
        if energy_unit is not None:
            grp.attrs[FIT_ENERGY_UNIT_ATTR] = str(energy_unit)
        grp.attrs["mpa_fit_n_p"] = np.int64(n_p)
        grp.attrs["mpa_fit_n_q"] = np.int64(n_q)
        grp.attrs["mpa_fit_n_mu_logical"] = np.int64(n_mu)
        grp.attrs["mpa_fit_complete"] = False
        grp.attrs["mpa_fit_writer"] = "file_io.mpa_store"
        grp.attrs["mpa_fit_generator_commit"] = qs.qirr_generator_commit()
        grp.attrs["mpa_fit_allocated_utc"] = _utc_now()
        for label, val in (("grid_hash", grid_hash),
                           ("table_hash", table_hash),
                           ("centroid_hash", centroid_hash)):
            if val is not None:
                grp.attrs["mpa_fit_w_" + label] = str(val)
        for key, val in (provenance or {}).items():
            grp.attrs["prov_" + str(key)] = val
        if unfold_tables is not None:
            stamp_fit_unfold_tables(grp, unfold_tables)
        return fit_completion_ledger(grp)


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
    model="multipole",
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
    with qs.QirrDest(dest, mode) as grp:
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
        head.attrs["ready"] = True


def read_head_fit(src, *, to_unit=None, mode="r"):
    """Read the complete scalar q->0 MPA fit; refuse absent/partial data."""
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
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
    if to_unit is not None:
        if source_unit not in FIT_ENERGY_UNITS or to_unit not in FIT_ENERGY_UNITS:
            raise ValueError(
                "head read: both stored and requested energy units must be "
                f"declared; stored={source_unit!r}, requested={to_unit!r}")
        scale = FIT_ENERGY_UNITS[source_unit] / FIT_ENERGY_UNITS[to_unit]
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
    with qs.QirrDest(dest, mode) as grp:
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

        lo, hi = int(cols[0]), int(cols[-1]) + 1
        contiguous = (hi - lo) == int(cols.size)
        sel = slice(lo, hi) if contiguous else cols.tolist()
        grp["Omega_p"][:, iq, :, sel] = Om
        grp["B_p"][:, iq, :, sel] = Bp
        grp["fit_condition"][iq, :, sel] = diag["condition"]
        grp["fit_backward_error"][iq, :, sel] = diag["backward_error"]
        for key, arr in diag.items():
            if key in ("condition", "backward_error"):
                continue
            name = "fit_" + key
            if name not in grp:
                grp.create_dataset(name, shape=(n_q, n_mu, n_mu),
                                   dtype=np.float64)
            grp[name][iq, :, sel] = arr

        done[iq, cols] = True
        led["blocks_done"][...] = done
        # THE JOURNAL RECORDS THE SPAN, ``blocks_done`` RECORDS THE
        # TRUTH.  ``fit_schedule`` only ever emits contiguous blocks, so
        # for a normal walk the two agree exactly; a caller that hands a
        # scattered selection gets a span WIDER than its column count,
        # which is why the ledger and not the journal is what
        # :func:`finalize_fit_store` and :func:`read_fit_block` refuse
        # on.  A sentinel in the journal would have made "which columns"
        # a question with two answers.
        _append(led["block_journal"],
                np.array([[iq, lo, hi]], dtype=np.int64))
        _append(led["block_condition_max"],
                np.array([float(np.max(diag["condition"]))]))
        _append(led["block_backward_error_max"],
                np.array([float(np.max(diag["backward_error"]))]))
        return fit_completion_ledger(grp)


def _canonical_diagnostics(diag_block, n_rows, n_cols):
    """The per-block fit diagnostics, validated and float64.

    Condition number and backward error are the two the Σ stage's
    certification is stated in (MPA_THEORY_PLAN §B: "condition numbers
    and backward error, diagonal/off-diagonal and norm-resolved
    distributions"), so they are REQUIRED and everything else is extra.
    """
    if not isinstance(diag_block, dict):
        raise TypeError(
            f"write_fit_block: diag_block must be a dict with "
            f"'condition' and 'backward_error'; got "
            f"{type(diag_block).__name__}")
    missing = [k for k in ("condition", "backward_error")
               if k not in diag_block]
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

    ``blocks_done`` is the authority (one bool per (q, column));
    ``journal`` is the append-only record of the order they arrived in,
    with each block's worst condition and backward error beside it.  The
    two are not redundant: the ledger answers "is this column done",
    the journal answers "what did the block that did it look like", and
    the Σ stage's certification needs the second.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        led = _open_fit(grp)
        done = np.asarray(led["blocks_done"][()], dtype=bool)
        journal = np.asarray(led["block_journal"][()], dtype=np.int64)
        cond = np.asarray(led["block_condition_max"][()], dtype=np.float64)
        berr = np.asarray(led["block_backward_error_max"][()],
                          dtype=np.float64)
        certification = {
            str(key)[len("mpa_cert_"):]: grp.attrs[key]
            for key in grp.attrs if str(key).startswith("mpa_cert_")
        }
        provenance = {
            str(key)[len("prov_"):]: grp.attrs[key]
            for key in grp.attrs if str(key).startswith("prov_")
        }
        return {
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
            "block_condition_max": cond,
            "block_backward_error_max": berr,
            "condition_max": float(cond.max()) if cond.size else None,
            "backward_error_max": float(berr.max()) if berr.size else None,
            "finalized_utc": qs.qirr_attr_str(grp, "mpa_fit_finalized_utc"),
            "certification": certification,
            "provenance": provenance,
        }


def validate_fit_store(src, *, expected_identity=None, mode="r"):
    """Validate the finalized fit contract before Sigma reads pole bytes.

    ``expected_identity`` may name ``w_grid_hash``, ``w_table_hash`` and
    ``w_centroid_hash`` from the screening object currently in use.  The
    fit's own declared ``*_max_allowed`` certification thresholds are always
    enforced against its observed maxima.
    """
    ledger = fit_completion_ledger(src, mode=mode)
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
    observed = {
        "condition_max": ledger["condition_max"],
        "backward_error_max": ledger["backward_error_max"],
    }
    for key, allowed in ledger["certification"].items():
        if not str(key).endswith("_allowed"):
            continue
        metric = str(key)[:-len("_allowed")]
        if metric not in observed:
            continue
        got = observed[metric]
        if got is None or float(got) > float(allowed):
            raise ValueError(
                f"MPA fit failed its stored certification: {metric}="
                f"{got!r} exceeds {allowed!r}")
    return ledger


def finalize_fit_store(dest, *, certification=None, mode="a"):
    """Stamp the store COMPLETE — once, and only when it is.

    Refuses a store with an unfitted (q, column) and NAMES the gaps,
    because "which columns are missing" is the only question a driver
    that crashed halfway actually has.  Refuses a second finalize for
    the same reason a second stamp is refused everywhere in this
    format: two claims about what the file says, differing on the day
    one of them is made against a different state.

    ``certification`` is stamped as ``mpa_cert_*`` — the thresholds the
    Σ stage should hold these poles to.  The OBSERVED maxima are
    stamped regardless, so a consumer can refuse on the evidence even
    when nobody declared a threshold.
    """
    qs = _qs()
    with qs.QirrDest(dest, mode) as grp:
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
        cond = np.asarray(led["block_condition_max"][()])
        berr = np.asarray(led["block_backward_error_max"][()])
        grp.attrs["mpa_fit_complete"] = True
        grp.attrs["mpa_fit_finalized_utc"] = _utc_now()
        grp.attrs["mpa_fit_condition_max"] = np.float64(
            cond.max() if cond.size else 0.0)
        grp.attrs["mpa_fit_backward_error_max"] = np.float64(
            berr.max() if berr.size else 0.0)
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
    with qs.QirrDest(src, mode) as grp:
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
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        contiguous = (hi - lo) == int(cols.size)
        sel = slice(lo, hi) if contiguous else cols.tolist()
        Om = np.asarray(grp["Omega_p"][:, iq, :, sel])
        Bp = np.asarray(grp["B_p"][:, iq, :, sel])
        diag = {
            "condition": np.asarray(grp["fit_condition"][iq, :, sel]),
            "backward_error": np.asarray(
                grp["fit_backward_error"][iq, :, sel]),
        }
        for key in grp:
            if str(key).startswith("fit_") and key not in (
                    "fit_condition", "fit_backward_error"):
                diag[str(key)[len("fit_"):]] = np.asarray(
                    grp[key][iq, :, sel])
        return Om, Bp, diag, ledger


def read_fit_tensors(src, *, allow_partial=False, mode="r"):
    """The whole ``(Omega_p, B_p, diagnostics, ledger)``.

    For the Σ stage, which consumes every pole of every element at a q,
    and for tests.  Same finalize refusal as :func:`read_fit_block`.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
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
    with qs.QirrDest(dest, mode) as grp:
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
    with qs.QirrDest(src, mode) as grp:
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
):
    """Apply the one unit/unfold/gather policy shared by pole readers."""
    if to_unit is not None:
        source_unit = ledger["energy_unit"]
        if source_unit not in FIT_ENERGY_UNITS or to_unit not in FIT_ENERGY_UNITS:
            raise ValueError(
                "pole read: both stored and requested energy units must be "
                f"declared; stored={source_unit!r}, requested={to_unit!r}")
        scale = FIT_ENERGY_UNITS[source_unit] / FIT_ENERGY_UNITS[to_unit]
        Omega, Bp = Omega * scale, Bp * scale

    if unfold and ledger["q_storage"] == "ibz":
        import jax.numpy as jnp

        if mesh_xy is None:
            raise ValueError("a wedge pole unfold requires mesh_xy")
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
    if return_sharded:
        if mesh_xy is None:
            raise ValueError("return_sharded requires mesh_xy")
        return Omega, Bp
    if mesh_xy is not None:
        from common.collectives import gather_to_host
        return gather_to_host(Omega), gather_to_host(Bp)
    return Omega, Bp


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
    completely; an integer reads a length-one range.  This is the sole pole
    tensor reader—the singular/plural compatibility wrappers below add no
    I/O policy of their own.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial, "read_poles")
        if pole_slice is None:
            lo, hi = 0, ledger["n_p"]
        elif isinstance(pole_slice, (int, np.integer)):
            lo, hi = int(pole_slice), int(pole_slice) + 1
        else:
            lo, hi, step = pole_slice.indices(ledger["n_p"])
            if step != 1:
                raise ValueError("read_poles requires a contiguous pole slice")
        if not (0 <= lo < hi <= ledger["n_p"]):
            raise IndexError(
                f"read_poles: pole range [{lo},{hi}) is outside "
                f"[0,{ledger['n_p']})")
        if mesh_xy is None:
            Omega = np.asarray(grp["Omega_p"][lo:hi])
            Bp = np.asarray(grp["B_p"][lo:hi])

    if mesh_xy is not None:
        from jax.sharding import PartitionSpec as P
        from runtime.padding import padded_mu_extent
        from file_io.slab_io import SlabIO

        n_pad = int(padded_mu_extent(ledger["n_mu"], mesh_xy))
        shape = (hi - lo, ledger["n_q"], n_pad, n_pad)
        with SlabIO(src, mode="r", mesh=mesh_xy) as io:
            Omega = io.read_slab(
                "Omega_p", shape=shape, offset=(lo, 0, 0, 0),
                partition_spec=P(None, None, "x", "y"))
            Bp = io.read_slab(
                "B_p", shape=shape, offset=(lo, 0, 0, 0),
                partition_spec=P(None, None, "x", "y"))
    return _finish_pole_read(
        src, Omega, Bp, ledger, mesh_xy=mesh_xy, unfold=unfold,
        return_sharded=return_sharded, to_unit=to_unit)


def read_pole_slices(src, **kwargs):
    """Compatibility name for :func:`read_poles` with the full axis."""
    return read_poles(src, **kwargs)


def read_pole_slice(src, p, **kwargs):
    """Read one leading pole slab; production reads stay sharded via SlabIO.

    ``to_unit='Ry'`` performs the fit-axis conversion once at this I/O seam.
    A wedge unfold uses the stored ``symmetry_maps`` tables and the general
    pair-transpose TRS rule, required because complex-frequency W is not
    Hermitian.
    """
    Omega, Bp = read_poles(src, pole_slice=int(p), **kwargs)
    return Omega[0], Bp[0]
