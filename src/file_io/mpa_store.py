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
import uuid

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

#: Bump when the fit store's layout changes.  Independent of the W
#: format's version: the two files have separate lifetimes and a reader
#: of one is not a reader of the other.
#:
#: v2 ADDS THE q -> 0 HEAD AXIS and nothing else.  The head has always
#: travelled beside W as a small object whose length is the number of
#: frequencies the scheme needs -- 1 for COHSEX, 2 for the two-point PPM
#: fits (``gw_output.persist_w0_and_head``) -- and MPA needs it at every
#: one of its ``2*n_p`` samples, plus the ``n_p`` head POLES those samples
#: are fitted to, because the Sigma stage consumes poles and not samples.
#: A store written before this axis existed is not upgraded and is not
#: guessed at: it reads back fine for everything that does not need the
#: head, and :func:`read_head_poles` refuses the MPA Sigma path by name.
#: That refusal is the point of the version, not a side effect of it --
#: a Sigma built from a headless store is silently missing the q -> 0
#: term at every frequency, which is a 200 meV error with no symptom.
MPA_FIT_FORMAT_VERSION = 2

#: Versions this reader accepts.  v1 stores predate the head axis; they
#: are readable and their poles are unchanged, so refusing them outright
#: would strand the first-light field for no gain.
MPA_FIT_READABLE_VERSIONS = (1, 2)

#: The name the fit store's q-wedge unfold tables are filed under, so
#: that ``symmetry_maps.qirr_store.read_tables`` reads them with its own
#: code -- its own key list, its own missing-table refusal, its own
#: digest.  ``"Omega_p"`` and not ``"B_p"`` because the tables describe a
#: q axis both datasets share and a group has to be filed under one of
#: them; the pole POSITIONS are the half that carries no phase, so they
#: are the honest owner of a table group that is pure geometry.
#:
#: THE TABLES ARE NOT A VERSION BUMP, and the criterion is the one this
#: format already applies to its W side: a version exists when an OLDER
#: READER WOULD MISREAD the newer bytes.  A v2 reader handed a fit store
#: that carries this group ignores an unknown sibling group and reads
#: exactly the wedge it read before -- there is no shape it can mistake
#: for something else, because the poles themselves did not move.  What
#: refuses instead is the CONSUMER: a wedge fit store with no tables
#: cannot be unfolded and :func:`read_pole_slice` says so by name.
FIT_TABLE_OWNER = "Omega_p"

#: THE POLE-AXIS ENERGY UNIT, and why the store must declare it.  The
#: first end-to-end MPA Sigma dispatch found the fit solved against the
#: W(omega) store's abscissae -- stamped ``mpa_omega_units = "Ha"`` --
#: while the Sigma pass loop fed ``Re Omega_p`` straight into its window
#: planner beside Rydberg band energies and converted nothing.  Every
#: pole entered Sigma at HALF its energy, silently, and mis-sized the
#: width split and the Laplace buckets chosen from the same numbers.
#: Settled by an EXTERNAL ORACLE, not by reading the code: the n_p = 1
#: head pole reads 18.118 eV as Hartree against BerkeleyGW's own
#: 18.009 eV, and would read 9.06 eV as Rydberg against a 16.7 eV
#: measured plasmon.  No shape check, no finiteness check and no
#: internal consistency gate can see a factor like that -- the model is
#: invariant under rescaling z, Omega and B together -- so the unit is
#: an attr the WRITER stamps and the READERS convert on, exactly once,
#: and an unstamped store is refused by the converting readers rather
#: than guessed at.
#:
#: BOTH tensors convert by the SAME factor: ``B_p`` carries one power of
#: the frequency unit in its numerator (``W_c = sum_p B_p/(z - Omega_p)
#: - ...`` with ``W_c`` itself an energy in Ry by the producers'
#: convention), so Ha -> Ry doubles ``Omega_p`` AND ``B_p`` and leaves
#: ``W_c`` invariant.  The table-closer's rescaled twin store verified
#: the factor at an exact elementwise ratio of 2.000000.
FIT_ENERGY_UNITS = {"Ry": 1.0, "Ha": 2.0}

#: The attr carrying the declaration, on the fit store's root group.
FIT_ENERGY_UNIT_ATTR = "mpa_fit_energy_unit"

#: WHICH SCREENING OBJECT THE TENSOR IS, and why the store must declare
#: it.  ``W(z) = v + W_c(z)`` (MPA_THEORY 1.1) and the multipole method
#: fits the CORRELATION part ``W_c`` and nothing else (1.2): the pole
#: expansion ``W_c(z) = sum_p 2 Omega_p B_p/(z^2 - Omega_p^2)`` vanishes
#: as ``z -> infinity`` and its tau transform ``W_c(tau) = sum_p B_p
#: exp(-i Omega_p tau)`` is what the Sigma kernel convolves against
#: ``G(tau)``.  The bare ``v`` is frequency-INDEPENDENT: it belongs to
#: ``Sigma_x``, it is already counted there, and it has no place in a
#: pole field at all.
#:
#: THE DEFECT THIS RETIRES, measured on the 2026-08-09 production bridge.
#: The n_p = 1 gate -- where the multipole scheme IS Godby-Needs by
#: construction -- came back at ``Sigma_c = -130.651 eV`` against the
#: window-matched GN arm's ``+0.6754 eV``, a systematic -120.1 eV per
#: state.  The W(omega) store had been filled straight from the Dyson
#: solve, i.e. with the FULL W, while the two sibling arms both subtract
#: ``v`` by hand at their own seams -- ``ppm_sigma.fit_ppm``'s
#: ``Wc0_q = W0_q - V_q`` and ``head_dipole``'s ``w_c = w_head -
#: v_head`` -- and nothing anywhere said which object the body store
#: held.  Measured on that store, ``|v|`` is 104-119 % of ``|W|`` at the
#: probe frequency, so the pole field was overwhelmingly a fit to the
#: bare Coulomb interaction.
#:
#: NO INTERNAL GATE CAN SEE IT, which is why it is an attr and not a
#: check.  The fit reproduces whatever it was handed -- that store's
#: backward error is 4.0e-16 -- the tensor is Hermitian either way, the
#: k-star relation survives (``v_q`` is symmetry-covariant, and the
#: full-BZ Sigma cube carries exactly the 8 distinct band-8 values this
#: mesh admits, with the same partition as the GN arm), and the head is
#: untouched.  The only symptom is the number, against an external
#: oracle.  So the WRITER states which object it wrote and the READERS
#: refuse anything that is not the correlation part.
W_SCREENING_CONTENTS = ("W_c", "W")

#: The attr carrying it, on the W(omega) dataset.
W_SCREENING_CONTENT_ATTR = "mpa_w_screening_content"

#: The fit store's copy, carried across by the fit driver so the Sigma
#: stage can refuse a pole field WITHOUT re-opening the W file it came
#: from -- which by then may not exist.
FIT_SCREENING_CONTENT_ATTR = "mpa_fit_w_screening_content"


def canonical_screening_content(content, *, where):
    """Normalise a screening-content spelling to a known key, or refuse.

    Case-insensitive on input, canonical on output, and an unrecognised
    spelling refuses rather than defaulting -- a default here is the
    130 eV the declaration exists to kill.
    """
    if content is None:
        raise ValueError(
            f"{where}: screening_content is required and there is no "
            f"default.  The multipole method fits the CORRELATION part "
            f"W_c = W - v (MPA_THEORY 1.1/1.2) and the pole field's tau "
            f"transform is convolved with G as if it were; a store filled "
            f"with the full W from the Dyson solve is the 2026-08-09 "
            f"bridge defect, worth -120.1 eV per state and invisible to "
            f"every internal gate.  Pass 'W_c' (the tensor has had v "
            f"subtracted) or 'W' (it has not).")
    raw = str(content).strip()
    for known in W_SCREENING_CONTENTS:
        if raw.lower() == known.lower():
            return known
    raise ValueError(
        f"{where}: screening_content={content!r} is not one of "
        f"{list(W_SCREENING_CONTENTS)}.  Refusing rather than guessing: "
        f"an unrecognised spelling silently treated as the correlation "
        f"part is the same defect as no declaration at all, wearing a "
        f"declaration's clothes.")


def _declared_screening_content(holder, attr):
    """The stored declaration, or ``None`` for a legacy store."""
    if attr not in holder.attrs:
        return None
    return canonical_screening_content(
        _qs().qirr_attr_str(holder, attr),
        where="mpa_store (stored declaration)")


def require_correlation_part(content, *, where, source=None):
    """Refuse anything that is not a declared ``W_c``.  THE reader gate.

    ONE implementation, called by both consumer seams -- the fit driver
    (which must not fit ``v``) and the MPA Sigma pass (which must not
    integrate a pole field that was fitted to it).  Two copies of this
    refusal would be two claims about which object the pipeline consumes,
    differing on the day one of them is edited.

    ``content`` is the declaration as read back: a canonical key, or
    ``None`` for a store written before the attr existed.  ``None`` is
    REFUSED and not assumed: the first-light production store holds the
    full ``W`` and every synthetic fixture in the tree holds ``W_c``, so
    neither guess is even usually right, and the wrong one costs 130 eV
    with no other symptom.
    """
    tail = f"  Store: {source}." if source else ""
    if content is None:
        raise ValueError(
            f"{where}: this store does not declare WHICH screening object "
            f"it holds ({W_SCREENING_CONTENT_ATTR} / "
            f"{FIT_SCREENING_CONTENT_ATTR} is unset), so it cannot be "
            f"consumed.  The multipole pole field must be a fit to the "
            f"CORRELATION part W_c = W - v: v is frequency-independent, "
            f"it is already counted in Sigma_x, and a pole field carrying "
            f"it puts the bare Coulomb interaction into the tau "
            f"convolution.  On the 2026-08-09 production deck that was "
            f"Sigma_c = -130.651 eV against a Godby-Needs +0.6754 eV at "
            f"the n_p = 1 bridge, with |v| = 104-119 % of |W| at the "
            f"probe frequency.  Fix: stamp the store once with "
            f"mpa_store.declare_w_screening_content / "
            f"declare_fit_screening_content, or re-produce it with the "
            f"current writers, which require the declaration at birth."
            f"{tail}")
    if content != "W_c":
        raise ValueError(
            f"{where}: this store declares screening_content={content!r} "
            f"-- the FULL screened interaction, with the bare Coulomb v "
            f"still in it.  The multipole fit and the Sigma pass both "
            f"require the correlation part W_c = W - v (MPA_THEORY "
            f"1.1/1.2); fitting v gives poles whose residues are "
            f"dominated by it and a Sigma_c wrong by two orders of "
            f"magnitude, which is what the 2026-08-09 n_p = 1 bridge gate "
            f"measured: Sigma_c = -130.651 eV against a Godby-Needs "
            f"+0.6754 eV on the same two samples.  Fix: subtract V_q "
            f"from every frequency slab -- "
            f"the same subtraction ppm_sigma.fit_ppm performs at "
            f"'Wc0_q = W0_q - V_q' and head_dipole at 'w_c = w_head - "
            f"v_head' -- into a NEW store declared 'W_c'.  This one is "
            f"not silently corrected, because a reader that repairs its "
            f"input is a reader whose output nobody can attribute.{tail}")
    return content

#: The head-set sibling of the declaration, per labelled ``__mpahead*``
#: group.  Separate attrs because the two axes have separate producers
#: and separate lifetimes: the body is fitted by the fit driver against
#: the W store's grid, the head by the screening sweep's dipole route,
#: and a store can legitimately hold a declared body beside a legacy
#: undeclared head set.
HEAD_ENERGY_UNIT_ATTR = "mpa_head_energy_unit"

#: Sibling group holding the q -> 0 head: the ``2*n_p`` sampled values and
#: the ``n_p`` poles fitted to them.
MPA_HEAD_SUFFIX = "__mpahead"

#: LABELLED head sets, and why the axis needed more than one slot.
#:
#: The q -> 0 head is built from the dipole matrix elements, and
#: ``common.mtxel_sweep.dipole_operator`` assembles the velocity as
#: ``v = 2(k+G) psi - dV_NL/dK psi``.  Measured against BerkeleyGW's own
#: q0 head at all 265 contour-deformation frequencies on the matched
#: nband = 100 deck, the shipped relative sign puts eps00(0) 31 % high with
#: the SHAPE right (one global scale of 1.377 on eps - 1 leaves a 0.3 %
#: median residual), while flipping it agrees to 1.0e-5 at z = 0 and a
#: median 4.8e-6 on the imaginary axis.  That is what a sign looks like and
#: not what a magnitude looks like -- but changing the character moves
#: every ``dipole.h5`` in the tree, the four regression fixtures
#: ``harness.protect_fixtures`` holds read-only, the BSE absorption
#: references and the plasmon-pole head, so it is an OWNER decision and it
#: is pending.
#:
#: A store that carried only one of the two would force that decision by
#: omission, and a store that carried the wrong one silently would be
#: indistinguishable from a store that carried the right one.  So the axis
#: takes NAMED sets: the default label writes the group this format already
#: had (a v2 store with one head is unchanged, byte for byte), and any
#: other label writes a sibling beside it.  The consumer names which one it
#: is using and that name reaches the output, so a number can always be
#: attributed to a convention.
MPA_HEAD_DEFAULT_LABEL = "as_shipped"


def head_group_name(label=None):
    """The group a head set with this label lives in.

    ``None`` and the default label BOTH resolve to the bare
    ``__mpahead``, which is what makes this backward compatible: a store
    written before labels existed is a store with exactly one head set,
    and it is the default one.
    """
    if label is None or str(label) == MPA_HEAD_DEFAULT_LABEL:
        return MPA_HEAD_SUFFIX
    lab = str(label)
    if not lab or "/" in lab or lab != lab.strip():
        raise ValueError(
            f"head_group_name: {label!r} is not a usable head-set label; "
            f"it becomes an HDF5 group name beside the tensor.")
    return f"{MPA_HEAD_SUFFIX}__{lab}"


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

#: Version-2 attrs that are OURS but are not REQUIRED — the difference
#: matters at two seams and they pull in opposite directions.
#: :func:`read_w_header` refuses a file missing anything in
#: ``_MPA_OWNED_ATTRS`` (a half-stamped file), and putting the screening
#: declaration there would make every store written before it existed
#: unreadable — including by
#: :func:`declare_w_screening_content`, whose whole job is to add it.
#: The removability comparison, on the other hand, has to set it aside
#: exactly as it sets the required ones aside, or a v2 file stops
#: matching a v1 file attr for attr.  So it is listed, and listed here.
_MPA_OPTIONAL_ATTRS = (W_SCREENING_CONTENT_ATTR,)

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
    screening_content=None,
    dtype=None,
    provenance=None,
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
    screening_content
        ``'W_c'`` or ``'W'`` — WHICH object these slabs hold.  Optional
        HERE and refused at the consumer (:func:`require_correlation_part`,
        called by the fit driver), because a producer that has not yet
        decided may legitimately allocate first and
        :func:`declare_w_screening_content` afterwards; what it may not
        do is reach a fit undeclared.  See :data:`W_SCREENING_CONTENTS`
        for the 130 eV this exists to catch.
    """
    qs = _qs()
    n_omega = int(n_omega)
    n_q_on_disk = int(n_q_on_disk)
    n_mu = int(n_mu)
    if min(n_omega, n_q_on_disk, n_mu) < 1:
        raise ValueError(
            f"mpa_store: extents must be positive; got n_omega="
            f"{n_omega}, n_q_on_disk={n_q_on_disk}, n_mu={n_mu}")
    dtype = np.complex128 if dtype is None else dtype
    shape = (n_omega, n_q_on_disk, n_mu, n_mu)
    # THE CLOSURE REFUSAL RUNS BEFORE ANY BYTE IS ALLOCATED.  The stamp
    # below refuses too — it is the same call — but by then a full-size
    # dataset exists, and a refused allocation that leaves a
    # correctly-shaped file behind is precisely the shape of the
    # all-zero-screening hazard this format spends a ledger to avoid.
    if closure_verdict is None:
        raise ValueError(
            "allocate_w_omega: closure_verdict= is required.  A wedge "
            "stored against a centroid set that is not orbit-closed has "
            "no permutation α and is unrecoverable — at EVERY "
            "frequency, so the ω axis multiplies the damage rather than "
            "diluting it.  Take one from "
            "symmetry_maps.verify_centroid_orbit_closure.")
    closure_verdict.raise_if_not_closed(
        f"allocate_w_omega({name!r}) refuses q_irr storage")
    with qs.QirrDest(dest, mode) as grp:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, shape=shape, dtype=dtype)
        return stamp_w_omega(
            grp, name, tables=tables, omega=omega, sampling=sampling,
            omega_line=omega_line, closure_verdict=closure_verdict,
            screening_content=screening_content,
            data_ready=np.zeros(n_omega, dtype=bool),
            provenance=provenance)


def stamp_w_omega(
    dest,
    name,
    *,
    tables,
    omega,
    sampling,
    omega_line=None,
    closure_verdict,
    screening_content=None,
    data_ready=None,
    n_rmu_logical=None,
    provenance=None,
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
        ds.attrs["mpa_omega_units"] = "Ha"
        ds.attrs["mpa_protocol"] = san["protocol"]
        ds.attrs["mpa_varpi"] = san["varpi"]
        ds.attrs["mpa_n_p"] = np.int64(san["n_p"])
        ds.attrs["mpa_alpha"] = np.int64(san["alpha"])
        ds.attrs["mpa_omega_max"] = np.float64(san["omega_max"])
        ds.attrs["mpa_grid_hash"] = grid_hash
        ds.attrs["mpa_writer"] = "file_io.mpa_store"
        if screening_content is not None:
            ds.attrs[W_SCREENING_CONTENT_ATTR] = canonical_screening_content(
                screening_content, where=f"stamp_w_omega({name!r})")
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
        led = mgrp["data_ready"][()]
        led[i] = bool(ready)
        mgrp["data_ready"][...] = led
        ds.attrs["qirr_data_ready"] = bool(led.all())
        return int(led.sum())


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
            "screening_content": _declared_screening_content(
                ds, W_SCREENING_CONTENT_ATTR),
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
        # THE PAIR TRANSPOSE, BECAUSE THIS SLAB IS NOT HERMITIAN.  The
        # unfold's default completion of a time-reversed row is the
        # elementwise conjugate, which equals the (μ, ν) transpose it is
        # standing in for exactly when the operator is Hermitian.  A W_c
        # slab is Hermitian at ω = 0 and NOT at a double-parallel sample:
        # measured on mpa_wcprod_0809/stores/W_omega_full_wc.h5, relative
        # non-Hermiticity 5.9e-13 at z = 0 against 0.58 at z = 0.3155 +
        # 0.1i and 1.69 at z = 2.524 + 1i.  Taking the default here would
        # be wrong by O(1) on every TRS row of a wedge store, at every
        # frequency but the one where the two rules agree — which is
        # exactly the frequency the existing arms all test at.
        trs_rule="pair_transpose",
    )
    return full, header


def declare_w_screening_content(dest, name, content, *, mode="a"):
    """Stamp a LEGACY W(ω) store's screening content — once, never twice.

    The migration path for the stores written before the declaration
    existed, and the way the 2026-08-09 production store is labelled for
    what it is: it holds the full ``W``, so it is declared ``'W'`` and
    the fit driver then refuses it BY NAME instead of by omission.  The
    bytes are evidence and are not rewritten.

    A re-declaration to a DIFFERENT value refuses, for the same reason
    :func:`declare_fit_energy_unit` does: the bytes did not change, so at
    most one of the two declarations is true.  Re-declaring the SAME
    value is a no-op, so the call is safe to leave in a setup script.

    Returns the canonical content stamped.
    """
    qs = _qs()
    can = canonical_screening_content(
        content, where="declare_w_screening_content")
    with qs.QirrDest(dest, mode) as grp:
        ds, _ = _open_w(grp, name)
        declared = _declared_screening_content(ds, W_SCREENING_CONTENT_ATTR)
        if declared is not None and declared != can:
            raise ValueError(
                f"declare_w_screening_content: {name!r} already declares "
                f"{declared!r} and the caller asked for {can!r}.  The "
                f"tensor did not change, so at most one of the two is "
                f"true; refusing to replace a declaration is what keeps "
                f"'which screening object is this' a question with one "
                f"answer.  If the first declaration was WRONG, that is a "
                f"corrupted store: rebuild it into a fresh file with the "
                f"right declaration and say so in its provenance.")
        ds.attrs[W_SCREENING_CONTENT_ATTR] = can
        return can


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

def canonical_energy_unit(unit, *, where):
    """Normalise an energy-unit spelling to a :data:`FIT_ENERGY_UNITS` key.

    Case-insensitive on input, canonical on output, and an unknown
    spelling refuses rather than defaulting -- a default here is the
    factor-of-two the declaration exists to kill.
    """
    if unit is None:
        raise ValueError(
            f"{where}: energy_unit is required and there is no default.  "
            f"The pole axis's unit is the fact this attr exists to pin "
            f"(see FIT_ENERGY_UNITS: a Hartree axis read as Rydberg "
            f"halves every pole with no symptom); defaulting it would "
            f"re-open exactly that hole.  Pass 'Ry' or 'Ha' -- the fit "
            f"driver takes it from the W store's own mpa_omega_units "
            f"stamp, which is where the abscissae's unit is recorded.")
    raw = str(unit).strip()
    for known in FIT_ENERGY_UNITS:
        if raw.lower() == known.lower():
            return known
    raise ValueError(
        f"{where}: energy_unit={unit!r} is not one of "
        f"{sorted(FIT_ENERGY_UNITS)}.  Refusing rather than guessing: an "
        f"unknown unit converted by a guessed factor is the same defect "
        f"as no unit at all, wearing a declaration's clothes.")


def _declared_fit_unit(grp):
    """The store's declared pole-axis unit, or ``None`` for legacy."""
    if FIT_ENERGY_UNIT_ATTR not in grp.attrs:
        return None
    return canonical_energy_unit(
        _qs().qirr_attr_str(grp, FIT_ENERGY_UNIT_ATTR),
        where="mpa_store (stored declaration)")


def _fit_to_ry_factor(grp, where):
    """The Ha/Ry -> Ry factor for this store's pole axis, or refuse.

    THE ONE CONVERSION SEAM.  Every converting reader below calls this
    and multiplies once; nothing downstream converts again, which is
    what makes 'declared and converted once' a property of the format
    rather than a discipline every consumer must remember.

    An UNDECLARED store is refused by name, with both fixes stated,
    because there is no safe reading of it: the first-light field was
    fitted on Hartree abscissae while every synthetic test store is
    Rydberg, so neither guess is even usually right, and the wrong one
    is invisible to every internal gate (the model is invariant under
    rescaling z, Omega and B together -- the only checks that can see
    the factor are external oracles, which is how it was caught).
    """
    declared = _declared_fit_unit(grp)
    if declared is None:
        raise ValueError(
            f"{where}: this fit store does not declare its pole-axis "
            f"energy unit ({FIT_ENERGY_UNIT_ATTR} is unset), so its "
            f"Omega_p/B_p cannot be converted to the Rydberg the Sigma "
            f"stage computes in -- and they MUST be converted, not "
            f"passed through: the first-light store's axis is Hartree "
            f"(its W abscissae stamp mpa_omega_units='Ha'; the fitted "
            f"n_p=1 head pole reads 18.118 eV as Ha against BerkeleyGW's "
            f"18.009 eV, and 9.06 eV as Ry against a 16.7 eV measured "
            f"plasmon), and reading it undeclared is how every pole "
            f"entered Sigma at half its energy with no symptom.  Fix: "
            f"stamp the store once with "
            f"mpa_store.declare_fit_energy_unit(path, 'Ha' or 'Ry'), or "
            f"re-fit with the current fit driver, which inherits the "
            f"unit from the W store's own stamp.  Pass raw=True only to "
            f"inspect the undeclared bytes deliberately.")
    return float(FIT_ENERGY_UNITS[declared])


def declare_fit_energy_unit(dest, unit, *, include_heads=True, mode="a"):
    """Stamp a LEGACY store's pole-axis unit -- once, and never twice.

    The migration path for stores written before the declaration
    existed.  A store allocated by the current :func:`allocate_fit_store`
    is stamped at birth and never needs this; a legacy store gets exactly
    one declaration, because a re-declaration to a DIFFERENT value would
    change what every subsequent read returns while the bytes stayed
    put -- two claims about one axis, differing on the day one of them
    is believed.  Re-declaring the SAME value is a no-op, so the call is
    safe to leave in a setup script.

    ``include_heads`` stamps every ``__mpahead*`` group that does not
    already carry its own declaration with the same unit, because the
    head sets on the first-light store were fitted against the same
    abscissae as the body.  A head set that was fitted on a different
    axis should be stamped individually (``write_head_axis`` takes
    ``energy_unit=`` for new writes).

    Returns the canonical unit stamped.
    """
    qs = _qs()
    can = canonical_energy_unit(unit, where="declare_fit_energy_unit")
    with qs.QirrDest(dest, mode) as grp:
        _open_fit(grp)                     # a fit store, not any h5 file
        declared = _declared_fit_unit(grp)
        if declared is not None and declared != can:
            raise ValueError(
                f"declare_fit_energy_unit: this store already declares "
                f"{declared!r} and the caller asked for {can!r}.  The "
                f"bytes did not change, so at most one of the two "
                f"declarations is true; refusing to replace a "
                f"declaration is what keeps 'what unit is this axis in' "
                f"a question with one answer.  If the first declaration "
                f"was WRONG, that is a corrupted store: re-fit it, or "
                f"copy the tensors into a fresh store with the right "
                f"declaration and say so in its provenance.")
        grp.attrs[FIT_ENERGY_UNIT_ATTR] = can
        if include_heads:
            for key in list(grp):
                if str(key).startswith(MPA_HEAD_SUFFIX):
                    hd = grp[key]
                    if HEAD_ENERGY_UNIT_ATTR not in hd.attrs:
                        hd.attrs[HEAD_ENERGY_UNIT_ATTR] = can
        return can


def declare_fit_screening_content(dest, content, *, mode="a"):
    """Stamp a LEGACY fit store's screening content — once, never twice.

    Sibling of :func:`declare_fit_energy_unit`, and it exists for the
    same population: the stores fitted before the declaration did.  It is
    also how the 2026-08-09 production n_p = 1 and n_p = 8 fit stores get
    labelled ``'W'`` — the truth about them — so that the Σ pass turns
    them away by name, with the mechanism in the message, instead of
    turning them away for having no attr.

    A re-declaration to a DIFFERENT value refuses.  Returns the canonical
    content stamped.
    """
    qs = _qs()
    can = canonical_screening_content(
        content, where="declare_fit_screening_content")
    with qs.QirrDest(dest, mode) as grp:
        _open_fit(grp)                     # a fit store, not any h5 file
        declared = _declared_screening_content(
            grp, FIT_SCREENING_CONTENT_ATTR)
        if declared is not None and declared != can:
            raise ValueError(
                f"declare_fit_screening_content: this store already "
                f"declares {declared!r} and the caller asked for {can!r}.  "
                f"The poles did not change, so at most one of the two is "
                f"true; refusing to replace a declaration is what keeps "
                f"'which screening object was fitted' a question with one "
                f"answer.  If the first declaration was WRONG, that is a "
                f"corrupted store: re-fit it, or copy the tensors into a "
                f"fresh store with the right declaration and say so in "
                f"its provenance.")
        grp.attrs[FIT_SCREENING_CONTENT_ATTR] = can
        return can


def stamp_legacy_fit_id(dest, *, mode="a"):
    """Give one completed pre-identity fit allocation a permanent ID.

    New allocations receive ``mpa_fit_id`` at birth.  This explicit
    migration exists for completed stores made before that stamp: the path
    is not an allocation identity because the fitter may deliberately reuse
    it, while partial Sigma cubes can outlive the allocation they read.
    Repeating the migration is an idempotent read of the existing ID; an
    incomplete store is refused because its allocation is still mutable.
    """
    qs = _qs()
    with qs.QirrDest(dest, mode) as grp:
        _open_fit(grp)
        existing = qs.qirr_attr_str(grp, "mpa_fit_id")
        if existing:
            return existing
        if not bool(grp.attrs.get("mpa_fit_complete", False)):
            raise ValueError(
                "stamp_legacy_fit_id: the fit store is incomplete.  Its "
                "pole field is still mutable, so it cannot be assigned the "
                "permanent allocation identity carried by partial Sigma "
                "artifacts; finalize it first.")
        fit_id = "fit-" + uuid.uuid4().hex
        grp.attrs["mpa_fit_id"] = fit_id
        return fit_id


def allocate_fit_store(
    dest,
    *,
    n_q,
    n_mu,
    n_p,
    energy_unit=None,
    screening_content=None,
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
    energy_unit
        REQUIRED: the unit the Ω_p/B_p this store will hold are stated
        in ('Ry' or 'Ha') — see :data:`FIT_ENERGY_UNITS` for why there
        is no default.  The fit driver passes the W store's own
        ``mpa_omega_units`` stamp, because the poles come out in the
        unit of the abscissae the fit was solved against.
    screening_content
        REQUIRED, and for the same reason ``energy_unit`` is: WHICH
        screening object these poles were fitted to.  The fit driver
        passes the W store's own :data:`W_SCREENING_CONTENT_ATTR` stamp
        after refusing anything that is not ``'W_c'``, so a fit store the
        driver made can only ever say ``'W_c'``.  The slot accepts
        ``'W'`` too, because a store may be assembled by hand out of
        bytes whose provenance is exactly that, and the Sigma stage has
        to be able to refuse it BY NAME rather than by omission.
    grid_hash, table_hash, centroid_hash
        The W(ω) file's stamps, carried here so the Σ stage can assert
        that these poles came from that screening on that centroid set.
        Optional only because a synthetic fit has no such file.
    unfold_tables
        The W(ω) file's own :class:`QirrTables`, stamped beside the
        poles by :func:`stamp_fit_unfold_tables` so a WEDGE fit store
        can be served to the full Bloch zone later without that file.
        ``None`` for a full-BZ fit, which needs no map back.
    """
    qs = _qs()
    unit = canonical_energy_unit(energy_unit, where="allocate_fit_store")
    content = canonical_screening_content(
        screening_content, where="allocate_fit_store")
    n_q = int(n_q)
    n_mu = int(n_mu)
    n_p = int(n_p)
    if min(n_q, n_mu, n_p) < 1:
        raise ValueError(
            f"allocate_fit_store: extents must be positive; got n_q="
            f"{n_q}, n_mu={n_mu}, n_p={n_p}")
    dtype = np.complex128 if dtype is None else dtype
    with qs.QirrDest(dest, mode) as grp:
        # EVERY ``fit_*`` GOES, not just the two required ones.  A
        # re-allocation that left an earlier run's extra diagnostic
        # behind would leave a full-size array of ITS numbers indexed by
        # THIS run's ledger, and the Σ stage would certify the new poles
        # against the old evidence.
        # ...AND THE UNFOLD TABLES GO WITH THEM.  They describe the q
        # axis of the poles being deleted; left behind, they would
        # describe the NEW allocation's q axis by coincidence of extent
        # and be validated against it without complaint.
        for key in ["Omega_p", "B_p", MPA_FIT_SUFFIX,
                    FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX] + [
                k for k in grp if str(k).startswith("fit_")]:
            if key in grp:
                del grp[key]
        # Root attrs are allocation-owned just as surely as the tensors.
        # In particular, leaving ``mpa_fit_finalized_utc`` or a certificate
        # behind makes a newly empty store present itself as the completed
        # allocation formerly at this path.  Clear the three namespaces the
        # fit writer owns, then state the new allocation from scratch below.
        for key in tuple(grp.attrs):
            if str(key).startswith(("mpa_fit_", "mpa_cert_", "prov_")):
                del grp.attrs[key]
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
        grp.attrs[FIT_ENERGY_UNIT_ATTR] = unit
        grp.attrs[FIT_SCREENING_CONTENT_ATTR] = content
        grp.attrs["mpa_fit_n_p"] = np.int64(n_p)
        grp.attrs["mpa_fit_n_q"] = np.int64(n_q)
        grp.attrs["mpa_fit_n_mu_logical"] = np.int64(n_mu)
        grp.attrs["mpa_fit_complete"] = False
        grp.attrs["mpa_fit_id"] = "fit-" + uuid.uuid4().hex
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
    if version not in MPA_FIT_READABLE_VERSIONS:
        raise ValueError(
            f"mpa_store: fit store is format version {version}; this "
            f"reader accepts {list(MPA_FIT_READABLE_VERSIONS)} and writes "
            f"{MPA_FIT_FORMAT_VERSION}.  Refusing rather than guessing.")
    return grp[MPA_FIT_SUFFIX]


def _append(dset, values):
    n = int(dset.shape[0])
    arr = np.asarray(values)
    dset.resize(n + arr.shape[0], axis=0)
    dset[n:] = arr


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
        return {
            "format_version": int(grp.attrs["mpa_fit_format_version"]),
            #: The DECLARED pole-axis unit, or None for a legacy store.
            #: Informational here (the converting readers enforce it);
            #: carried so a driver can announce it without a second open.
            "energy_unit": _declared_fit_unit(grp),
            #: WHICH screening object these poles were fitted to, or None
            #: for a store written before the declaration existed.  The Σ
            #: pass runs it through :func:`require_correlation_part`; it
            #: is surfaced here so a driver can announce it in one open.
            "screening_content": _declared_screening_content(
                grp, FIT_SCREENING_CONTENT_ATTR),
            "n_p": int(grp.attrs["mpa_fit_n_p"]),
            "n_q": int(grp.attrs["mpa_fit_n_q"]),
            #: WHICH ZONE THE q AXIS IS, and how big the full one is.
            #: ``'full'`` for a store with no unfold tables — which is
            #: what every store written before they existed is, and what
            #: a full-BZ fit legitimately is — so a consumer reads one
            #: key instead of comparing ``n_q`` against a k-grid it would
            #: have to be handed.  ``stamp_fit_unfold_tables`` is the
            #: only thing that can make this ``'ibz'``, and it gets the
            #: verdict from the symmetry service's own validator.
            #: WHICH q HAVE FIT DATA — ``blocks_done`` reduced along the
            #: column axis, which is the granularity every whole-q reader
            #: works at.  Not a second ledger: the same bools, summarised
            #: at the axis the q-major readers index.  A farm fit that
            #: loses a leg loses whole q, and "a fit store missing four q
            #: looks exactly like a fit store" unless someone asks this.
            "q_done": done.all(axis=1) if done.ndim == 2 else done,
            "q_missing": (np.flatnonzero(~done.all(axis=1))
                          if done.ndim == 2 else np.flatnonzero(~done)),
            "q_storage": qs.qirr_attr_str(grp, "mpa_fit_q_storage") or "full",
            "n_q_full": int(grp.attrs.get("mpa_fit_n_q_full",
                                          grp.attrs["mpa_fit_n_q"])),
            "table_hash": qs.qirr_attr_str(grp, "mpa_fit_table_hash"),
            # The ISDF basis the fitted W used.  Counts are insufficient:
            # two centroid tables of the same size define different B_p
            # matrix elements, yet every tensor shape still agrees.
            "centroid_hash": qs.qirr_attr_str(
                grp, "mpa_fit_w_centroid_hash"),
            # The allocation, not its reusable filesystem path.  Partial
            # Sigma cubes carry this ID so a later fit allocated at the same
            # path cannot impersonate the poles they integrated.
            "fit_id": qs.qirr_attr_str(grp, "mpa_fit_id"),
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
        }


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


def _refuse_missing_q(ledger, where, *, unfolding=False):
    """Refuse a whole-q read whose q axis has holes, and NAME them.

    THE HOLE THIS CLOSES.  ``_refuse_unfinalized`` asks one question —
    "is this store stamped complete" — and ``allow_partial=True`` turns
    it off entirely, which is how the Σ pass reads a staged store.
    Neither path ever looked at ``blocks_done``.  So a store that is
    stamped complete but has whole q of zeros in it, or a staged store
    read with ``allow_partial``, hands the pass loop a q of ``B_p = 0``
    at ``Ω_p = 0``, which contributes nothing to ``W(τ) = Σ_p B_p
    e^{−iΩ_p τ}`` and is therefore indistinguishable from a screening
    channel that is genuinely dark.  A farm fit reaches that state by
    LOSING A LEG: the 2026-08-10 sixteen-way run dropped one leg to a
    pool timeout and left ``q[48, 52)`` unfitted, and nothing downstream
    could tell, because a fit store missing four q looks exactly like a
    fit store.

    Per q and not per file, because that is the granularity a whole-q
    reader works at and the granularity a farm loses things at.  Named
    and not counted, because "which q" is the only question the operator
    of a re-run actually has.

    ``unfolding`` sharpens the message: on a wedge store one missing IBZ
    parent is not one missing row, it is every full-BZ row that folds
    onto it — up to ``n_sym`` of them from a single hole.
    """
    missing = np.asarray(ledger.get("q_missing", ()), dtype=np.int64)
    if missing.size == 0:
        return
    n_q = int(ledger["n_q"])
    extra = ""
    if unfolding:
        extra = (f"  This store is a WEDGE and the read was going to "
                 f"unfold it, so each missing q is not one row of the "
                 f"answer but every full-BZ q that folds onto it — one "
                 f"hole here becomes a whole star of zeros in Σ.")
    raise ValueError(
        f"{where}: {missing.size} of {n_q} q have NO fit data — q "
        f"{_ranges(missing)}.  They read back as zeros, and a zero pole "
        f"is not an absent pole: B_p = 0 at Omega_p = 0 contributes "
        f"nothing to W(tau) = sum_p B_p exp(-i Omega_p tau) and so looks "
        f"exactly like a screening channel that is genuinely dark.  This "
        f"is the state a farm fit reaches by losing a leg.{extra}  Re-run "
        f"the missing q into this store (the ledger is per (q, column), "
        f"so a resumed walk fits exactly the blocks that are absent), or "
        f"pass raw=True to inspect the holes deliberately."
    )


def read_fit_block(src, q, mu_cols, *, allow_partial=False, raw=False,
                   mode="r"):
    """One column block's ``(Omega_p, B_p, diagnostics, ledger)``, in Ry.

    Refuses an unfinalized store unless ``allow_partial=True``, and
    when partial, refuses the specific columns that are not fitted —
    "the file is incomplete" and "the columns you asked for are
    incomplete" are different facts and a driver resuming a crashed fit
    needs the second one.

    RETURNS RYDBERG.  ``Omega_p`` and ``B_p`` are converted from the
    store's DECLARED pole-axis unit at this seam and nowhere else; an
    undeclared store refuses by name (see :func:`_fit_to_ry_factor`).
    ``raw=True`` skips both the refusal and the conversion and hands
    back the stored bytes -- for format tooling and migration only,
    never for a Σ consumer, and the flag's name is the audit trail.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        scale = 1.0 if raw else _fit_to_ry_factor(
            grp, f"read_fit_block(q={q})")
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
        if scale != 1.0:
            Om = Om * scale
            Bp = Bp * scale
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


def read_pole_slice(src, p, *, unfold=False, mesh_xy=None,
                    allow_partial=False, raw=False, mode="r"):
    """``(Omega_p, B_p)`` for ONE pole -- ``(n_q, N_mu, N_mu)`` each, in Ry.

    THE READ THE SIGMA ACCUMULATION ACTUALLY PERFORMS, and the reason the
    pole axis is leading on disk.  :func:`read_fit_block` is all ``n_p``
    poles of a few columns (the fit driver's shape) and
    :func:`read_fit_tensors` is everything at once (which is the object
    the staged design exists to avoid holding).  One pole, every q, every
    element is one contiguous slab, and it is what one pass of the
    fourteen-pass self-energy needs resident.

    THE LEDGER REFUSAL COMES WITH IT.  Slicing the dataset directly would
    step around "an unfitted column reads back as zeros, and a zero pole
    is not an absent pole", so the same refusal guards this read.  This
    function was written in ``gw.mpa.fit_driver`` as a named stopgap; it
    lives here now, beside the other two readers, and ``fit_driver``
    re-exports it so no caller moved.

    AND THE UNIT CONVERSION COMES WITH IT TOO, since 2026-08-09.  This is
    the read the Sigma pass loop performs once per pole, and it is the
    read that fed Hartree poles into a Rydberg window planner on the
    first end-to-end dispatch -- every pole at half its energy, the
    width split and the Laplace buckets mis-sized from the same numbers,
    and no internal gate able to see it.  So the returned ``Omega_p``
    and ``B_p`` are RYDBERG, converted from the store's declared unit at
    this seam and nowhere else; an undeclared store refuses by name with
    both fixes stated, and ``raw=True`` is the tooling escape that skips
    refusal and conversion together (never for a Σ consumer).

    AND THE q WEDGE IS UNFOLDED HERE, WHEN ASKED.  ``unfold=True`` serves
    the FULL Bloch zone out of a store written on the symmetry wedge, by
    :func:`unfold_pole_field` — the same ``symmetry_maps`` map the W
    unfold uses, applied twice (with the lattice wrap for ``B_p``, with
    it zeroed for ``Omega_p``).  A store already at full BZ returns
    unchanged, so the flag is safe to pass unconditionally and the
    CALLER never branches on the store's shape; that is what makes "a
    wedge store with fit data simply works" a property of this reader
    rather than of every consumer of it.

    AND EVERY q IT RETURNS IS CHECKED FOR HAVING BEEN FITTED AT ALL.
    This read hands back the whole q axis, so a q with no fit data is a
    slab of zeros inside an otherwise real answer — indistinguishable
    from a dark screening channel, and the state a farm fit reaches by
    losing a leg.  :func:`_refuse_missing_q` names the q, and it is NOT
    gated on ``allow_partial``: that flag is a statement about the file
    not being finalized, and this is a statement about the data the
    caller is about to integrate.

    Parameters
    ----------
    unfold
        Expand the stored q wedge to the full zone.  Needs ``mesh_xy``
        and needs the store to carry its unfold tables
        (:func:`stamp_fit_unfold_tables`); both are refused by name.
        On a wedge, the missing-q refusal is sharper for a reason: one
        unfitted IBZ parent becomes every full-BZ q that folds onto it.
    mesh_xy
        Device mesh for the sharded unfold.  Ignored when the store is
        already full-BZ.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        scale = 1.0 if raw else _fit_to_ry_factor(
            grp, f"read_pole_slice(p={p})")
        _refuse_unfinalized(grp, ledger, allow_partial,
                            f"read_pole_slice(p={p})")
        ip = int(p)
        if not 0 <= ip < ledger["n_p"]:
            raise IndexError(
                f"read_pole_slice: p={ip} is outside [0, {ledger['n_p']}); "
                f"the pass order is gw.mpa.fit_driver.pole_pass_order(n_p).")
        # EVERY q COMES BACK FROM THIS READ, so every q has to be real.
        # Deliberately NOT gated on allow_partial: that flag says "this
        # store is not finalized and I know it", which is a statement
        # about the FILE, and this is a statement about the DATA the
        # caller is about to integrate.  raw=True is the escape, and its
        # name is the audit trail.
        if not raw:
            _refuse_missing_q(ledger, f"read_pole_slice(p={ip})",
                              unfolding=bool(unfold)
                              and ledger["q_storage"] == "ibz")
        Om = np.asarray(grp["Omega_p"][ip])
        Bp = np.asarray(grp["B_p"][ip])
        if scale != 1.0:
            Om = Om * scale
            Bp = Bp * scale
        if not unfold or ledger["q_storage"] == "full":
            return Om, Bp

        tables = read_fit_unfold_tables(grp)
        if tables is None:
            raise ValueError(
                f"read_pole_slice: this store holds n_q={ledger['n_q']} "
                f"of a {ledger['n_q_full']}-point zone — it is on the "
                f"symmetry wedge — and carries no unfold tables, so there "
                f"is no way back to the full zone from these bytes alone. "
                f"The tables are the W(ω) file's own "
                f"{FIT_TABLE_OWNER}{qs.QIRR_TABLE_SUFFIX} group; stamp "
                f"them with mpa_store.stamp_fit_unfold_tables(store, "
                f"mpa_store.read_w_tables(w_src, w_name)) while that file "
                f"still exists.  Re-deriving them from a k-grid here "
                f"would put a second opinion about the symmetry map in "
                f"the tree, which is what the service exists to prevent.")
        if mesh_xy is None:
            raise ValueError(
                f"read_pole_slice: unfolding the pole wedge "
                f"({ledger['n_q']} of {ledger['n_q_full']} q) is a "
                f"sharded gather and needs a mesh; pass mesh_xy= or "
                f"unfold=False to take the wedge as stored.")
        Om_f, B_f = unfold_pole_field(Om, Bp, tables, mesh_xy=mesh_xy)
        # HOST NUMPY, as the wedge read already returned.  The pass loop's
        # next move is host-side window planning off Re/Im Ω, so handing
        # back a device array would only move the gather, not remove it.
        # At P>1 these are GLOBAL arrays sharded over processes, so bare
        # ``np.asarray`` cannot fetch their non-addressable shards; use the
        # one collective wrapper that distinguishes global sharding from a
        # replicated array and reconstructs the full source shape exactly.
        from common.collectives import gather_to_host
        return gather_to_host(Om_f), gather_to_host(B_f)


def stamp_fit_unfold_tables(dest, tables, *, mode="a"):
    """Put the q-wedge unfold tables BESIDE the poles, once and for good.

    A wedge fit store is ``n_q_ibz`` rows of a zone the Sigma kernel sums
    over in full, and the map back is the same ``irr_idx``/``sym_idx``/
    ``sym_perm``/``L_table``/``q_irr_frac`` group the W(omega) file it was
    fitted from already carries.  Copying it here is the rule this format
    states for its own W side, applied to the poles: *a tensor whose
    reconstruction tables live anywhere but beside it silently decays
    when anything upstream is regenerated.*  By the time Sigma runs, the
    W file may be gone; :func:`read_pole_slice` must not need it.

    Filed under :data:`FIT_TABLE_OWNER` so ``qirr_store.read_tables``
    reads it back verbatim -- the tables are validated by that service's
    own :func:`~symmetry_maps.qirr_store.validate_qirr_tables` against
    THIS store's ``n_q`` and ``n_mu``, so a table group from a different
    deck refuses on the extents rather than unfolding into plausible
    nonsense.

    Re-stamping the SAME tables is a no-op, so the call is safe to leave
    in a driver.  Re-stamping DIFFERENT tables refuses, for the reason
    every declaration in this format refuses one: the poles did not
    change, so at most one of the two table groups describes them.

    Returns the canonical :class:`~symmetry_maps.qirr_store.QirrTables`
    stamped.
    """
    qs = _qs()
    can = tables.canonical()
    with qs.QirrDest(dest, mode) as grp:
        ledger = fit_completion_ledger(grp)
        n_q, n_mu = int(ledger["n_q"]), int(ledger["n_mu"])
        # The service's own validator decides ibz-vs-full from the extents.
        q_storage = qs.validate_qirr_tables(can, n_q, n_mu)
        digest = can.digest()
        stamped = qs.qirr_attr_str(grp, "mpa_fit_table_hash")
        if stamped is not None and stamped != digest:
            raise ValueError(
                f"stamp_fit_unfold_tables: this store already carries "
                f"unfold tables with digest {stamped} and the caller "
                f"offered {digest}.  The poles did not change, so at "
                f"most one of the two groups describes the q axis they "
                f"live on; unfolding with the wrong one returns a "
                f"finite, plausible, wrongly-permuted pole field.  If "
                f"the first stamp was wrong, that is a corrupted store: "
                f"rebuild it and say so in its provenance.")
        tgrp_name = FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX
        if tgrp_name in grp:
            del grp[tgrp_name]
        tgrp = grp.create_group(tgrp_name)
        for key in ("irr_idx_q", "sym_idx_q", "q_irr_frac", "sym_perm",
                    "L_table"):
            tgrp.create_dataset(key, data=getattr(can, key))
        tgrp.attrs["n_sym_spatial"] = np.int64(can.n_sym_spatial)
        tgrp.attrs["table_hash"] = digest
        grp.attrs["mpa_fit_table_hash"] = digest
        grp.attrs["mpa_fit_q_storage"] = q_storage
        grp.attrs["mpa_fit_n_q_full"] = np.int64(can.n_q_full)
        return can


def read_fit_unfold_tables(src, *, mode="r"):
    """The stored q-wedge unfold tables, or ``None`` when there are none.

    ``None`` is a legitimate answer and not an error: a full-BZ fit store
    needs no tables and every store written before
    :func:`stamp_fit_unfold_tables` existed has none.  The refusal for
    "wedge, and no way back to the full zone" belongs to the consumer,
    which is :func:`read_pole_slice`, because only it knows the caller
    actually asked to unfold.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        if FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX not in grp:
            return None
        return qs.read_tables(grp, FIT_TABLE_OWNER, mode=mode)


def unfold_pole_field(Omega_p, B_p, tables, *, mesh_xy):
    """One pole's ``(Omega, B)`` wedge slab -> the full Bloch zone.

    THE MAP, AND WHY IT IS TWO CALLS TO ONE HELPER.  The per-element
    multipole model is ``W_c(z) = sum_p 2*Omega_p*B_p / (z**2 -
    Omega_p**2)``, and every entry of the unfold's action on ``W`` is a
    z-INDEPENDENT scalar times the value of a DIFFERENT element::

        W_full[q, mu, nu](z) = c(q, mu, nu) * W_ibz[i(q), a(mu), a(nu)](z)

    with ``c = exp(2*pi*i*q_irr.(L_mu - L_nu))`` on a spatial row, and on
    a time-reversed row ``conj(c)`` against the TRANSPOSED element
    ``[a(nu), a(mu))]`` -- the pair-transpose rule, derived and measured
    in ``symmetry_maps.unfold_isdf_operator``.  A z-independent scalar
    multiple of a multipole model is a multipole model with the SAME
    poles and scaled residues, so::

        Omega'_p = Omega_p[parent element]        (permutation only)
        B'_p     = c * B_p[parent element]        (permutation + phase)

    and NOTHING IS CONJUGATED -- not the residues and, decisively, not
    the pole positions.  ``Im Omega_p < 0`` is preserved by construction
    rather than by a guard, so the failure the wedge refusal was written
    against -- a fourth-quadrant pole conjugated into ``exp(+Gamma*tau)``,
    which grows -- cannot be reached from here.  Time reversal never
    conjugated a pole; it swapped a pair of centroid indices, and the
    conjugate was the Hermitian shorthand for that swap.

    So this is ONE map applied twice, not two maps: ``B_p`` goes through
    ``unfold_isdf_operator`` with the store's own ``L_table``, and
    ``Omega_p`` goes through the same call with the lattice wrap ZEROED,
    which is exactly what "the pole positions carry only the
    permutation" means when written as code.  Re-deriving the
    permutation here instead would put a second opinion about the
    symmetry map in the tree, which is the thing the service exists to
    prevent.

    Parameters
    ----------
    Omega_p, B_p
        ``(n_q_ibz, N_mu, N_mu)`` complex -- one pole's wedge slab, at
        the store's LOGICAL centroid extent.
    tables
        The store's :class:`~symmetry_maps.qirr_store.QirrTables`.
    mesh_xy
        Device mesh; the unfold is sharded ``P(None, 'x', 'y')``.

    Returns
    -------
    ``(Omega_full, B_full)``, each at the reader mesh's
    ``(n_q_full, N_mu_padded, N_mu_padded)`` in-memory extent.  The logical
    block is unchanged and every pad row/column is exactly zero.
    """
    import jax.numpy as jnp
    # THROUGH THE SERVICE'S DOOR, never a submodule -- the same import
    # ``read_w_slab`` makes, so both unfolds are the same code.
    from symmetry_maps import unfold_isdf_operator

    can = tables.canonical()
    Omega_arr = np.asarray(Omega_p)
    B_arr = np.asarray(B_p)
    if (Omega_arr.ndim != 3 or B_arr.shape != Omega_arr.shape
            or Omega_arr.shape[-2] != Omega_arr.shape[-1]):
        raise ValueError(
            "unfold_pole_field: Omega_p and B_p must share one square "
            f"(n_q_ibz, n_mu, n_mu) slab; got {Omega_arr.shape} and "
            f"{B_arr.shape}.")
    n_log = int(Omega_arr.shape[-1])
    if int(can.n_mu) != n_log:
        raise ValueError(
            f"unfold_pole_field: pole slabs have logical n_mu={n_log}, but "
            f"their unfold tables address {int(can.n_mu)} centroids.")

    # Disk owns the LOGICAL extent; the reader owns today's mesh pad.  This
    # is the same inverse-of-storage operation as qirr_store.read_tensor:
    # zero tensor tails, identity permutation tail, zero lattice-wrap tail.
    # The unfold's two all_to_all axes require divisibility by the whole
    # mesh, while a fit written on another device count must remain usable.
    from runtime.padding import padded_mu_extent
    n_pad = int(padded_mu_extent(n_log, mesh_xy))
    if n_pad != n_log:
        pad = n_pad - n_log
        Omega_arr = np.pad(Omega_arr, ((0, 0), (0, pad), (0, pad)))
        B_arr = np.pad(B_arr, ((0, 0), (0, pad), (0, pad)))
        can = can.padded(n_pad)
    kw = dict(irr_idx=can.irr_idx_q, sym_idx=can.sym_idx_q,
              sym_perm=can.sym_perm, q_irr_frac=can.q_irr_frac,
              mesh_xy=mesh_xy, n_sym_spatial=int(can.n_sym_spatial),
              # W_c(q, z) at a multipole sample is NOT Hermitian -- 0.58
              # to 1.69 relative on the production Si store -- so the
              # elementwise-conj completion of the time-reversed rows is
              # wrong for it by O(1).  Its residues inherit that.
              trs_rule="pair_transpose")
    L_zero = np.zeros_like(np.asarray(can.L_table))
    Omega_full = unfold_isdf_operator(
        jnp.asarray(Omega_arr), L_table=L_zero, **kw)
    B_full = unfold_isdf_operator(
        jnp.asarray(B_arr), L_table=can.L_table, **kw)
    return Omega_full, B_full


def allocate_head_axis(dest, *, n_p, label=None, mode="a"):
    """Create the q -> 0 head's ``2*n_p`` sample slots and ``n_p`` pole slots.

    Separate from :func:`allocate_fit_store` on purpose: the head is a
    different producer (the screening sweep's ``HeadResolver``, evaluated
    on the strip) reaching the same file, and the body fit does not wait
    for it.  Allocating it here means a store can be finalized for its
    poles and still be missing the head, which is a state
    :func:`read_head_poles` names rather than one it papers over.
    """
    qs = _qs()
    n_p = int(n_p)
    if n_p < 1:
        raise ValueError(
            f"allocate_head_axis: n_p={n_p} must be positive.")
    gname = head_group_name(label)
    with qs.QirrDest(dest, mode) as grp:
        if gname in grp:
            del grp[gname]
        hd = grp.create_group(gname)
        # The 2*n_p SAMPLES: the double-parallel grid is complex, so the
        # frequency axis is complex too.  ``persist_w0_and_head`` stores a
        # float omega_grid because {0, i*omega_p} has one nonzero part per
        # point; MPA's strip points have both, and rounding one away is how
        # a restart artifact comes to describe a file it does not match.
        hd.create_dataset("head_z", shape=(2 * n_p,), dtype=np.complex128)
        hd.create_dataset("head_w", shape=(2 * n_p,), dtype=np.complex128)
        # The n_p POLES fitted to them.  Sigma consumes poles, so a store
        # that carried only samples would make every pass re-fit the head.
        hd.create_dataset("head_Omega_p", shape=(n_p,), dtype=np.complex128)
        hd.create_dataset("head_B_p", shape=(n_p,), dtype=np.complex128)
        hd.attrs["mpa_head_n_p"] = np.int64(n_p)
        hd.attrs["mpa_head_ready"] = False
        hd.attrs["mpa_head_label"] = str(
            MPA_HEAD_DEFAULT_LABEL if label is None else label)
        grp.attrs["mpa_fit_format_version"] = np.int64(
            MPA_FIT_FORMAT_VERSION)
        return int(n_p)


def write_head_axis(dest, head_z, head_w, head_Omega_p, head_B_p, *,
                    vhead=None, label=None, energy_unit=None,
                    provenance=None, mode="a"):
    """Fill the head axis and mark it ready, in one call and only once.

    ``head_z`` / ``head_w`` are the ``2*n_p`` sample points and the head of
    ``W_c`` at them; ``head_Omega_p`` / ``head_B_p`` are the ``n_p`` poles
    fitted to that pair.  Both go in together because a store holding
    samples without poles, or poles without the samples they came from, is
    a store whose head nobody can certify.
    """
    qs = _qs()
    gname = head_group_name(label)
    with qs.QirrDest(dest, mode) as grp:
        if gname not in grp:
            raise ValueError(
                f"write_head_axis: no head axis labelled "
                f"{MPA_HEAD_DEFAULT_LABEL if label is None else label!r} is "
                f"allocated on this store (group {gname!r}); call "
                f"allocate_head_axis(n_p=..., label=...) first.")
        hd = grp[gname]
        n_p = int(hd.attrs["mpa_head_n_p"])
        if bool(hd.attrs.get("mpa_head_ready", False)):
            raise ValueError(
                "write_head_axis: this head axis is already stamped ready.  "
                "A second write would replace the head the poles beside it "
                "were certified against.")
        for name, arr, want in (("head_z", head_z, 2 * n_p),
                                ("head_w", head_w, 2 * n_p),
                                ("head_Omega_p", head_Omega_p, n_p),
                                ("head_B_p", head_B_p, n_p)):
            a = np.asarray(arr, dtype=np.complex128).reshape(-1)
            if a.size != want:
                raise ValueError(
                    f"write_head_axis: {name} has {a.size} entries, expected "
                    f"{want} at n_p={n_p}.  The sample axis is 2*n_p and the "
                    f"pole axis is n_p; a length that is neither means the "
                    f"head was built on a different grid from the body.")
            hd[name][...] = a
        bad = np.asarray(head_Omega_p, dtype=np.complex128).reshape(-1)
        if np.any(np.imag(bad) > 0.0):
            raise ValueError(
                "write_head_axis: a head pole has Im Omega > 0, which enters "
                "the tau stage as exp(+|Im Omega| tau) and grows.  The body "
                "fit's guards put poles in the closed fourth quadrant; the "
                "head's fit owes the same.")
        if vhead is not None:
            hd.attrs["mpa_head_vhead"] = complex(vhead)
        # The POLE-AXIS unit of this head set: what z, Omega_p and B_p
        # are stated in.  head_w and vhead are ENERGY VALUES of W and
        # stay in the producers' Ry regardless -- the axis and the
        # ordinate have different units and only the axis is being
        # declared.  Optional (None writes a legacy, undeclared set)
        # because the head has a working caller-owned fallback in the
        # deck key; the body's declaration is required because the body
        # readers have no such fallback.
        if energy_unit is not None:
            hd.attrs[HEAD_ENERGY_UNIT_ATTR] = canonical_energy_unit(
                energy_unit, where="write_head_axis")
        for key, val in (provenance or {}).items():
            hd.attrs["prov_" + str(key)] = val
        hd.attrs["mpa_head_ready"] = True
        hd.attrs["mpa_head_written_utc"] = _utc_now()
        return int(n_p)


def read_head_poles(src, *, label=None, mode="r"):
    """The q -> 0 head's poles and samples, or a refusal that names the gap.

    THE RED TWIN THIS EXISTS FOR.  A fit store written before the head
    axis existed is a complete, finalized, readable file whose poles are
    correct -- and a Sigma built from it is missing the q -> 0 head at
    every frequency, which on silicon is the term the anchor deck cares
    enough about to inject BerkeleyGW's own ``vhead`` / ``whead_0freq``
    by hand.  There is no value to fall back to and no zero that means
    "absent", so this refuses by name instead of returning anything.

    THE UNIT, IN BOTH REGIMES.  A head set written with
    ``energy_unit=`` comes back CONVERTED TO RYDBERG -- ``z``,
    ``Omega_p`` and ``B_p`` by the declared factor, ``w`` and ``vhead``
    untouched because they are energy VALUES of W in the producers' Ry
    and not points on the pole axis -- with ``energy_unit: "Ry"`` in the
    dict saying so.  A LEGACY set (no declaration) comes back exactly as
    stored with ``energy_unit: None``, and the caller owns the unit the
    way it always has (the deck's ``mpa_pole_energy_unit`` through
    ``gw.mpa.sigma_head``).  Returning rather than refusing on the
    legacy half is deliberate and is the asymmetry with the body
    readers: the head path HAS a working caller-owned conversion that
    predates the declaration, and hard-refusing every store in the field
    would retire it for no correctness gain -- ``None`` is a legible
    "the file does not say", not a guess.
    """
    qs = _qs()
    gname = head_group_name(label)
    with qs.QirrDest(src, mode) as grp:
        version = int(grp.attrs.get("mpa_fit_format_version", -1))
        if gname not in grp:
            have = sorted(k for k in grp
                          if str(k).startswith(MPA_HEAD_SUFFIX))
            raise ValueError(
                f"read_head_poles: this fit store (format version {version}) "
                f"carries no {gname!r} group (it has {have}), so the head "
                f"set asked for is absent.  It has no q -> 0 "
                f"head.  The MPA Sigma path REFUSES it by name rather than "
                f"running head-less: the head is not a correction that can "
                f"be omitted and noticed later -- Sigma_c would be missing "
                f"it at every one of its frequencies and would come back "
                f"finite, smooth and wrong.  Re-run the screening sweep's "
                f"head leg against this store (allocate_head_axis + "
                f"write_head_axis, format version {MPA_FIT_FORMAT_VERSION}), "
                f"or use compute_mode = gn_ppm, whose head is analytic.")
        hd = grp[gname]
        if not bool(hd.attrs.get("mpa_head_ready", False)):
            raise ValueError(
                "read_head_poles: the head axis is allocated but not stamped "
                "ready, so its slots read back as zeros -- and a zero head "
                "pole is not an absent one, it is a head that contributes "
                "nothing, which is exactly the reading a converged dark "
                "channel would give.  Finish the head leg or refuse the run.")
        declared = None
        if HEAD_ENERGY_UNIT_ATTR in hd.attrs:
            declared = canonical_energy_unit(
                qs.qirr_attr_str(hd, HEAD_ENERGY_UNIT_ATTR),
                where="read_head_poles (stored declaration)")
        scale = FIT_ENERGY_UNITS[declared] if declared else 1.0
        z = np.asarray(hd["head_z"][()])
        Om = np.asarray(hd["head_Omega_p"][()])
        Bp = np.asarray(hd["head_B_p"][()])
        if scale != 1.0:
            z, Om, Bp = z * scale, Om * scale, Bp * scale
        return {
            "label": qs.qirr_attr_str(hd, "mpa_head_label")
            if "mpa_head_label" in hd.attrs else MPA_HEAD_DEFAULT_LABEL,
            "group": gname,
            #: "Ry" for a declared set (arrays already converted), None
            #: for a legacy one (arrays as stored; caller owns the unit).
            "energy_unit": "Ry" if declared else None,
            "n_p": int(hd.attrs["mpa_head_n_p"]),
            "z": z,
            "w": np.asarray(hd["head_w"][()]),
            "Omega_p": Om,
            "B_p": Bp,
            "vhead": hd.attrs.get("mpa_head_vhead", None),
            "written_utc": qs.qirr_attr_str(hd, "mpa_head_written_utc"),
        }


def read_fit_tensors(src, *, allow_partial=False, raw=False, mode="r"):
    """The whole ``(Omega_p, B_p, diagnostics, ledger)``, in Ry.

    For the Σ stage, which consumes every pole of every element at a q,
    and for tests.  Same finalize refusal as :func:`read_fit_block`, and
    the same unit seam: converted from the store's declared unit, an
    undeclared store refused, ``raw=True`` the tooling escape.
    """
    qs = _qs()
    with qs.QirrDest(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        scale = 1.0 if raw else _fit_to_ry_factor(grp, "read_fit_tensors")
        _refuse_unfinalized(grp, ledger, allow_partial,
                            "read_fit_tensors")
        # Whole-q reader, same hole, same refusal — see read_pole_slice.
        if not raw:
            _refuse_missing_q(ledger, "read_fit_tensors")
        Om = np.asarray(grp["Omega_p"][()])
        Bp = np.asarray(grp["B_p"][()])
        if scale != 1.0:
            Om = Om * scale
            Bp = Bp * scale
        diag = {str(k)[len("fit_"):]: np.asarray(grp[k][()])
                for k in grp if str(k).startswith("fit_")}
        return Om, Bp, diag, ledger
