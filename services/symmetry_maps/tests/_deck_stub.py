"""``DeckHeader`` — the eleven attributes ``SymMaps`` reads, and nothing else.

WHY A STUB AND NOT ``WfnLoader``.  ``SymMaps.__init__`` consumes exactly
eleven attributes of the object it is handed.  Grepped out of
``symmetry_maps/maps.py`` (``grep -o 'wfn\\.[a-zA-Z_]*'``, whole module,
2026-08-07) they are, with their use counts::

    kpoints 12   kgrid 10   ntran 8   nkpts 6   shift 3   trs_holds 2
    translations 2   sym_matrices 2   avec 1   atom_types 1   atom_crys 1

Ten come straight out of the WFN's ``mf_header``; ``atom_crys`` is the one
DERIVATION, ``inv(avec).T @ apos``, copied from ``file_io.wfn_loader``
(:219-224, and again in its own ``_sym_wfn_stub`` at :549-551) so the stub
and the production loader answer the same question the same way — the
parity arm in ``test_symmetry_maps_deck_tables.py`` is what proves they
still do.

Reading the header with h5py directly, rather than through ``WfnLoader``,
is what makes the L-a+ tier a SERVICE test: it needs no lorrax on the path,
opens 14-52 MiB files but touches only the ``mf_header`` group (kilobytes),
and takes milliseconds per deck.  It also means the acceptance gate — the
four decks' ``(irr_idx_k, sym_idx_k)`` bit-compared against
``data/star_tables_e9340d1.json`` — is a property of ``symmetry_maps``
alone and not of the loader that usually feeds it.

``trs_holds=True`` IS PINNED AND REQUIRED, NOT DEFAULTED.  ``SymMaps`` refuses
an object carrying no verdict because omission cannot assert a symmetry. The
automatic occupied-density check is 2c-only; the loader records the applicable
layout verdict explicitly. If a 2c deck's verdict flips, the parity arm against
``SymMaps(WfnLoader(...))`` is what says so, and it can only say so because
the two sides disagree about a value that was written down.

FIXTURES ARE READ-ONLY.  ``tests/regression/*`` is chmod'd read-only on
purpose and ``harness.protect_fixtures()`` re-applies it every session
(``cohsex_debug/README.md:44-52``).  Everything here opens ``'r'``.  A test
that needs a CORRUPTED header builds one with :meth:`DeckHeader.replacing`,
in memory, from an already-read stub — never by writing to the tree, and
never by symlinking a stage (``README.md:29-42``: a symlinked stage
destroyed ``sigma_mnk.h5`` on 2026-07-25 with no error and no test
failure).
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TESTS)))

#: The in-tree decks, and the WFN each one keeps its header in.  ``cohsex``
#: ships ``WFNsmall.h5``; the other three ship ``WFN.h5``.
DECK_WFN = {
    "si_cohsex_debug": "WFN.h5",
    "cohsex_debug": "WFNsmall.h5",
    "gnppm_debug": "WFN.h5",
    "bispinor_debug": "WFN.h5",
}

#: Deck order for the acceptance gate's parametrization.  Sorted so a
#: failure report reads the same on every machine.
DECKS = tuple(sorted(DECK_WFN))


def regression_dir() -> str:
    return os.path.join(_REPO, "tests", "regression")


def deck_path(deck: str, name: str | None = None) -> str:
    """Absolute path to ``deck``'s WFN (or to another file in its dir)."""
    return os.path.join(regression_dir(), deck, name or DECK_WFN.get(deck, "WFN.h5"))


def deck_available(deck: str) -> bool:
    return os.path.isfile(deck_path(deck))


@dataclasses.dataclass(frozen=True)
class DeckHeader:
    """Exactly what ``SymMaps(wfn)`` reads.  Frozen: a table builder that
    mutated its input would be a different bug than the one being tested.

    ``bvec``/``bdot``/``blat`` are NOT in ``SymMaps``' eleven — they are
    here for the metric-convention contract test (survey §5.3), which is a
    claim about the deck's own reciprocal basis and needs no lorrax code at
    all.  Keeping them on the same object keeps the header read to one
    ``h5py.File`` open per deck.
    """

    kpoints: np.ndarray
    kgrid: np.ndarray
    shift: np.ndarray
    nkpts: int
    ntran: int
    sym_matrices: np.ndarray
    translations: np.ndarray
    avec: np.ndarray
    atom_types: np.ndarray
    atom_crys: np.ndarray
    trs_holds: bool
    # Not read by SymMaps; see the class docstring.
    bvec: np.ndarray
    bdot: np.ndarray
    blat: float
    deck: str = ""

    def replacing(self, **kw) -> "DeckHeader":
        """A COPY with fields replaced — the only way a test corrupts a deck.

        The fixture on disk is never touched; ``dataclasses.replace``
        builds a new frozen instance and the arrays it does not replace are
        shared read-only views of the ones h5py already handed over.
        """
        return dataclasses.replace(self, **kw)


#: The eleven names, as a tuple, so a test can assert the stub still covers
#: what the module reads instead of trusting this docstring.
SYMMAPS_ATTRS = ("kpoints", "kgrid", "ntran", "nkpts", "shift", "trs_holds",
                 "translations", "sym_matrices", "avec", "atom_types",
                 "atom_crys")


def read_deck(deck: str, *, trs_holds: bool = True) -> DeckHeader:
    """Read ``deck``'s ``mf_header`` into a :class:`DeckHeader`.

    ``trs_holds`` defaults to the MEASURED value for all four in-tree decks
    (see the module docstring); it is a parameter so the TRS-disallowed
    refusal (``SymMaps``' ``trs_allowed=False`` branch) can be exercised
    without inventing a fake deck.
    """
    import h5py

    with h5py.File(deck_path(deck), "r") as f:
        g = f["mf_header"]
        avec = g["crystal/avec"][:]
        apos = g["crystal/apos"][:]
        return DeckHeader(
            kpoints=g["kpoints/rk"][:],
            kgrid=g["kpoints/kgrid"][:],
            shift=g["kpoints/shift"][:],
            nkpts=int(g["kpoints/nrk"][()]),
            ntran=int(g["symmetry/ntran"][()]),
            sym_matrices=g["symmetry/mtrx"][:],
            translations=g["symmetry/tnp"][:],
            avec=avec,
            atom_types=g["crystal/atyp"][:],
            # The ONE derivation, verbatim from file_io.wfn_loader:223-224.
            atom_crys=np.einsum("ij,kj->ki", np.linalg.inv(avec).T, apos),
            trs_holds=trs_holds,
            bvec=g["crystal/bvec"][:],
            bdot=g["crystal/bdot"][:],
            blat=float(g["crystal/blat"][()]),
            deck=deck,
        )
