"""Adversarial contracts for AFIXES2 (branch_review.md §2.4.1, round two).

Tiny host-only checks. The device contracts for the same items are the
``tests/multi_device/*_p4.py`` gates and the end-to-end identity legs.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_the_timed_band_still_blocks_on_what_it_watches():
    """The profiling regime is unchanged: watching is what attributes a band."""
    from common import timing
    blocked = []
    class Watched:
        def block_until_ready(self):
            blocked.append(True)
    with timing.section("afixes2.probe") as sec:
        sec.watch(Watched())
    assert blocked == [True]


def test_shared_tau_kernel_cache_never_keeps_a_callers_spatial_kernel():
    """The resident τ body is cached only when it owns its ``sigma_kij``."""
    from gw import ppm_tau_kernel
    tree = ast.parse(Path(ppm_tau_kernel.__file__).read_text())
    writes = [node for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                      and t.value.id == "_sigma_shared_tau_kernel_cache"
                      for t in node.targets)]
    assert writes, "no shared tau kernel cache write found"
    for write in writes:
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(inner is write for stmt in node.body
                          for inner in ast.walk(stmt))]
        assert any(isinstance(n, ast.Name) and n.id == "_sigma_kij"
                   for guard in guards for n in ast.walk(guard.test)), \
            "a cache write is not guarded by the caller-kernel flag"


def test_screening_writes_no_receipt_outside_an_agreed_transaction():
    """Item 11: a rank-0 write with no agreement is a hang, not an error.

    ``record`` and the final ``construction_receipt.json`` wrote under a
    bare ``process_index() == 0``. A rank-0 I/O error (quota, EIO on
    purge-eligible scratch) then raises on rank 0 while ranks 1..P-1 enter
    ``timing.fence("spole.moments")`` and hang to walltime (INVARIANTS 21).
    The sibling write in the same function already used
    ``rank0_transaction``, which broadcasts the verdict.
    """
    from gw import shared_pole_screening
    tree = ast.parse(Path(shared_pole_screening.__file__).read_text())
    screen = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "screen_shared_poles")
    guards = [node for node in ast.walk(screen) if isinstance(node, ast.If)
              and any(isinstance(inner, ast.Attribute)
                      and inner.attr == "process_index"
                      for inner in ast.walk(node.test))]
    assert not guards, \
        "screen_shared_poles gates I/O on a bare rank test"
    routed = [node for node in ast.walk(screen)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "rank0_transaction"]
    assert len(routed) >= 3, "the receipt writes are not routed"


def test_an_existing_export_from_this_model_is_retained_not_rewritten(monkeypatch):
    """Item 4: ``restart = true`` with ``write_poles`` must not die.

    The restart branch re-runs ``export_shared_pole_outputs`` and the writer
    refused "export already exists", so a restart that rebuilds nothing died
    after the head build. An export is written only once a map is complete,
    so a committed file stamped with this model's identity and digest is the
    export this run would write; it is retained with a receipt line. A file
    stamped with anything else still refuses.
    """
    from file_io import shared_pole_store as store
    identity = dict(hamiltonian="h", wavefunctions="w")
    header = dict(identity=identity, construction_receipts=[
        dict(q_span=[0, 1], receipt=dict(source_model_digest="digest-a"))])
    monkeypatch.setattr(store, "_read_header", lambda _path: header)
    assert store._export_is_current(
        Path("x.h5"), identity=identity, digest="digest-a")
    assert not store._export_is_current(
        Path("x.h5"), identity=identity, digest="digest-b")
    assert not store._export_is_current(
        Path("x.h5"), identity=dict(hamiltonian="other"), digest="digest-a")

    def uncommitted(_path):
        raise ValueError("not committed")

    monkeypatch.setattr(store, "_read_header", uncommitted)
    assert not store._export_is_current(
        Path("x.h5"), identity=identity, digest="digest-a")


def test_sc_occupation_solve_honours_the_declared_smearing_family(monkeypatch):
    """Item 8: a physics collision, not a naming one.

    ``gw_config`` REFUSES a shared-pole metallic deck that declares
    ``occ_smearing_family = mp1`` ("shared_pole needs a positive spectral
    measure"), but the SC map called ``solve_mp1_occupations``
    unconditionally and stamped ``mp1`` on the state: the shared-pole bank
    was built from exactly the occupations the deck was forced to disclaim.
    Only the one-shot owner (``gw.gw_jax``) read the declared family.
    """
    from gw import sc_iteration
    assert sc_iteration._declared_smearing_family(
        SimpleNamespace(occ_smearing_family="FD ")) == "fd"
    assert sc_iteration._declared_smearing_family(
        SimpleNamespace(occ_smearing_family=None)) is None
    tree = ast.parse(Path(sc_iteration.__file__).read_text())
    assert not [node for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and node.id == "solve_mp1_occupations"], \
        "the SC map still has an MP1-only occupation solver"
    solves = [node for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "solve_smearing_occupations"]
    assert solves and all(
        any(kw.arg == "family" for kw in call.keywords) for call in solves)
    # Owner ruling 2026-09-17: the WFN startup gate and the density rebuild
    # were the last OccupationState.solve_mp1 hard-codes on the SC map.
    assert not [node for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "solve_mp1"], \
        "the SC map still hard-codes an MP1 OccupationState solve"
    state_solves = [node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "solve_smearing"]
    assert len(state_solves) >= 2 and all(
        any(kw.arg == "family" for kw in call.keywords)
        for call in state_solves)


def test_one_shot_and_sc_read_the_same_deck_key():
    """One source of truth: both owners resolve the same configured family.

    ``gw.gw_jax`` initializes the communicator stack at import, so it is read
    from disk rather than imported: importing a driver module inside a test
    process refuses at P > 1 (``jax.distributed.initialize() must be called
    before any JAX calls``), which is a P4 failure a P1 run cannot see.
    """
    from gw import sc_iteration
    root = Path(sc_iteration.__file__).parent
    one_shot = (root / "gw_jax.py").read_text()
    assert "family=config.occ_smearing_family" in one_shot
    assert "occ_smearing_family" in Path(sc_iteration.__file__).read_text()
    config = SimpleNamespace(occ_smearing_family="mp1")
    assert sc_iteration._declared_smearing_family(config) == "mp1"


def test_a_retained_export_stays_in_the_retention_keep_set():
    """Item 4, second order: retention must not release what was retained.

    ``_retain_current_map_exports`` UNLINKS every managed export not in the
    set it is handed, so the keep-set is this map's whole target set --
    including an export that was retained rather than rewritten. Dropping a
    retained kind from that set would delete the file the restart just chose
    to keep.
    """
    from file_io import shared_pole_store
    tree = ast.parse(Path(shared_pole_store.__file__).read_text())
    export = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "export_shared_pole_outputs")
    call = next(node for node in ast.walk(export)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_retain_current_map_exports")
    assert call.args and isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "targets", (
        "retention was handed %r, not the whole target set" % call.args[0].id)
