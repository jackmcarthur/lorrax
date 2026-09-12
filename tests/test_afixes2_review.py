"""Adversarial contracts for AFIXES2 (branch_review.md §2.4.1, round two).

Tiny host-only checks. The device contracts for the same items are the
``tests/multi_device/*_p4.py`` gates and the end-to-end identity legs.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


#: What opens a timing band: the collector directly, or the tau sweep's own
#: helper, which fences and then opens the same-named section when profiling.
_BAND_OPENERS = ("section", "tau_band")


def _section_names(module_path):
    """Every literal name a module opens a timing band with.

    Module-level string constants are resolved, because that is exactly how
    a name is supposed to be spelled once and used twice.
    """
    tree = ast.parse(Path(module_path).read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args and (
                (isinstance(node.func, ast.Attribute)
                 and node.func.attr in _BAND_OPENERS)
                or (isinstance(node.func, ast.Name)
                    and node.func.id in _BAND_OPENERS))):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            names.add(argument.value)
        elif isinstance(argument, ast.Name) and argument.id in constants:
            names.add(constants[argument.id])
    return names


def test_tau_profile_phases_are_the_names_the_sections_are_opened_with():
    """Item 7: one source of truth for the tau profiler's band names.

    ``_TAU_PROFILE_PHASES`` named ``sigma.tau.kernel`` and its two siblings
    while the sweep opened ``tau.kernel``: three rows silently dropped from
    the profile. The aggregator selects by exact name, so a phase this
    module never opens is a row that can never appear.
    """
    import gw.mpa.sigma as sigma
    root = Path(sigma.__file__).resolve().parents[2]
    opened = (_section_names(root / "gw" / "mpa" / "sigma.py")
              | _section_names(root / "gw" / "ppm_tau_kernel.py"))
    missing = [name for name in sigma._TAU_PROFILE_PHASES if name not in opened]
    assert not missing, missing


def test_tau_profile_phases_cover_the_kernel_owner_and_the_sweep():
    """The kernel's own tuple is imported, not restated."""
    import gw.mpa.sigma as sigma
    from gw.ppm_tau_kernel import TAU_KERNEL_PROFILE_PHASES
    assert sigma._TAU_PROFILE_PHASES[:len(TAU_KERNEL_PROFILE_PHASES)] == \
        TAU_KERNEL_PROFILE_PHASES
    assert sigma._TAU_PROFILE_PHASES[len(TAU_KERNEL_PROFILE_PHASES):] == (
        sigma._TAU_SWEEP_KERNEL_PHASE,
        sigma._TAU_SWEEP_ACCUMULATOR_PHASE,
        sigma._TAU_SWEEP_PROGRESS_PHASE)


def _sigma_tree():
    import gw.mpa.sigma as sigma
    return ast.parse(Path(sigma.__file__).read_text())


def _tau_node_loop(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "t"
                and isinstance(node.iter, ast.Name) and node.iter.id == "t_nodes"):
            return node
    raise AssertionError("the per-tau-node loop was not found in gw.mpa.sigma")


def test_untimed_band_accepts_a_result_without_blocking_on_it():
    """Item 3: the unprofiled band must not synchronize the host.

    ``TimingSection.__exit__`` runs every watcher, so a per-tau-node section
    with a watched result is a host sync per node -- on the incumbent
    elementwise-MPA route too, which never asked for the measurement.
    """
    from gw.mpa.sigma import _UNTIMED_BAND

    class Tripwire:
        def block_until_ready(self):
            raise AssertionError("the unprofiled tau band blocked on its result")

    with _UNTIMED_BAND as band:
        band.watch(Tripwire())


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


def test_tau_node_loop_times_nothing_outside_the_profile_switch():
    """Every per-node band goes through ``tau_band``, which reads the switch.

    The merge resolution dropped the ``if tau_profile:`` gate and left a
    ``timing.section(...) as sec: sec.watch(...)`` on the hot loop. Opening a
    section directly here is the defect, whatever it is named.
    """
    tree = _sigma_tree()
    loop = _tau_node_loop(tree)
    opened = [node for node in ast.walk(loop)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "section"]
    assert not opened, "the tau node loop opens a timing.section directly"
    managers = [item.context_expr for stmt in loop.body
                if isinstance(stmt, ast.With) for item in stmt.items]
    assert managers and all(
        isinstance(m, ast.Call) and isinstance(m.func, ast.Name)
        and m.func.id == "tau_band" for m in managers)
    band = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "tau_band")
    assert any(isinstance(n, ast.Name) and n.id == "tau_profile"
               for n in ast.walk(band)), "tau_band ignores the profile switch"


def test_inherited_sigma_peak_is_gated_and_never_caches_its_control():
    """Item 17: the double compile is gated, and its control is uncached.

    ``_shared_pole_inherited_peak`` lowered and compiled the full spatial tau
    kernel twice -- once for the incumbent route, once for the shared-pole
    injection -- on every shared-pole Sigma stage, with no gate, dial or env
    var, only to produce a ``record_sigma_peak`` receipt.  The
    ``w_synthesis=None`` leg also inserted the INCUMBENT executable into the
    process-wide ``_sigma_shared_tau_kernel_cache`` on a run that never
    dispatches that route.
    """
    import inspect
    from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
    tree = _sigma_tree()
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_shared_pole_inherited_peak")
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(inner is call for stmt in node.body
                      for inner in ast.walk(stmt))]
    assert guards, "the inherited-peak compile has no enclosing guard"
    assert any(isinstance(n, ast.Name) and n.id == "tau_profile"
               for guard in guards for n in ast.walk(guard.test)), \
        "the inherited-peak compile is not gated on the profile switch"
    peak = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "_shared_pole_inherited_peak")
    uncached = [kw for node in ast.walk(peak) if isinstance(node, ast.Call)
                for kw in node.keywords
                if kw.arg == "cache" and getattr(kw.value, "value", None) is False]
    assert uncached, "the control leg must ask for cache=False"
    assert "cache" in inspect.signature(get_shared_sigma_tau_kernel).parameters


def test_shared_tau_kernel_cache_writes_all_honour_the_cache_flag():
    """Neither the fused nor the staged return may write a cache entry."""
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
        assert any(isinstance(n, ast.Name) and n.id == "cache"
                   for guard in guards for n in ast.walk(guard.test)), \
            "a cache write is not guarded by the cache flag"


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


def test_one_shot_and_sc_read_the_same_deck_key():
    """One source of truth: both owners resolve the same configured family."""
    from gw import gw_jax, sc_iteration
    for module in (gw_jax, sc_iteration):
        source = Path(module.__file__).read_text()
        assert "occ_smearing_family" in source
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
