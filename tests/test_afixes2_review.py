"""Adversarial contracts for AFIXES2 (branch_review.md §2.4.1, round two).

Tiny host-only checks. The device contracts for the same items are the
``tests/multi_device/*_p4.py`` gates and the end-to-end identity legs.
"""
import ast
from pathlib import Path

import numpy as np
import pytest


def _section_names(module_path):
    """Every literal name a module opens a ``timing.section`` with.

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
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "section" and node.args):
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
