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
