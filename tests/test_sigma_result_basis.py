"""Gate: every ``SigmaResult`` field declares its band basis, and the SC
finalize rotates exactly the fields declared rotatable.

The defect this exists to prevent.  ``sc_iteration.run_sc_driver``'s
finalize rotates the Σ matrices from the QP basis back to the DFT basis
before handing the object to the post-Σ seam.  The rotation used to be a
hand-written list of field names inside that ``dataclasses.replace``
call.  A Σ channel added to the dataclass and forgotten there is
returned in the QP basis with the RIGHT shape, dtype and sharding: no
shape gate, no finiteness gate and no SC invariance gate can see it
(``test_invariance_gates.py::test_sc_iteration1_equals_one_shot`` runs
at ``max_iter=1``, i.e. U = identity, where the two bases agree).

So the rotation set is declared once beside the dataclass
(``gw.sigma_dispatch.ROTATED_TO_DFT_FIELDS``) together with the fields
that stay in the Σ compute basis, and these tests pin the three
couplings that make the declaration load-bearing:

1. the four tuples partition ``dataclasses.fields(SigmaResult)`` — a new
   field is unclassified until its author says which basis it is in;
2. the finalize rotates exactly ``ROTATED_TO_DFT_FIELDS``;
3. nothing rotates a ``SIGMA_BASIS_FIELDS`` entry.

Source-level for (2) and (3): the finalize needs a mesh, a WFN and a
full SC loop to execute, so its shape is what is cheap to pin.  Same
technique as ``test_bispinor_zeta_reuse_ast.py``.
"""
from __future__ import annotations

import ast
import dataclasses
import os

from gw.sigma_dispatch import (
    BASIS_FREE_FIELDS,
    DFT_BASIS_FIELDS,
    ROTATED_TO_DFT_FIELDS,
    SIGMA_BASIS_FIELDS,
    SigmaResult,
)

SC_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "gw",
                      "sc_iteration.py")

#: The finalize's rotation primitive, and the other two band rotations in
#: the Σ path — no Σ_c(ω) may pass through any of them.
_ROTATORS = ("_rotate_to_dft_basis", "rotate_band_matrix", "_rotate_v_h_to_qp")


def _sc_module():
    with open(SC_SRC, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=SC_SRC)


def _function(mod, name):
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SC_SRC}")


def _mentions(node, name) -> bool:
    """True when ``name`` is called or referenced anywhere under ``node``."""
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and x.id == name:
            return True
        if isinstance(x, ast.Attribute) and x.attr == name:
            return True
    return False


def _rotated_locals(fn) -> set:
    """Local names bound to the result of a rotation, e.g. ``sig_h``."""
    out = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assign):
            continue
        if not any(_mentions(n.value, r) for r in _ROTATORS):
            continue
        for t in n.targets:
            if isinstance(t, ast.Name):
                out.add(t.id)
    return out


def test_every_sigma_result_field_declares_its_basis():
    declared = (tuple(ROTATED_TO_DFT_FIELDS) + tuple(SIGMA_BASIS_FIELDS)
                + tuple(DFT_BASIS_FIELDS) + tuple(BASIS_FREE_FIELDS))
    dupes = sorted({n for n in declared if declared.count(n) > 1})
    assert not dupes, (
        f"field(s) {dupes} appear in more than one basis tuple in "
        f"gw.sigma_dispatch; a field is in exactly one basis")

    actual = [f.name for f in dataclasses.fields(SigmaResult)]
    missing = [n for n in actual if n not in declared]
    assert not missing, (
        f"SigmaResult field(s) {missing} are in no basis tuple in "
        f"gw.sigma_dispatch.  Add each to exactly one of "
        f"ROTATED_TO_DFT_FIELDS (band-basis matrix, rotated to DFT by "
        f"sc_iteration.run_sc_driver's finalize), SIGMA_BASIS_FIELDS "
        f"(band-indexed, deliberately left in the Σ compute basis), "
        f"DFT_BASIS_FIELDS or BASIS_FREE_FIELDS.  A band-basis matrix "
        f"left out of the rotation set is returned in the QP basis with "
        f"the right shape and dtype and no other symptom.")

    stale = [n for n in declared if n not in actual]
    assert not stale, (
        f"basis tuple(s) in gw.sigma_dispatch name {stale}, which is not "
        f"a SigmaResult field any more")


def test_sc_finalize_rotates_exactly_the_declared_set():
    fn = _function(_sc_module(), "run_sc_driver")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "replace"
             and n.args
             and isinstance(n.args[0], ast.Name)
             and n.args[0].id == "sigma_result"]
    assert len(calls) == 1, (
        f"expected exactly one dataclasses.replace(sigma_result, ...) in "
        f"run_sc_driver's finalize; found {len(calls)}")

    local = _rotated_locals(fn)
    rotated = {kw.arg for kw in calls[0].keywords
               if any(_mentions(kw.value, r) for r in _ROTATORS)
               or (isinstance(kw.value, ast.Name) and kw.value.id in local)}
    assert rotated == set(ROTATED_TO_DFT_FIELDS), (
        f"run_sc_driver's finalize rotates {sorted(rotated)} but "
        f"gw.sigma_dispatch.ROTATED_TO_DFT_FIELDS declares "
        f"{sorted(ROTATED_TO_DFT_FIELDS)}.  Every declared field must be "
        f"rotated in that one replace() call: the ones it misses are "
        f"returned in the QP basis and read as DFT basis by the post-Σ "
        f"seam.  (If the finalize is refactored to derive its kwargs "
        f"from the tuple, teach this gate that spelling — do not delete "
        f"it.)")


def test_sigma_basis_fields_are_never_rotated():
    mod = _sc_module()
    for n in ast.walk(mod):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name not in _ROTATORS or not n.args:
            continue
        arg = n.args[0]
        attr = getattr(arg, "attr", None)
        assert attr not in SIGMA_BASIS_FIELDS, (
            f"{SC_SRC} line {n.lineno}: {name}() is applied to "
            f"{attr}, which gw.sigma_dispatch declares as staying in the "
            f"Σ compute basis.  Σ_c(ω) in particular is the operand of "
            f"the QSGW ansatz ½[Σ_ij(E_i) + Σ_ij(E_j)]ʰ, which is only "
            f"itself in the basis whose energies E_i, E_j it uses.")
