"""The S-tensor representation ruling: one convention, and the writer obeys it.

``SMALL_ISSUES.md`` row 22 — two builders of the same physical object disagreed
in representation.  ``common.chi_from_dipole.compute_S_omega`` returns the
Cartesian q²-coefficient and has readers; ``psp.run_sternheimer`` produced a
crystal-coordinate Hessian and had none.  The Cartesian q²-coefficient is now
canonical and ``run_sternheimer`` converts before writing.

Two gates, both fixture-free, no GPU, no h5, seconds:

1. the conversion algebra is right, checked against a χ built by hand so a
   dropped ½ or a transposed frame change both show up as a red twin;
2. the driver's write site actually routes through it, so the disk file cannot
   drift back to the raw Hessian.

CENSUS-CLASS: these are convention gates, not physics gates — they should carry
the ``census`` pytest marker the moment it exists.  It does not yet exist on
``main`` as of 2026-08-09 (a parallel lane is introducing it).
"""
import ast
import pathlib

import numpy as np
import pytest

from common.chi_from_dipole import s_tensor_crystal_hessian_to_cartesian_q2


# A deliberately non-cubic, non-orthogonal reciprocal cell: a frame change that
# is not a scalar multiple of the identity, so B⁻¹ H B⁻ᵀ and its transpose (and
# B H Bᵀ, and the un-halved form) are all distinguishable.
BVEC = np.array([[1.7, 0.0, 0.0],
                 [0.4, 2.3, 0.0],
                 [0.1, -0.6, 3.1]], dtype=np.float64)


def _chi_of_q_cart(q_cart, S_cart):
    """The canonical contract: χ₀₀(q) = q_a S_ab q_b, q Cartesian."""
    return float(np.real(q_cart @ S_cart @ q_cart))


def test_conversion_reproduces_the_same_chi_the_hessian_encodes():
    """A round trip through both representations must give one χ.

    Build a χ directly from a Cartesian q²-coefficient, read off the crystal
    Hessian it implies, push that back through the converter, and require the
    Cartesian tensor to come back.  This is the whole of the ruling in one
    assertion: the two representations describe the same quadratic form.
    """
    rng = np.random.default_rng(22)
    A = rng.normal(size=(3, 3))
    S_true = 0.5 * (A + A.T)                      # symmetric Cartesian q²-coef

    # χ = q_cart S q_cart with q_cart = q_crys @ B, so as a function of q_crys
    # it is q_crys (B S Bᵀ) q_crys, i.e. a Hessian of twice that.
    H_crys = 2.0 * (BVEC @ S_true @ BVEC.T)

    S_back = s_tensor_crystal_hessian_to_cartesian_q2(H_crys, BVEC)
    assert np.allclose(S_back, S_true, atol=1e-12), (
        f"conversion did not invert the representation change:\n"
        f"{S_back}\nvs\n{S_true}")

    # and the physical statement it stands for, at a few random q
    for _ in range(5):
        q_crys = rng.normal(size=3)
        q_cart = q_crys @ BVEC
        chi_hess = 0.5 * float(q_crys @ H_crys @ q_crys)
        assert np.isclose(chi_hess, _chi_of_q_cart(q_cart, S_back), rtol=1e-12)


@pytest.mark.parametrize("wrong,label", [
    (lambda H, B: np.linalg.inv(B) @ H @ np.linalg.inv(B).T, "missing the 1/2"),
    (lambda H, B: 0.5 * H, "no frame change (crystal indices kept)"),
    (lambda H, B: 0.5 * (B @ H @ B.T), "frame change inverted (B not B⁻¹)"),
])
def test_red_twin_each_half_of_the_conversion_is_load_bearing(wrong, label):
    """Drop either the factor or the frame change and χ stops matching."""
    rng = np.random.default_rng(2222)
    A = rng.normal(size=(3, 3))
    S_true = 0.5 * (A + A.T)
    H_crys = 2.0 * (BVEC @ S_true @ BVEC.T)
    S_wrong = wrong(H_crys, BVEC)
    assert not np.allclose(S_wrong, S_true, atol=1e-8), (
        f"the {label} variant was NOT distinguishable from the correct "
        f"conversion — this gate cannot see the defect it exists for")


def _writer_block(src_path):
    """The ``with_s_tensor`` branch of ``run_sternheimer``, as an AST."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_sternheimer":
            return node
    raise AssertionError("run_sternheimer() not found")


def test_writer_converts_before_it_writes_s_tensor_q0():
    """The disk dataset must be canonical, and provably so from the source.

    Fixture-free: parsing beats running here, because running needs a WFN, a
    pseudo directory and 3 Sternheimer solves per k.  What the gate has to
    stop is a future edit that writes ``s_tensor_q0`` straight from the
    kernel's crystal Hessian again.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "psp" / "run_sternheimer.py"
    fn = _writer_block(src)

    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    called = names | attrs
    assert "s_tensor_crystal_hessian_to_cartesian_q2" in called, (
        "run_sternheimer writes s_tensor_q0 without routing through the "
        "one conversion helper — the file would carry a crystal-coordinate "
        "Hessian under a name every reader treats as a Cartesian "
        "q²-coefficient (SMALL_ISSUES.md row 22)")

    # the dataset value must be the converted tensor, not the raw sum
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_dataset"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "s_tensor_q0"):
            kw = {k.arg: k.value for k in node.keywords}
            assert "data" in kw and isinstance(kw["data"], ast.Name), (
                "s_tensor_q0 is written from an expression this gate cannot "
                "follow; keep it a plain name bound to the converted tensor")
            assert kw["data"].id == "S_total", (
                f"s_tensor_q0 written from {ast.dump(kw['data'])}, expected "
                f"the converted S_total")
            break
    else:
        raise AssertionError("no create_dataset('s_tensor_q0', ...) found")


def test_the_file_declares_its_convention():
    """A reader must be able to tell from the file, not from the git log."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "psp" / "run_sternheimer.py"
    text = src.read_text(encoding="utf-8")
    assert "s_tensor_convention" in text, (
        "sternheimer.h5 must stamp s_tensor_convention so a file written "
        "before the ruling is distinguishable from one written after")
    assert "cartesian_q2_coefficient" in text
