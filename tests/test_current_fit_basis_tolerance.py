"""``select_current_basis`` decides "monomial" at the accuracy of ``R_cart`` itself.

CPU, pytest, numpy only.  The Cartesian action is ``R_cart``: ``Aᵀ S A⁻ᵀ`` from
lattice vectors that carry the input file's digits, rounded to
``R_CART_DECIMALS`` decimals.  A fixed 1e-10 test refused bulk CrCl3 (R-3,
ferromagnetic along c, cell typed with nine digits): its three-fold
operations are orthogonal to 2.0e-10 and their off-pattern circular entries
come out at 1.0000001e-10.  The lattice vectors below are that run's
(QE XML, bohr); the operations are its six crystal-basis matrices.
"""

import types

import numpy as np
import pytest

from symmetry_maps import maps as M

CRCL3_AVEC = np.array([[5.614376316263163, -3.241461677975437, 10.91820764000940],
                       [0.000000000000000, 6.482923355950874, 10.91820764000940],
                       [-5.614376316263163, -3.241461677975437, 10.91820764000940]])
_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
_E = np.eye(3, dtype=int)
R3BAR = np.array([_E, _P, _P @ _P, -_E, -_P, -(_P @ _P)])


def _sym(avec, ops, rows=None):
    """The members ``select_current_basis`` reads, with SymMaps' own conversions."""
    n = len(ops)
    wfn = types.SimpleNamespace(avec=np.asarray(avec), sym_matrices=np.asarray(ops), ntran=n)
    sym = types.SimpleNamespace(
        sym_matrices=np.asarray(ops), sym_mats_k=np.concatenate([ops, -np.asarray(ops)]),
        translations=np.zeros((n, 3)), R_cart=M.SymMaps.syms_crystal_to_cartesian(wfn),
        sym_idx_k=np.arange(n) if rows is None else np.asarray(rows),
        sym_idx_q=np.array([0]))
    sym.operation_rows = types.MethodType(M.SymMaps.operation_rows, sym)
    sym.cartesian_action = types.MethodType(M.SymMaps.cartesian_action, sym)
    return sym


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_bulk_crcl3_takes_the_circular_basis():
    sym = _sym(CRCL3_AVEC, R3BAR)
    rows = np.arange(6)
    cart = M.current_component_action(sym, rows, None)
    circ = M.current_component_action(sym, rows, M.CIRCULAR_CURRENT_ROWS)
    # The defect this test exists for: not orthogonal at 1e-10, off-pattern entries above it.
    assert 1.5e-10 < M.cartesian_action_noise(cart) < 3.0e-10
    assert 1.0e-10 < M._largest_off_pattern(circ) < 2.0e-10
    assert not M._monomial(circ, 1e-10)
    name, basis = M.select_current_basis(sym)
    assert name == "circular" and basis is M.CIRCULAR_CURRENT_ROWS


@pytest.mark.parametrize("digits", [9, 8, 7, 6])
def test_cells_typed_with_fewer_digits_are_admitted(digits):
    assert M.select_current_basis(_sym(np.round(CRCL3_AVEC, digits), R3BAR))[0] == "circular"


def test_antiunitary_rows_of_a_magnetic_group_are_admitted():
    """Rows >= n carry time reversal: -R, and the conjugated j+ <-> j- pattern."""
    sym = _sym(CRCL3_AVEC, R3BAR, rows=np.arange(12))
    assert M.select_current_basis(sym)[0] == "circular"


def test_exact_cells_keep_their_basis():
    hexagonal = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(0.75), 0.0], [0.0, 0.0, 1.7]]) * 6.5
    c6 = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
    ops = np.array([np.linalg.matrix_power(c6, p) for p in range(6)])
    assert M.select_current_basis(_sym(hexagonal, ops))[0] == "circular"
    fcc = 5.1 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert M.select_current_basis(_sym(fcc, np.array([_E, _P, _P @ _P])))[0] == "cartesian"


@pytest.mark.parametrize("frame", [
    np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),   # three-fold axis along x
    _rot_x(1.0e-4), _rot_x(1.0e-6), _rot_x(1.0e-8),                    # axis tilted off z
])
def test_a_group_that_is_not_monomial_still_refuses(frame):
    with pytest.raises(ValueError, match="GATE current_fit_basis: got a group") as info:
        M.select_current_basis(_sym(CRCL3_AVEC @ frame.T, R3BAR))
    assert "largest off-pattern entry" in str(info.value)       # the numbers are in the refusal


def test_operations_that_are_not_isometries_refuse_first():
    """A lattice the operations do not belong to: no basis test is run on it."""
    strained = CRCL3_AVEC * np.array([1.0, 1.0 + 1.0e-4, 1.0])
    with pytest.raises(ValueError, match="GATE current_fit_basis_isometry"):
        M.select_current_basis(_sym(strained, R3BAR))


def test_the_tolerance_is_tied_to_the_rounding_of_r_cart():
    assert M.R_CART_DECIMALS == 10
    exact = np.eye(3)[None]
    assert M.cartesian_action_noise(exact) == 10.0 ** -M.R_CART_DECIMALS
    wfn = types.SimpleNamespace(avec=CRCL3_AVEC, sym_matrices=R3BAR, ntran=6)
    R = M.SymMaps.syms_crystal_to_cartesian(wfn)
    assert np.array_equal(R, np.around(R, decimals=M.R_CART_DECIMALS))
