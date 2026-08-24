"""Red-twin and ownership gates for the consolidated pair-space matvec."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import harness  # noqa: F401  -- installs the test runtime policy

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _tda_operands(nspinor: int):
    psi_c = jnp.ones((1, 1, nspinor, 1), dtype=jnp.complex128)
    psi_v = jnp.ones((1, 1, nspinor, 1), dtype=jnp.complex128)
    x = jnp.asarray([[[[0.7 - 0.2j]]]])
    eps_c = jnp.asarray([[1.4]])
    eps_v = jnp.asarray([[0.3]])
    w_r = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex128)
    v_q0 = jnp.asarray([[1.8]], dtype=jnp.complex128)
    m_x = jnp.asarray([[[[0.9 + 0.4j]]]])
    m_y = jnp.asarray([[[[1.1 - 0.3j]]]])
    return (x, psi_c, psi_c, psi_v, psi_v, eps_c, eps_v,
            w_r, v_q0, m_x, m_y)


@pytest.mark.parametrize(("nspinor", "spin_factor"), [(1, 2.0), (2, 1.0)])
def test_scalar_singlet_exchange_factor_has_an_independent_red_twin(
        nspinor, spin_factor):
    """The TDA stack and simple paths implement D+sV with s=2 scalar, 1 spinor.

    The expected value is an independent one-element D/V contraction.  The
    wrong sibling convention is asserted unequal, so two shared wrong kernels
    cannot make this cell green.
    """
    from bse.bse_simple import build_bse_simple_matvec
    from bse.bse_stack_matvec import (
        build_bse_stack_matvec,
        build_bse_stack_pair_matvec,
    )

    args = _tda_operands(nspinor)
    x, _, _, _, _, eps_c, eps_v, _, v_q0, m_x, m_y = args
    d_term = (complex(eps_c[0, 0]) - complex(eps_v[0, 0])) * complex(x[0, 0, 0, 0])
    v_term = (complex(m_x[0, 0, 0, 0]) * complex(v_q0[0, 0])
              * np.conj(complex(m_y[0, 0, 0, 0]))
              * complex(x[0, 0, 0, 0]))
    expected = d_term + spin_factor * v_term
    wrong = d_term + (1.0 if spin_factor == 2.0 else 2.0) * v_term

    mesh = _mesh()
    with mesh:
        stack = build_bse_stack_matvec(mesh, 1, 1, 1, kernel="rpa")
        pair = build_bse_stack_pair_matvec(mesh, 1, 1, 1, kernel="rpa")
        simple = build_bse_simple_matvec(mesh, 1, 1, 1, include_W=False)
        got_stack = complex(np.asarray(stack(*args))[0, 0, 0, 0])
        got_pair = complex(np.asarray(
            pair(args[0], jnp.asarray(0.0), *args[1:]))[0, 0, 0, 0])
        got_simple = complex(np.asarray(simple(*args))[0, 0, 0, 0])

    np.testing.assert_allclose(got_stack, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got_pair, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got_simple, expected, rtol=2e-14, atol=2e-14)
    assert not np.isclose(got_stack, wrong, rtol=1e-10, atol=1e-10)
    assert not np.isclose(got_pair, wrong, rtol=1e-10, atol=1e-10)
    assert not np.isclose(got_simple, wrong, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("low_mem", [True, False])
def test_former_ring_entry_is_identical_to_shared_stack_entry(low_mem):
    """The compatibility entry and resolvent kernel execute one full builder."""
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from bse.bse_serial import compute_pair_amplitude
    from bse.bse_stack_matvec import build_bse_stack_matvec

    nspinor = 2
    psi_c = jnp.asarray([[[[0.8 + 0.1j], [0.2 - 0.3j]]]]).reshape(1, 1, nspinor, 1)
    psi_v = jnp.asarray([[[[0.4 - 0.2j], [0.5 + 0.1j]]]]).reshape(1, 1, nspinor, 1)
    x_full = jnp.asarray([[[[[0.6 + 0.2j]]]], [[[[0.1 - 0.4j]]]]])
    eps_c = jnp.asarray([[1.5]])
    eps_v = jnp.asarray([[0.25]])
    w_r = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex128)
    v_q0 = jnp.asarray([[0.7]], dtype=jnp.complex128)
    m = compute_pair_amplitude(psi_c, psi_v)
    args = (x_full, psi_c, psi_c, psi_v, psi_v, eps_c, eps_v,
            w_r, v_q0, m, m)

    mesh = _mesh()
    with mesh:
        shared = build_bse_stack_matvec(
            mesh, 1, 1, 1, kernel="rpa", full=True, low_mem=low_mem,
            screening=True)
        compatibility = build_bse_ring_matvec_full(
            mesh, 1, 1, 1, low_mem=low_mem, include_W=False,
            screening=True)
        got_shared = np.asarray(shared(*args))
        got_compatibility = np.asarray(compatibility(*args))

    np.testing.assert_array_equal(got_compatibility, got_shared)


def test_ring_adapter_and_resolvents_cannot_own_physics_arithmetic():
    """Structural ownership gate: adapters select options; stack owns D/V/W."""
    from bse import bse_ring_comm, bse_w_exact, w_ladder

    adapter = inspect.getsource(bse_ring_comm.build_bse_ring_matvec_full)
    assert "build_bse_stack_matvec" in adapter
    for forbidden in ("jnp.", "lax.", "einsum", "apply_V", "apply_W"):
        assert forbidden not in adapter

    rpa = inspect.getsource(bse_w_exact._build_rpa_resolvent)
    ladder = inspect.getsource(w_ladder.build_ladder_resolvent)
    assert "build_bse_stack_matvec" in rpa
    assert "build_bse_ring_matvec_full" not in rpa
    assert "build_bse_stack_matvec" in ladder
    assert "build_bse_ring_matvec_full" not in ladder
