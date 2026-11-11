"""The bispinor carrier: two-component charge, raw-lift (Pauli) current.

``test_carrier_blocks_and_norms`` (CPU, pytest) checks the carrier blocks
against an independent NumPy oracle: the resolver's choice per vertex, the
large-block carrier's exact two-component norm and zero lower block, the raw
lift's small component and its norm ``1 + (alpha/2)^2 |k+G|^2`` per plane wave,
the q = 0 current identity ``<Psi_m| c alpha |Psi_n> = <psi_m| p |psi_n>`` that
makes the two-spinor charge head and the current vertex one operator, and the
refusals that keep a load from picking a carrier silently.

``main()`` (run directly under the MPI launcher, P4) is the end-to-end check:
the bispinor direct field's charge part equals the two-spinor V_H on the hsuite
fixture to round-off (docs/theory/bispinor-gw.md#carrier).
"""

import pytest

HALFALPHA = 0.00364867628215


def _oracle_raw_lift(psi_L, cart):
    """[psi_L; (alpha/2) sigma.K psi_L] with explicit Pauli matrices."""
    import numpy as np
    x, y, z = (cart[..., i][:, None, :] for i in range(3))
    up, down = psi_L[:, :, 0], psi_L[:, :, 1]
    small = HALFALPHA * np.stack(
        (z * up + (x - 1j * y) * down, (x + 1j * y) * up - z * down), axis=2)
    return np.concatenate((psi_L, small), axis=2)


def test_carrier_blocks_and_norms():
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from common.bispinor_init import (
        ALPHA_FS, LARGE_BLOCK_LIFT, RAW_KINETIC_BALANCE_LIFT,
        apply_dirac_velocity_to_ket, lift_to_4spinor)
    from common.four_current_model import (
        PAULI_FOUR_CURRENT_REPRESENTATION, SOURCE_WFN_CHARGE_REPRESENTATION,
        resolve_four_current_representation)
    from wfn_loader.loader import _check_bispinor_lift

    # The resolver: one carrier for every bispinor_gw model.
    for model in ("bare_transverse", "full_static_cohsex", "full_tangential"):
        rep = resolve_four_current_representation(True, model)
        assert (rep.charge_bispinor, rep.charge_lift) == (True, LARGE_BLOCK_LIFT)
        assert (rep.current_bispinor, rep.current_lift) == (
            True, RAW_KINETIC_BALANCE_LIFT)
        assert rep.scalar_head_bispinor is False
        assert rep.charge_representation == PAULI_FOUR_CURRENT_REPRESENTATION
    rep = resolve_four_current_representation(False, None)
    assert (rep.charge_lift, rep.current_lift, rep.scalar_head_bispinor) == (
        None, None, False)
    assert rep.charge_representation == SOURCE_WFN_CHARGE_REPRESENTATION

    rng = np.random.default_rng(20261111)
    nk, nb, ng = 2, 3, 40
    psi_L = rng.normal(size=(nk, nb, 2, ng)) + 1j * rng.normal(size=(nk, nb, 2, ng))
    gvecs = rng.integers(-6, 7, size=(nk, ng, 3)).astype(np.float64)
    kvecs = rng.uniform(-0.5, 0.5, size=(nk, 3))
    bvec = np.array([[1.1, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, -0.3, 1.3]])
    cart = (gvecs + kvecs[:, None, :]) @ bvec            # (nk, ng, 3) bohr^-1
    args = (jnp.asarray(psi_L), jnp.asarray(gvecs), jnp.asarray(kvecs),
            jnp.asarray(bvec))

    # Charge carrier: [psi_L; 0], so Psi^dagger Psi = psi_L^dagger psi_L exactly.
    charge = np.asarray(lift_to_4spinor(*args, representation=LARGE_BLOCK_LIFT))
    assert np.array_equal(charge[:, :, :2], psi_L)
    assert not np.any(charge[:, :, 2:])
    pair = lambda a: np.einsum("kmsg,knsg->kmn", a.conj(), a)
    assert np.array_equal(pair(charge), pair(psi_L))

    # Current carrier: the raw kinetic-balance lift, not normalized.
    current = np.asarray(lift_to_4spinor(
        *args, representation=RAW_KINETIC_BALANCE_LIFT))
    np.testing.assert_allclose(current, _oracle_raw_lift(psi_L, cart),
                               rtol=0, atol=1e-14)
    per_g = np.sum(np.abs(current) ** 2, axis=2)
    np.testing.assert_allclose(
        per_g, np.sum(np.abs(psi_L) ** 2, axis=2)
        * (1.0 + HALFALPHA ** 2 * np.sum(cart ** 2, axis=-1))[:, None, :],
        rtol=1e-13, atol=0)

    # q = 0 current = two-spinor velocity: <Psi_m| (2/alpha) alpha_i |Psi_n>
    # = <psi_m| 2 (k+G)_i |psi_n> (Rydberg), i.e. the charge head's kinetic
    # dipole and the current vertex are the same operator on this carrier.
    v_dirac = np.asarray(apply_dirac_velocity_to_ket(jnp.asarray(current)))
    lhs = np.einsum("kmsg,iknsg->ikmn", current.conj(), v_dirac)
    rhs = np.einsum("kmsg,kgi,knsg->ikmn", psi_L.conj(), 2.0 * cart, psi_L)
    np.testing.assert_allclose(lhs, rhs, rtol=0,
                               atol=1e-12 * np.max(np.abs(rhs)))
    assert ALPHA_FS == 2.0 * HALFALPHA

    # Refusals: a lift is named or the load refuses; the normalized lift is gone.
    with pytest.raises(TypeError):
        lift_to_4spinor(*args)
    with pytest.raises(ValueError):
        lift_to_4spinor(*args, representation="normalized_rkb")
    with pytest.raises(ValueError, match="requires bispinor_lift"):
        _check_bispinor_lift(True, None)
    with pytest.raises(ValueError, match="requires bispinor=True"):
        _check_bispinor_lift(False, RAW_KINETIC_BALANCE_LIFT)
    _check_bispinor_lift(False, None)
    _check_bispinor_lift(True, LARGE_BLOCK_LIFT)


def main():
    """P4: bispinor direct-field charge part == two-spinor V_H (hsuite fixture)."""
    import runtime
    stack = runtime.initialize_communicator_stack()
    import json
    from dataclasses import replace
    import numpy as np
    from pathlib import Path
    import jax
    from common.meta import Meta
    from gw.hartree import direct_field_matrices
    from wfn_loader import WfnLoader

    ry_mev = 13605.693122994
    fixture = Path(__file__).parent / "hsuite/fixture/WFN.h5"
    with WfnLoader(fixture, mesh=stack.mesh, backend="phdf5") as wfn:
        assert int(wfn.nspinor) == 2, "the carrier check needs a two-spinor fixture"
        sym = wfn.symmetry()
        nb = int(wfn.nbands)
        nval = min(int(wfn.nelec), nb)
        meta2 = Meta.from_system(wfn, sym, nval, nb - nval, nb, 1, False)
        meta4 = replace(meta2, nspinor=4, npol=4)
        common = dict(truncation_2d=False, nb=nb, mesh=stack.mesh)
        v_h_2c = np.asarray(direct_field_matrices(wfn, sym, meta2, **common))
        exact = direct_field_matrices(wfn, sym, meta4, include_transverse=True,
                                      **common)
        charge = np.asarray(exact.charge)
        transverse = np.asarray(exact.transverse)
    scale = float(np.max(np.abs(v_h_2c)))
    diff_mev = float(np.max(np.abs(charge - v_h_2c))) * ry_mev
    # Round-off on a ~|V_H| scale; the old four-component charge moved this by
    # meV (Fe 3d: -5.5 meV Darwin alone).
    assert diff_mev <= 1e-9 * scale * ry_mev + 1e-9, (diff_mev, scale)
    assert np.all(np.isfinite(transverse)) and np.any(transverse != 0)
    print("BISPINOR_PAULI_DIRECT_FIELD " + json.dumps(dict(
        status="PASS", rank=jax.process_index(), max_charge_minus_vh_mev=diff_mev,
        vh_scale_mev=scale * ry_mev,
        max_alpha_A_mev=float(np.max(np.abs(transverse))) * ry_mev)), flush=True)
    runtime.finalize_process(0)


def test_direct_field_requires_distributed_launcher():
    pytest.skip("run directly under the configured MPI launcher (P4)")


if __name__ == "__main__":
    main()
