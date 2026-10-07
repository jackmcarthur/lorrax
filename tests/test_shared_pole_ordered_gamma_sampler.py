"""Literal causal branches, magnetic odd channel and exact scalar derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.shared_pole_gates import ordered_shared_pole_value, ordered_shared_pole_weights
from gw.shared_pole_head import (
    realized_gamma_correlation_sampler, realized_ordered_gamma_correlation_sampler)


def identity(a, t):
    return a, t


def transpose_average(a, t):
    return .5 * (a + t), t


def magnetic_average(a, t):
    permutation = jnp.asarray([1, 0, 3, 2])
    return .5 * (a + t[:, permutation][:, :, permutation]), t


@pytest.fixture
def mesh():
    # This fixture is a P1 unit geometry. Actual MPI P4 is a separate plant.
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def data(mesh, *, real=False):
    rng = np.random.default_rng(7102026)
    b = rng.normal(size=(1, 4, 4)).astype(complex)
    if not real:
        b += 1j * rng.normal(size=b.shape)
    b[..., 2:] *= 1e20
    poles2 = np.array([[.2, 1.3, -4., 0.]])
    counts = np.array([2], np.int32)
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    return b, poles2, counts, jax.device_put(b, face), jax.device_put(poles2, rep), jax.device_put(counts, rep)


def literal(b, poles2, z, count=2):
    value = np.zeros((1, 4, 4), complex)
    slope = value.copy()
    for p in range(count):
        omega = np.sqrt(poles2[0, p])
        residue = np.outer(b[0, :, p], b[0, :, p].conj())
        value[0] += residue / (2 * omega * (z - omega))
        value[0] -= residue.T / (2 * omega * (z + omega))
        slope[0] -= residue / (2 * omega * (z - omega) ** 2)
        slope[0] += residue.T / (2 * omega * (z + omega) ** 2)
    return value, slope


def realize_literal(a, realize):
    t = a.swapaxes(-1, -2)
    if realize is transpose_average:
        return .5 * (a + t)
    if realize is magnetic_average:
        permutation = [1, 0, 3, 2]
        return .5 * (a + t[:, permutation][:, :, permutation])
    return a


@pytest.mark.parametrize("z", [.2j, .7 + .3j, -.7 + .3j, 1 + .02j, 2 + .4j])
@pytest.mark.parametrize("realize", [identity, transpose_average, magnetic_average])
def test_literal_value_dz_and_existing_ordered_value(mesh, z, realize):
    b, poles2, _, db, dp, dc = data(mesh)
    evaluate = realized_ordered_gamma_correlation_sampler(
        mesh, realize, representation="scalar-ordered-ph")
    value, slope = evaluate(jnp.asarray(z, jnp.complex128), db, dp, dc)
    expected, derivative = literal(b, poles2, z)
    expected = realize_literal(expected, realize)
    derivative = realize_literal(derivative, realize)
    np.testing.assert_allclose(value, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(slope, derivative, rtol=3e-13, atol=3e-13)
    face = NamedSharding(mesh, P(None, "x", "y"))
    assert value.sharding == face and slope.sharding == face

    # Existing value owner independently contracts the conjugated partner.
    from distrib_la import matmul
    def mm(a, b, **kwargs):
        return matmul(a, b, mesh=mesh, backend="auto", batched_route="auto", **kwargs)
    @jax.jit(out_shardings=face)
    def existing_value(site, factors, poles, counts):
        active = jnp.arange(poles.shape[-1])[None] < counts[:, None]
        model = factors, poles, active
        return ordered_shared_pole_value(model, model, site, matmul=mm)
    raw = existing_value(jnp.asarray(z), db, dp, dc)
    np.testing.assert_allclose(value, realize_literal(np.asarray(raw), realize), rtol=3e-13, atol=3e-13)


def test_magnetic_odd_channel_and_same_frequency_endpoint_transport(mesh):
    b, poles2, _, db, dp, dc = data(mesh)
    z = .7 + .3j
    evaluate = realized_ordered_gamma_correlation_sampler(mesh, magnetic_average,
                                                         representation="scalar-ordered-ph")
    value, _ = evaluate(jnp.asarray(z), db, dp, dc)
    residues = [np.outer(b[0, :, p], b[0, :, p].conj()) for p in range(2)]
    even_wrong = sum(r / (z*z - poles2[0, p]) for p, r in enumerate(residues))[None]
    even_wrong = realize_literal(even_wrong, magnetic_average)
    assert np.max(abs(np.asarray(value) - even_wrong)) > .1
    expected, _ = literal(b, poles2, z)
    # Wrong causal conjugation, including coefficient conjugation, is visible.
    wrong = .5 * (expected + expected.conj()[:, [1, 0, 3, 2]][:, :, [1, 0, 3, 2]])
    assert np.max(abs(np.asarray(value) - wrong)) > .1
    assert np.max(abs(np.asarray(value) - np.asarray(value).swapaxes(-1, -2))) > .1


def test_frequency_transpose_and_causal_adjoint_relations(mesh):
    _, _, _, b, p, c = data(mesh)
    evaluate = realized_ordered_gamma_correlation_sampler(mesh, identity,
                                                         representation="scalar-ordered-ph")
    z = .7 + .3j
    w, d = evaluate(jnp.asarray(z), b, p, c)
    minus, minus_d = evaluate(jnp.asarray(-z), b, p, c)
    conjugate, conjugate_d = evaluate(jnp.asarray(z.conjugate()), b, p, c)
    np.testing.assert_allclose(minus, np.asarray(w).swapaxes(-1, -2), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(minus_d, -np.asarray(d).swapaxes(-1, -2), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(conjugate, np.asarray(w).conj().swapaxes(-1, -2), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(conjugate_d, np.asarray(d).conj().swapaxes(-1, -2), rtol=3e-13, atol=3e-13)


def test_real_residues_reduce_to_even_sampler_with_chain_rule(mesh):
    _, _, _, b, p, c = data(mesh, real=True)
    z = .7 + .3j
    ordered = realized_ordered_gamma_correlation_sampler(mesh, identity, representation="scalar-ordered-ph")
    even = realized_gamma_correlation_sampler(mesh, identity, representation="scalar-trs-even-s")
    w, dz = ordered(jnp.asarray(z), b, p, c)
    even_w, ds = even(jnp.asarray(z*z), b, p, c)
    np.testing.assert_allclose(w, even_w, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(dz, 2*z*ds, rtol=3e-13, atol=3e-13)


def test_scalar_derivative_refinement_and_nonzero_ds_conversion(mesh):
    _, _, _, b, p, c = data(mesh)
    z = .7 + .3j
    evaluate = realized_ordered_gamma_correlation_sampler(mesh, identity,
                                                         representation="scalar-ordered-ph")
    _, dz = evaluate(jnp.asarray(z), b, p, c)
    errors = []
    for h in [1e-3, 5e-4, 1e-4]:
        wp, _ = evaluate(jnp.asarray(z+h), b, p, c)
        wm, _ = evaluate(jnp.asarray(z-h), b, p, c)
        errors.append(float(np.max(abs(np.asarray((wp-wm)/(2*h)-dz)))))
    assert errors[1] < .3*errors[0] and errors[2] < .05*errors[1]
    assert errors[-1]/np.max(abs(np.asarray(dz))) < 1e-7
    h = 1e-5
    wp, _ = evaluate(jnp.asarray(np.sqrt(z*z+h)), b, p, c)
    wm, _ = evaluate(jnp.asarray(np.sqrt(z*z-h)), b, p, c)
    np.testing.assert_allclose((wp-wm)/(2*h), dz/(2*z), rtol=2e-9, atol=2e-9)


def test_empty_and_inactive_negative_poles_never_enter_sqrt(mesh):
    _, _, _, b, p, c = data(mesh)
    evaluate = realized_ordered_gamma_correlation_sampler(mesh, identity,
                                                         representation="scalar-ordered-ph")
    w, d = evaluate(jnp.asarray(1+0j), b, p, 0*c)
    np.testing.assert_array_equal(w, np.zeros((1, 4, 4)))
    np.testing.assert_array_equal(d, np.zeros((1, 4, 4)))
    active = jnp.arange(p.shape[-1])[None] < c[:, None]
    positive, negative = ordered_shared_pole_weights(p, active, jnp.asarray(.7+.3j))
    assert np.isfinite(np.asarray(positive)).all() and np.isfinite(np.asarray(negative)).all()
    np.testing.assert_array_equal(np.asarray(positive)[:, 2:], 0)
    np.testing.assert_array_equal(np.asarray(negative)[:, 2:], 0)


@pytest.mark.parametrize("representation", ["scalar-trs-even-s", "charge-ordered-z", "unknown", None])
def test_unbound_or_wrong_representation_refuses(mesh, representation):
    with pytest.raises(ValueError, match="requires scalar-ordered-ph"):
        realized_ordered_gamma_correlation_sampler(mesh, identity, representation=representation)


def test_scalar_complex_single_parent_contract(mesh):
    _, _, _, b, p, c = data(mesh)
    evaluate = realized_ordered_gamma_correlation_sampler(mesh, identity,
                                                         representation="scalar-ordered-ph")
    with pytest.raises(ValueError, match="scalar complex z"):
        evaluate(jnp.asarray(.7), b, p, c)
    with pytest.raises(ValueError, match="one Γ"):
        evaluate(jnp.asarray(.7+.3j), jnp.concatenate((b, b)), jnp.concatenate((p, p)), jnp.concatenate((c, c)))
