"""The contour block accumulate's loop structure on a host device (CPU, seconds).

On ROCm every trip of a loop whose bound is device data costs a predicate D2H and a
stream sync (a known trip count skips it).  ``contour_block_accumulate_local`` keeps one
device-bounded loop (the live output blocks) and runs the groups as a loop of static
trip count, so a chunk costs one sync per live output block; its bytes are those of the
group-major loop it replaces (each element gets its groups' terms in group order).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _loops(jaxpr):
    out = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in ("while", "scan"):
            out.append(eqn.primitive.name)
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (tuple, list)) else (v,)):
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    out += _loops(getattr(inner, "jaxpr", inner))
    return out


def _reference(acc, contribution, projection, valid, sets, terms, outputs):
    acc = np.array(acc)
    for t in range(terms):
        for o in range(outputs):
            acc[o, :, :valid[0], :valid[1]] += projection[t, o] * contribution[t, :, :valid[0], :valid[1]]
    return acc


def test_groups_are_a_static_trip_loop_and_match_the_term_order_sum():
    from gw.contour_accumulator import contour_block_accumulate_local
    rng = np.random.default_rng(3)
    n_out, terms_cap, sets = 5, 6, 2
    shape = (1, 3, 4)
    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    acc, contribution, projection = c(n_out, *shape), c(terms_cap, *shape), c(terms_cap, n_out)
    valid = np.array([2, 3], np.int32)
    f = lambda a, r, p, t, o: contour_block_accumulate_local(a, r, p, jnp.asarray(valid), m0=0, n0=0,
                                                            sets=sets, terms=t, outputs=o, out_block=2)
    jaxpr = jax.make_jaxpr(f)(acc, contribution, projection, jnp.int32(4), jnp.int32(3))
    loops = _loops(jaxpr.jaxpr)
    # one device-bounded while (output blocks) around one static-trip loop (groups)
    assert loops.count("while") == 1 and loops.count("scan") == 1, loops
    for terms, outputs in ((4, 3), (6, 5), (2, 5), (0, 5), (6, 1)):
        got = np.asarray(jax.jit(f)(acc, contribution, projection, jnp.int32(terms), jnp.int32(outputs)))
        np.testing.assert_allclose(got, _reference(acc, contribution, projection, valid, sets, terms, outputs),
                                   rtol=1e-13, atol=1e-13)
        assert np.all(got[outputs:] == acc[outputs:])
