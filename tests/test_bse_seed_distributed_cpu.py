"""BSE seed physics and owned storage, including a 144-device abstract case."""
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse.bse_davidson_helpers import init_bse_subspace, _bse_seed_program

jax.config.update('jax_enable_x64', True)


def _mesh(side):
    if jax.device_count() < side*side:
        pytest.skip(f'requires {side*side} virtual CPU devices')
    return Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))


def _energies(nk, nc, nv, logical):
    ec = np.full((nk, nc), np.nan)
    ev = np.full((nk, nv), np.nan)
    ec[:, :logical[0]] = .4 + .07*np.arange(logical[0])[None] + .009*np.arange(nk)[:, None]
    ev[:, :logical[1]] = -.2 - .013*np.arange(logical[1])[None] + .001*np.arange(nk)[:, None]
    return ec, ev


def test_selected_transitions_random_support_and_grid_invariance(monkeypatch):
    logical = (3, 5, 2)
    reference = None
    for side in (1, 2, 3):
        if jax.device_count() < side*side:
            continue
        mesh = _mesh(side)
        nc, nv = (((n+side-1)//side)*side for n in logical[:2])
        ec, ev = _energies(logical[2], nc, nv, logical)
        # Nonpositive/NaN physical energies remain behind valid transitions;
        # poison in padding must never be selected, irrespective of its value.
        ec[1, 2] = np.nan
        sh = NamedSharding(mesh, P(None, 'x', 'y', None))
        shape = (8, nc, nv, logical[2])
        original = np.zeros

        def no_host_bank(shape_arg, *args, **kwargs):
            assert tuple(np.atleast_1d(shape_arg)) != shape, 'global host seed bank'
            return original(shape_arg, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np, 'zeros', no_host_bank)
            rows = np.asarray(init_bse_subspace(ec, ev, 8, n_cond=3, n_val=5,
                n_random=3, seed=37, mesh=mesh, sharding=sh))
        delta = (ec[:, :3, None]-ev[:, None, :5]).reshape(-1)
        ids = np.arange(delta.size)
        picks = np.lexsort((ids, np.where(np.isfinite(delta) & (delta > 1e-12), delta, np.inf)))[:5]
        k, c, v = np.unravel_index(picks, (logical[2], 3, 5))
        expected = np.zeros(shape[:1]+shape[1:], np.complex128)
        expected[np.arange(5), c, v, k] = 1
        np.testing.assert_array_equal(rows[:5], expected[:5])
        assert np.max(np.abs(rows[:, 3:]), initial=0.) == 0.
        assert np.max(np.abs(rows[:, :, 5:]), initial=0.) == 0.
        np.testing.assert_allclose(np.sum(np.abs(rows)**2, axis=(1, 2, 3)), 1., atol=3e-15)
        assert np.all(np.abs(rows[5:, :3, :5]) > 0), 'physical sectors must all receive random support'
        repeat = init_bse_subspace(ec, ev, 8, n_cond=3, n_val=5,
            n_random=3, seed=37, mesh=mesh, sharding=sh)
        np.testing.assert_array_equal(rows, repeat)
        physical = rows[:, :3, :5]
        if reference is None:
            reference = physical
        else:
            np.testing.assert_allclose(physical, reference, atol=3e-16, rtol=3e-15)
    # If a request exceeds the finite-positive count, physical +Inf-score
    # entries still precede every padded coordinate (the old count semantics).
    rows = np.asarray(init_bse_subspace(ec, ev, 29, n_cond=3, n_val=5,
        n_random=0, mesh=mesh, sharding=sh))
    selected = np.lexsort((ids, np.where(np.isfinite(delta) & (delta > 1e-12), delta, np.inf)))[:29]
    k, c, v = np.unravel_index(selected, (logical[2], 3, 5))
    expected = np.zeros(rows.shape, np.complex128)
    expected[np.arange(29), c, v, k] = 1
    np.testing.assert_array_equal(rows, expected)
    # A requested alternate output layout belongs to that same priced program,
    # rather than an extra unpriced copy after the canonical band-face kernel.
    alternate = NamedSharding(mesh, P(None, 'y', 'x', None))
    changed = init_bse_subspace(ec, ev, 29, n_cond=3, n_val=5,
        n_random=0, mesh=mesh, sharding=alternate)
    assert changed.sharding.is_equivalent_to(alternate, 4)
    np.testing.assert_array_equal(changed, expected)


@pytest.mark.parametrize('side', (2, 3, 12))
def test_seed_optimized_hlo_keeps_transition_and_random_fields_owned(side):
    mesh = _mesh(side)
    nc_log, nv_log, nk, n_eig, n_random = 128, 128, 144, 128, 5
    nc = ((nc_log+side-1)//side)*side
    nv = ((nv_log+side-1)//side)*side
    sc, sv = NamedSharding(mesh, P(None, 'x')), NamedSharding(mesh, P(None, 'y'))
    ec = jax.ShapeDtypeStruct((nk, nc), jnp.float64, sharding=sc)
    ev = jax.ShapeDtypeStruct((nk, nv), jnp.float64, sharding=sv)
    exe = _bse_seed_program.lower(ec, ev, jax.ShapeDtypeStruct((), jnp.int64),
        sharding=NamedSharding(mesh, P(None, 'x', 'y', None)),
        nc_log=nc_log, nv_log=nv_log, n_eig=n_eig,
        n_random=n_random, dtype=jnp.complex128).compile()
    text = exe.as_text()
    local_seed = n_eig*nc*nv*nk//mesh.size
    local_transitions = nc*nv*nk//mesh.size
    candidate_bound = mesh.size*(n_eig-n_random)
    found = False
    for line in text.splitlines():
        match = re.match(r'\s*(?:ROOT )?%\S+ = (c128|f64|s64)\[([\d,]*)\]', line)
        if match:
            shape = tuple(int(s) for s in match[2].split(',') if s)
            count = math.prod(shape)
            if match[1] == 'c128':
                found = True
                assert count <= local_seed, line
            else:
                # Scalar energy/index candidates and RNG temporaries can be
                # larger than the local transition tile, never the full field.
                assert count <= max(local_seed, candidate_bound, local_transitions), line
                if match[1] == 'f64' and len(shape) == 3:
                    assert count <= max(local_transitions, candidate_bound), line
        if re.search(r'\ball-gather(?:-start)?\(', line):
            assert 'c128[' not in line, line
    assert found
    assert exe.memory_analysis().output_size_in_bytes <= 16*local_seed

