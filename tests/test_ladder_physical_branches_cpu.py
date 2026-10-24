"""Literal four-orbital R/A and velocity-head references for the production kernel."""

import numpy as np

import pytest



def theta(x):
    return x.conj() if x.shape[-2] == 1 else np.stack([x[..., 1, :].conj(), -x[..., 0, :].conj()], axis=-2)



def fixture(ns=2):
    rng = np.random.default_rng(931 + ns)
    nk, nmu = (3, 7)
    nc, nv = (3, 2) if ns == 1 else (2, 2)
    nb = nc + nv
    psi = np.empty((nk, nb, ns, nmu), complex)
    if ns == 1:
        psi[0, :, 0, :] = np.linalg.qr(rng.normal(size=(nmu, nb)))[0].T
    else:
        basis = []
        for _ in range(nb // 2):
            x = rng.normal(size=(ns, nmu)) + 1j * rng.normal(size=(ns, nmu))
            for old in basis:
                x -= np.vdot(old, x) * old
            x /= np.linalg.norm(x)
            basis.extend([x, theta(x)])
        psi[0] = np.array(basis)
    psi[1] = np.linalg.qr(rng.normal(size=(ns * nmu, nb)) + 1j * rng.normal(size=(ns * nmu, nb)))[0].T.reshape(nb, ns, nmu)
    psi[2] = theta(psi[1])
    ev = np.array([[-0.8, -0.8] if ns == 2 else [-0.8, -0.5], [-0.9, -0.6], [-0.9, -0.6]])
    ec = np.array([[0.6, 0.6] if ns == 2 else [0.4, 0.6, 0.9], [0.4, 0.7] if ns == 2 else [0.3, 0.6, 0.8], [0.4, 0.7] if ns == 2 else [0.3, 0.6, 0.8]])
    v = np.empty((nk, nmu, nmu), complex)
    for k in (0, 1):
        x = rng.normal(size=(nmu, nmu))
        if k:
            x = x + 1j * rng.normal(size=x.shape)
        v[k] = 0.06 * x @ x.conj().T / nmu
    v[2] = v[1].conj()
    return dict(ns=ns, nk=nk, nmu=nmu, nc=nc, nv=nv, psi_c=psi[:, nv:], psi_v=psi[:, :nv], eps_c=ec, eps_v=ev, V=v, W=0.18 * v, cell_volume=80.0)



def physical_fields(f, q):
    """Actual branch labels: R=(c,k+q;v,k), A=(v,k+q;c,k)."""
    nc, nv, nk = (f['nc'], f['nv'], f['nk'])
    cs, vs, ks = np.indices((nc, nv, nk)).reshape(3, -1)
    plus = (ks + q) % nk
    m = [f['psi_c'][plus, cs], f['psi_v'][plus, vs]]
    n = [f['psi_v'][ks, vs], f['psi_c'][ks, cs]]
    energies = [f['eps_c'][plus, cs] - f['eps_v'][ks, vs], f['eps_c'][ks, cs] - f['eps_v'][plus, vs]]
    vertices = [np.einsum('ism,ism->im', a.conj(), b) for a, b in zip(m, n)]
    return dict(m=m, n=n, delta=energies, F=vertices, k=ks)



def dense(f, q, *, direct=True, ring=True):
    """Literal four-orbital kernel for all four branch blocks, in c,v,k order."""
    p = physical_fields(f, q)
    count = len(p['k'])
    eta = np.diag(np.r_[np.ones(count), -np.ones(count)])
    F = np.vstack(p['F']) * np.sqrt((2 if f['ns'] == 1 else 1) / f['nk'])
    G = F @ f['V'][q] @ F.conj().T if ring else np.zeros((2 * count, 2 * count), complex)
    Wblocks = []
    transfer = (p['k'][:, None] - p['k'][None, :]) % f['nk']
    kernel = f['W'][transfer]
    for a in range(2):
        row = []
        for b in range(2):
            row.append(np.einsum('ism,jsm,ijmn,itn,jtn->ij', p['m'][a].conj(), p['m'][b], kernel, p['n'][a], p['n'][b].conj()) / f['nk'])
        Wblocks.append(row)
    direct_matrix = np.block(Wblocks) if direct else np.zeros_like(G)
    H = np.diag(np.r_[p['delta'][0], -p['delta'][1]]) + eta @ (G - direct_matrix)
    seed = eta @ F @ f['V'][q]
    readout = f['V'][q] @ F.conj().T
    return dict(H=H, seed=seed, readout=readout, eta=eta, fields=p, F=F, direct=direct_matrix)



def response(ref, z):
    return ref['readout'] @ np.linalg.solve(z * np.eye(len(ref['H'])) - ref['H'], ref['seed'])



def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))



def dyson(f, q, z):
    p = physical_fields(f, q)
    chi = np.zeros((f['nmu'], f['nmu']), complex)
    weight = (2 if f['ns'] == 1 else 1) / f['nk']
    for F, d, sign in zip(p['F'], p['delta'], (1, -1)):
        chi += weight * F.conj().T @ ((sign / (z - sign * d))[:, None] * F)
    V = f['V'][q]
    return np.linalg.solve(np.eye(f['nmu']) - V @ chi, V) - V



def head_fields(f):
    rng = np.random.default_rng(312)
    coordinates = rng.normal(size=(3, f['nmu']))
    r = np.einsum('kcsm,am,kvsm->akcv', f['psi_c'].conj(), coordinates, f['psi_v'])
    delta = f['eps_c'][:, :, None] - f['eps_v'][:, None, :]
    velocity = 1j * delta[None] * r
    d = (velocity / delta[None]).transpose(2, 3, 1, 0).reshape(-1, 3)
    pref = 2 / (f['cell_volume'] * f['nk'] * f['ns'])
    return (velocity, delta, d, pref)



def head_response(H, d, pref, z):
    B = np.vstack([d, d.conj()])
    C = np.hstack([d.conj().T, -d.T])
    return pref * C @ np.linalg.solve(z * np.eye(len(H)) - H, B)



def independent_pole_inverse_moment(H, d, pref, polarization):
    B = np.vstack([d, d.conj()]) @ polarization
    C = polarization @ np.hstack([d.conj().T, -d.T]) * pref
    energy, right = np.linalg.eig(H)
    left = np.linalg.inv(right)
    residues = C @ right * (left @ B)
    good = energy.real > 0
    return np.sum(residues[good] / energy[good] ** 2)




def _production_payload(f, mesh):
    import jax
    import jax.numpy as jnp
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_preconditioner import compute_pair_amplitude
    from bse.bse_feast import ensure_W_R
    sh = make_bse_shardings(mesh)
    p = int(mesh.shape['x'])
    nc, nv, mu = [((n+p-1)//p)*p for n in (f['nc'], f['nv'], f['nmu'])]
    pad = lambda x, n: np.pad(x, ((0, 0), (0, n-x.shape[1]), (0, 0), (0, mu-x.shape[-1])))
    put = lambda x, layout: jax.device_put(jnp.asarray(x), layout)
    c, v = pad(f['psi_c'], nc), pad(f['psi_v'], nv)
    V = np.pad(f['V'].transpose(1, 2, 0)[..., None, None],
               ((0, mu-f['nmu']), (0, mu-f['nmu']), (0, 0), (0, 0), (0, 0)))
    W = np.pad(f['W'].transpose(1, 2, 0)[..., None, None],
               ((0, mu-f['nmu']), (0, mu-f['nmu']), (0, 0), (0, 0), (0, 0)))
    data = dict(nkx=f['nk'], nky=1, nkz=1,
        psi_c_X=put(c, sh.psi_x), psi_c_Y=put(c, sh.psi_y),
        psi_v_X=put(v, sh.psi_x), psi_v_Y=put(v, sh.psi_y),
        eps_c=put(np.pad(f['eps_c'], ((0, 0), (0, nc-f['nc']))), sh.eps),
        eps_v=put(np.pad(f['eps_v'], ((0, 0), (0, nv-f['nv']))), sh.eps),
        V_q_full=put(V, sh.W), V_q0=put(V[..., 0, 0, 0], sh.V),
        W_q=put(W, sh.W), n_cond_pad=nc, n_val_pad=nv)
    data['M'] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(data['psi_c_X'], data['psi_v_X']), sh.M)
    ensure_W_R(data, True, mesh)
    return data, sh


def _materialize_physical(f, data, sh, mesh, q, include_w, apply=None):
    import jax
    import jax.numpy as jnp
    from bse.bse_w_exact import build_finite_q_data
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from bse.bse_feast import ladder_matvec_operands
    nc, nv = int(data['n_cond_pad']), int(data['n_val_pad'])
    n = f['nc']*f['nv']*f['nk']
    trial = np.zeros((2, 2*n, nc, nv, f['nk']), complex)
    coordinates = np.indices((2, f['nc'], f['nv'], f['nk'])).reshape(4, -1)
    trial[tuple([coordinates[0], np.arange(2*n), *coordinates[1:]])] = 1
    dq = build_finite_q_data(data, (q, 0, 0), mesh, physical_branches=True)
    if apply is None:
        apply = build_bse_ring_matvec_full(mesh, f['nk'], 1, 1,
            include_W=include_w, physical_branches=True)
    out = np.asarray(apply(jax.device_put(jnp.asarray(trial), sh.X_full),
                           *ladder_matvec_operands(dq)))
    valid = np.zeros((nc, nv, f['nk']), bool)
    valid[:f['nc'], :f['nv']] = True
    np.testing.assert_array_equal(out[:, :, ~valid], 0)
    actual = out[:, :, :f['nc'], :f['nv']].transpose(0, 2, 3, 4, 1).reshape(2*n, 2*n)
    return actual, dq


def _assert_matched_production_vertices(f, data, sh, mesh, q, state):
    import jax
    import jax.numpy as jnp
    from bse.bse_ring_comm import (build_realspace_random_transition_generator,
        build_density_snapshot_operator)
    from bse.bse_w_exact import build_probe_rhs, _project_screening_actions
    nc, nv = int(data['n_cond_pad']), int(data['n_val_pad'])
    mu = int(data['V_q0'].shape[0])
    gen = build_realspace_random_transition_generator(mesh, f['nk'], 1, 1, nc, nv)
    snapshot = build_density_snapshot_operator(mesh, f['nk'], 1, 1, scatter_nu_on_y=True)
    rhs = np.asarray(build_probe_rhs(np.eye(mu), data, gen, sh))
    logical = rhs[:, :, :f['nc'], :f['nv']].transpose(0, 2, 3, 4, 1).reshape(
        len(state['H']), mu)
    spin = 2 if f['ns'] == 1 else 1
    np.testing.assert_allclose(logical[:, :f['nmu']], state['seed']/np.sqrt(spin),
                               rtol=2e-13, atol=3e-15)
    np.testing.assert_array_equal(logical[:, f['nmu']:], 0)
    z = .31+.42j
    solved = np.linalg.solve(z*np.eye(len(state['H']))-state['H'], logical)
    padded = np.zeros_like(rhs)
    padded[:, :, :f['nc'], :f['nv']] = solved.reshape(
        2, f['nc'], f['nv'], f['nk'], mu).transpose(0, 4, 1, 2, 3)
    projected = np.asarray(_project_screening_actions(
        {'value': jax.device_put(jnp.asarray(padded), sh.X_full)}, data, snapshot)['value'])
    np.testing.assert_allclose(projected[:f['nmu'], :f['nmu']], response(state, z),
                               rtol=3e-12, atol=3e-14)
    np.testing.assert_array_equal(projected[f['nmu']:], 0)
    np.testing.assert_array_equal(projected[:, f['nmu']:], 0)


def _check_physical_ring_blocks(ns, p):
    """Actual P1/P4 kernels match raw complex four-orbital algebra at every q."""
    import jax
    from jax.sharding import Mesh
    from unittest.mock import patch
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from ffi.fft import xla_reference
    jax.config.update('jax_enable_x64', True)
    devices = jax.devices('cpu')
    if len(devices) < p*p:
        pytest.fail('Run this regression with XLA_FLAGS=--xla_force_host_platform_device_count=4')
    mesh = Mesh(np.array(devices[:p*p]).reshape(p, p), ('x', 'y'))
    f = fixture(ns)
    with xla_reference(), patch('common.contract_bands.bands_gemm_ffi_enabled', lambda: False):
        data, sh = _production_payload(f, mesh)
        for include_w in (False, True):
            apply = build_bse_ring_matvec_full(mesh, f['nk'], 1, 1,
                include_W=include_w, physical_branches=True)
            for q in range(f['nk']):
                actual, dq = _materialize_physical(f, data, sh, mesh, q, include_w, apply)
                expected = dense(f, q, direct=include_w)
                np.testing.assert_allclose(actual, expected['H'], atol=3e-13, rtol=3e-12)
                state = dict(expected, H=actual)
                _assert_matched_production_vertices(f, dq, sh, mesh, q, state)
                for z in (.23j, .31+.42j):
                    np.testing.assert_allclose(response(state, z), response(expected, z), atol=3e-14, rtol=3e-12)
                    np.testing.assert_allclose(response(state, z), response(state, -z), atol=3e-14, rtol=3e-12)
                    if not include_w:
                        np.testing.assert_allclose(response(state, z), dyson(f, q, z), atol=3e-14, rtol=3e-12)
        if ns == 2 and p == 2:
            # Arbitrary Bloch phases, plus unitary mixing inside each Gamma
            # Kramers doublet, preserve the physical window without keeping
            # the canonical pair gauge. The response must not need that gauge.
            rng = np.random.default_rng(507)
            rotated = dict(f)
            for key in ('psi_c', 'psi_v'):
                value = f[key].copy()
                value *= np.exp(1j*rng.uniform(-2., 2., value.shape[:2]))[:, :, None, None]
                unitary = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))[0]
                value[0] = np.einsum('ab,bsm->asm', unitary, value[0])
                rotated[key] = value
            changed, sh = _production_payload(rotated, mesh)
            actual, dq = _materialize_physical(rotated, changed, sh, mesh, 1, True, apply)
            state = dict(dense(rotated, 1), H=actual)
            _assert_matched_production_vertices(rotated, dq, sh, mesh, 1, state)
            np.testing.assert_allclose(response(state, .31+.42j),
                response(dense(f, 1), .31+.42j), rtol=3e-12, atol=3e-14)


def _check_compiled_actions_and_head():
    """The executed P4 solve keeps branch phases through body and velocity readouts."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from unittest.mock import patch
    from bse.bse_ring_comm import (build_bse_ring_matvec_full,
        build_realspace_random_transition_generator, build_density_snapshot_operator)
    from bse.bse_feast import ladder_matvec_operands, build_preconditioner_diagonal_sharded
    from bse.bse_w_exact import (apply_screening_resolvent_block, build_probe_rhs,
        _get_block_moment_actions, _project_screening_actions, build_finite_q_data)
    from bse.head_resolvent import (build_head_operator, build_head_dipole,
        solve_head_tensor, project_head_response, build_head_rhs)
    from common.chi_from_dipole import compute_S_omega_cv
    from ffi.fft import xla_reference
    jax.config.update('jax_enable_x64', True)
    devices = jax.devices('cpu')
    assert len(devices) >= 4
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y'))
    f = fixture(2)
    with xla_reference(), patch('common.contract_bands.bands_gemm_ffi_enabled', lambda: False):
        data, sh = _production_payload(f, mesh)
        data.update(n_val=f['nv'], n_cond=f['nc'])
        apply = build_bse_ring_matvec_full(mesh, f['nk'], 1, 1,
            include_W=True, physical_branches=True)
        _, dq = _materialize_physical(f, data, sh, mesh, 1, True, apply)
        expected = dense(f, 1)
        diag = build_preconditioner_diagonal_sharded(dq, mesh, include_W=True, use_tda=False)
        mu = int(dq['V_q0'].shape[0])
        gen = build_realspace_random_transition_generator(mesh, f['nk'], 1, 1,
            int(data['n_cond_pad']), int(data['n_val_pad']))
        snapshot = build_density_snapshot_operator(mesh, f['nk'], 1, 1, scatter_nu_on_y=True)
        probe = np.eye(mu)
        z = .31+.42j
        value, residual, _, fields = apply_screening_resolvent_block(probe, z,
            dq, apply, diag, gen, snapshot, sh, max_iter=48, tol=1e-12,
            return_iters=True, operands_fn=ladder_matvec_operands, with_slope=True)
        assert float(np.max(residual)) < 2e-11
        R = np.linalg.inv(z*np.eye(len(expected['H']))-expected['H'])
        reference = response(expected, z)
        slope = -expected['readout']@R@R@expected['seed']/(2*z)
        for actual, target in ((value, reference), (fields['Wc_minus_z'], reference),
                               (fields['dWc_ds'], slope)):
            np.testing.assert_allclose(np.asarray(actual)[:f['nmu'], :f['nmu']], target,
                                       rtol=3e-10, atol=2e-13)
        rhs = build_probe_rhs(probe, dq, gen, sh)
        actions = _get_block_moment_actions(apply, sh, rhs.dtype, retain_branches=True)(
            rhs, ladder_matvec_operands(dq))
        moments = _project_screening_actions(actions, dq, snapshot)
        for j in range(4):
            target = expected['readout']@np.linalg.matrix_power(expected['H'], j)@expected['seed']/2
            np.testing.assert_allclose(np.asarray(moments[f'M{j}'])[:f['nmu'], :f['nmu']],
                                       target, rtol=2e-11, atol=3e-14)

        velocity, delta, d, pref = head_fields(f)
        d_pair = build_head_dipole(velocity, delta,
            n_cond_pad=int(data['n_cond_pad']), n_val_pad=int(data['n_val_pad']))
        points = np.array([0j, .23j, .31+.42j])
        for zero in (True, False):
            head_data = dict(data)
            if zero:
                head_data['V_q_full'] = jnp.zeros_like(data['V_q_full'])
                head_data['W_q'] = jnp.zeros_like(data['W_q'])
            head_data = build_finite_q_data(head_data, (0, 0, 0), mesh, physical_branches=True)
            operator = build_head_operator(mesh, head_data, include_w=True)
            result = solve_head_tensor(operator, head_data, d_pair, points,
                pref=pref, max_iter=48, tol=1e-12, delta_cvk=delta)
            assert np.max(result.resids) < 2e-11
            H = dense(f, 0, direct=not zero, ring=not zero)['H']
            target = np.array([head_response(H, d, pref, z) for z in points])
            np.testing.assert_allclose(result.xi, target, rtol=4e-10, atol=2e-12)
            if zero:
                independent = np.asarray(compute_S_omega_cv(velocity, delta,
                    f['cell_volume'], f['nk'], 1, f['ns'], points))
                np.testing.assert_allclose(result.xi, independent, rtol=4e-10, atol=2e-12)
            physical_rhs = build_head_rhs(d_pair, sh, physical_branches=True)
            applied = operator.matvec(physical_rhs, *operator.operands_fn(head_data))
            measured = np.asarray(project_head_response(d_pair, applied, pref=pref))
            B = np.vstack([d, d.conj()])
            C = pref*np.hstack([d.conj().T, -d.T])
            np.testing.assert_allclose(measured, C@H@B, rtol=2e-11, atol=3e-14)


def test_physical_density_and_head_oracles():
    """Run all five controls in a clean four-device CPU process, with no user setup."""
    import os
    from pathlib import Path
    import shlex
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    flags = [flag for flag in shlex.split(env.get("XLA_FLAGS", ""))
             if not flag.startswith("--xla_force_host_platform_device_count")]
    env["XLA_FLAGS"] = " ".join(flags + ["--xla_force_host_platform_device_count=4"])
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(root/"src") + os.pathsep + env.get("PYTHONPATH", "")
    worker = """import runtime
import runpy, sys
ns = runpy.run_path(sys.argv[1])
for spin in (1, 2):
    for mesh in (1, 2):
        ns['_check_physical_ring_blocks'](spin, mesh)
ns['_check_compiled_actions_and_head']()
print('PHYSICAL_R_A_ORACLES: all five controls PASS')
"""
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run([sys.executable, "-c", worker, str(Path(__file__).resolve())],
                            cwd=root, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'all five controls PASS' in result.stdout
