"""Band operators of htransform: spin, atomic character, orbital moment.

Three steps, one owner each:

    band_operators             <psi_kn|O|psi_km> on the coarse full BZ
    interpolate_band_operator  the operator on htransform path states
    grid_moments               sum_n f(e_qn) <qn|O|qn> on a uniform q grid

Each operator is projected on the fitted band space at every coarse k and
carried into htransform's Galerkin basis as ``C^T O C^*``; that matrix is
Fourier-interpolated exactly as fH is.  A QP rotation ``U`` is unitary on
the fitted window, so ``C_QP^T O_QP C_QP^* = C^T O C^*``: the operators come
from the WFN states, the QP enters only through the fH eigenvectors.

This does not differentiate H.  The itinerant (modern-theory) orbital
moment needs the Berry connection of the fitted states and is not formed;
the orbital moment here is the atomic-sphere ``<L>`` of Loewdin-
orthogonalized PP_PSWFC projections (QE projwfc's projector; radial
functions j-averaged as in ``psp.hubbard_ops``).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.fft_helpers import make_flat_k_ifftn
from common.shard_map import shard_map
from common.staged_reshard import face_to_batch_reshard
from gw.qsgw_head import rotate_velocity_to_qp
from runtime.padding import pad_axis, padded_axis

PAULI = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=np.complex128)
_AXES = ("x", "y", "z")
_L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}


def angular_momentum_matrices(l: int) -> np.ndarray:
    """``<Y_m'|L_a|Y_m>`` (3, 2l+1, 2l+1) in QE's real-harmonic order.

    ``L_a = -i eps_abc x_b d_c`` acts on the solid harmonics of
    ``psp.radial.solid_harmonics`` (the V_NL and atomic-row owner), so the
    matrices carry its ordering and signs by construction.
    """
    from psp.radial.solid_harmonics import solid_harmonics_jax
    n = 2 * l + 1
    if l == 0:
        return np.zeros((3, 1, 1), dtype=np.complex128)
    x = np.random.default_rng(0).normal(size=(8 * n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    Y = np.asarray(solid_harmonics_jax(l, jnp.asarray(x)))          # (m, q)
    dY = np.asarray(jax.vmap(jax.jacfwd(
        lambda p: solid_harmonics_jax(l, p[None, :])[:, 0]))(jnp.asarray(x)))
    eps = np.zeros((3, 3, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[a, b, c], eps[a, c, b] = 1.0, -1.0
    LY = -1j * np.einsum("abc,qb,qmc->aqm", eps, x, dY)             # (a, q, m)
    L = np.stack([np.linalg.lstsq(Y.T, LY[a], rcond=None)[0]
                  for a in range(3)])                                 # (a, m', m)
    L2 = np.einsum("aij,ajk->ik", L, L)
    if (np.max(np.abs(L - np.conj(np.swapaxes(L, 1, 2)))) > 1e-10
            or np.max(np.abs(L2 - l * (l + 1) * np.eye(n))) > 1e-9):
        raise ValueError(f"angular_momentum_matrices: l={l} fit is not a "
                         "Hermitian L with L^2 = l(l+1)")
    return L


def _atomic_operator_table(row_labels, channels):
    """(n_op, R, R) atomic-row operators and their names.

    Per atom I: ``L_a`` on each (atom, radial function) shell.  Per
    channel ``[element:]l``: the projector on those rows.  The spin axis
    is the identity (pure-spin spinor rows).
    """
    R = len(row_labels)
    atoms = sorted({int(lab[0]) for lab in row_labels})
    elements = {int(lab[0]): lab[1] for lab in row_labels}
    L_of = {}
    full = np.zeros((len(atoms), 3, R, R), dtype=np.complex128)
    r = 0
    while r < R:
        atom, _el, _label, l, _m = row_labels[r]
        n = 2 * int(l) + 1
        if l not in L_of:
            L_of[l] = angular_momentum_matrices(int(l))
        full[atoms.index(int(atom)), :, r:r + n, r:r + n] = L_of[l]
        r += n
    ops, names = [], []
    for i, atom in enumerate(atoms):
        for a in range(3):
            ops.append(full[i, a])
            names.append(f"L_{_AXES[a]}:{elements[atom]}{atom}")
    for spec in channels:
        element, _, letter = spec.rpartition(":")
        if letter.lower() not in _L_OF:
            raise ValueError(f"orbital channel {spec!r}: want [element:]s|p|d|f")
        mask = np.array([int(lab[3]) == _L_OF[letter.lower()]
                         and (not element or lab[1] == element)
                         for lab in row_labels], dtype=np.float64)
        if not mask.any():
            raise ValueError(f"orbital channel {spec!r}: no PP_PSWFC row "
                             f"(have {sorted({(l[1], l[2]) for l in row_labels})})")
        ops.append(np.diag(mask).astype(np.complex128))
        names.append(f"char_{spec}")
    return np.stack(ops), names


def band_operators(wfn, band_range, mesh, *, pseudos, channels=()):
    """``<psi_kn|O|psi_km>`` on the full BZ, ``(n_op, nk, nb_pad, nb_pad)``.

    Operators, in order: sigma_x, sigma_y, sigma_z; L_x, L_y, L_z of each
    atom (atomic sphere); one projector per orbital ``channels`` entry.
    psi is the loader's full-BZ unfold (the Galerkin fit's own source, so
    the band gauge is ctilde's).  Bands stay sharded over the mesh; one k's
    bands (and its atomic projections) are gathered at a time.
    """
    from common.collectives import device_put_process_local
    from common.wfn_layout import band_sphere_spec
    from psp import vnl_ops
    from psp.dft_operators import padded_gvectors
    from psp.hubbard_ops import build_atwfc_setup

    if int(wfn.nspinor) != 2:
        raise ValueError("band_operators: spin and L need a spinor WFN")
    upf = {el: getattr(p, "_source_path") for el, p in pseudos.items()}
    setup, row_labels = build_atwfc_setup(wfn, upf, nspinor=2)
    table, atomic_names = _atomic_operator_table(row_labels, channels)
    names = [f"sigma_{a}" for a in _AXES] + atomic_names

    psi = wfn.load(bands=tuple(int(b) for b in band_range), k="full_bz",
                   sharding=band_sphere_spec())
    gtab = padded_gvectors(wfn, k="full_bz")
    rep = NamedSharding(mesh, P())
    kvecs, gvecs, gmask, table = (
        device_put_process_local(np.asarray(a), rep) for a in (
            np.asarray(gtab.kvecs, dtype=np.float64),
            np.asarray(gtab.gvecs, dtype=np.int32),
            np.asarray(gtab.mask, dtype=np.float64), table))
    axes = ("x", "y")
    pauli = jnp.asarray(PAULI)

    def _local(psi_l, kv, gv, gm, tab):
        def one_k(args):
            psi_k, k, G, m = args                       # (nb_loc, 2, nG)
            psi_all = jax.lax.all_gather(psi_k, axes, axis=0, tiled=True)
            spin = jnp.einsum("nsG,ast,mtG->anm", jnp.conj(psi_k), pauli,
                              psi_all, optimize=True)
            Z = vnl_ops.build_vnl_kdata_traced(k, G, setup).Z * m[None, :]
            O = jnp.conj(Z) @ Z.T
            lam, U = jnp.linalg.eigh(0.5 * (O + jnp.conj(O.T)))
            Zt = ((U / jnp.sqrt(lam)[None, :]) @ jnp.conj(U.T)).T @ Z
            proj = jnp.einsum("rG,nsG->rsn", jnp.conj(Zt), psi_k)
            proj_all = jax.lax.all_gather(proj, axes, axis=2, tiled=True)
            atomic = jnp.einsum("rsn,orq,qsm->onm", jnp.conj(proj), tab,
                                proj_all, optimize=True)
            return jnp.concatenate([spin, atomic], axis=0)
        return jax.lax.map(one_k, (psi_l, kv, gv, gm))

    run = jax.jit(shard_map(
        _local, mesh=mesh,
        in_specs=(band_sphere_spec(), P(), P(), P(), P()),
        out_specs=P(None, None, axes, None), check_vma=False))
    ops = run(psi, kvecs, gvecs, gmask, table)          # (k, op, n, m)
    del psi
    return jnp.moveaxis(ops, 1, 0), names


def _operator_R(operator_cart, source_coefficients, kgrid, mesh):
    """Lattice image ``(nk, rank, rank)`` of ONE band operator ``(1,k,n,n)``.

    The shared two-sided contraction is ``U^H O U``; here ``U = conj(C)``,
    giving ``C^T O C^*`` since ``C[k,n,a] = <B_a|psi_kn>``.  Sharded
    ``P(None,'x','y')`` like ``fH_R``.
    """
    coefficients = pad_axis(
        source_coefficients, operator_cart.shape[-1], axis=1).array
    operator_basis = rotate_velocity_to_qp(
        operator_cart, jnp.conj(coefficients), mesh=mesh)
    inverse = make_flat_k_ifftn(
        mesh, tuple(kgrid), P(None, None, None, None, 'x', 'y'),
        norm='backward')
    return jax.jit(
        lambda value: inverse(jnp.moveaxis(value, 0, 1))[:, 0],
        out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))(operator_basis)


def _check_shapes(operator_cart, source_coefficients, kgrid):
    nk, nb, _rank = source_coefficients.shape
    if (operator_cart.ndim != 4 or operator_cart.shape[1] != nk
            or operator_cart.shape[-1] != operator_cart.shape[-2]
            or operator_cart.shape[-1] < nb
            or int(np.prod(kgrid)) != nk):
        raise ValueError('Band operator, source basis and k grid disagree')


def interpolate_band_operator(operator_cart, source_coefficients,
                              path_coefficients, kpath, kgrid, mesh):
    """Return (q_carrier,component,band,band) matrices in the path basis.

    ``operator_cart`` is (component,k,n,n), ``source_coefficients`` is
    (k,n,rank), and ``path_coefficients`` is (q_carrier,rank,band), exactly
    as returned by htransform.  Matrix elements use bra n, ket m.  The q
    carrier is cut into passes of one q per device; one ``lax.scan`` runs
    every pass of a component (the phase sum is local on the matrix face,
    one face->q exchange per pass, then the local band rotation).
    """
    from bandstructure.fh_interp import build_R_grid_np
    _check_shapes(operator_cart, source_coefficients, kgrid)
    rank = int(source_coefficients.shape[2])
    nq_carrier = int(path_coefficients.shape[0])
    step = int(mesh.size)
    if (path_coefficients.shape[1] != rank or nq_carrier % step
            or len(kpath) > nq_carrier):
        raise ValueError('Path coefficients do not cover the requested k path')
    n_pass = nq_carrier // step
    q_tab = np.zeros((nq_carrier, 3))
    q_tab[:len(kpath)] = np.asarray(kpath)
    # Device d holds carrier rows [d*n_pass, (d+1)*n_pass); pass p takes
    # row d*n_pass + p on every device, so no coefficient moves.
    q_tab = np.swapaxes(q_tab.reshape(step, n_pass, 3), 0, 1)
    R = jnp.asarray(build_R_grid_np(kgrid))
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    batch = NamedSharding(mesh, P(('x', 'y'), None, None))
    exchange = face_to_batch_reshard(mesh)

    @partial(jax.jit, out_shardings=NamedSharding(
        mesh, P(('x', 'y'), None, None)))
    def _scan(q_tab, coefficients, operator_R):
        c_tab = jnp.swapaxes(coefficients.reshape(
            (step, n_pass) + coefficients.shape[1:]), 0, 1)

        def one_pass(_, xs):
            q, c = xs
            phase = jnp.exp(-2j * jnp.pi * (q @ R.T))
            value = 0.5 * jnp.einsum('qk,kmn->qmn', phase, operator_R)
            value = exchange(jax.lax.with_sharding_constraint(value, face))
            value = value + jnp.conj(jnp.swapaxes(value, -1, -2))
            c = jax.lax.with_sharding_constraint(c, batch)
            return None, jnp.einsum('qmi,qmn,qnj->qij', jnp.conj(c), value, c,
                                    optimize=True)

        _, out = jax.lax.scan(one_pass, None, (q_tab, c_tab), unroll=1)
        return jnp.swapaxes(out, 0, 1).reshape((nq_carrier,) + out.shape[2:])

    q_tab = jnp.asarray(q_tab)
    parts = []
    for a in range(operator_cart.shape[0]):
        operator_R = _operator_R(operator_cart[a:a + 1], source_coefficients,
                                 kgrid, mesh)
        parts.append(jax.block_until_ready(
            _scan(q_tab, path_coefficients, operator_R)))
        del operator_R
    return jnp.stack(parts, axis=1)


def grid_moments(fH_R, f_params, operators_R, kgrid, grid, n_states, mesh):
    """Energies and band diagonals of the fitted states on a uniform q grid.

    Returns ``(E (Nq, n_states) Ry, D (n_op, Nq, n_states), newton
    residual)`` with q in C order over ``grid``.  One pass is one q_z plane:
    a separable phase sum ``z -> y -> x`` over the coarse lattice R (local on
    the ``(rank, rank)`` face; no dense-grid FFT carrier), one face->q
    exchange, the fH eigensolve, ``f^-1`` and ``<n|O|n>`` for every
    operator.  The passes are one ``lax.scan``.
    """
    from bandstructure.fh_interp import build_R_grid_np, newton_inv
    a_f, n_f, shift = f_params
    nx, ny, nz = (int(v) for v in kgrid)
    Nx, Ny, Nz = (int(v) for v in grid)
    rank = int(fH_R.shape[-1])
    R = build_R_grid_np(kgrid).reshape(nx, ny, nz, 3)

    def phases(N, Rvals):
        return np.exp(-2j * np.pi * np.outer(np.arange(N) / N, Rvals))

    Px = jnp.asarray(phases(Nx, R[:, 0, 0, 0]))
    Py = jnp.asarray(phases(Ny, R[0, :, 0, 1]))
    Pz = jnp.asarray(phases(Nz, R[0, 0, :, 2]))
    n_plane = Nx * Ny
    carrier = padded_axis(n_plane, mesh, name="moment grid plane",
                          spec=P(('x', 'y'), None, None), axis=0).carrier
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    exchange = face_to_batch_reshard(mesh)
    stack = (fH_R,) + tuple(operators_R)

    def plane(pz, X):
        X = X.reshape(nx, ny, nz, rank, rank)
        A = jnp.einsum('z,xyzab->xyab', pz, X)
        A = jnp.einsum('jy,xyab->xjab', Py, A)
        A = 0.5 * jnp.einsum('ix,xjab->ijab', Px, A).reshape(
            n_plane, rank, rank)
        A = pad_axis(A, carrier, axis=0).array
        A = exchange(jax.lax.with_sharding_constraint(A, face))
        return A + jnp.conj(jnp.swapaxes(A, -1, -2))

    @partial(jax.jit, out_shardings=(
        NamedSharding(mesh, P(None, ('x', 'y'), None)),
        NamedSharding(mesh, P(None, None, ('x', 'y'), None)),
        NamedSharding(mesh, P())))
    def _scan(stack):
        def one_pass(_, pz):
            values, vectors = jax.vmap(jnp.linalg.eigh)(plane(pz, stack[0]))
            vectors = vectors[:, :, :n_states]
            energies, residual = newton_inv(
                a_f, n_f, shift, values[:, :n_states].real)
            diag = jnp.stack([jnp.einsum(
                'qan,qab,qbn->qn', jnp.conj(vectors), plane(pz, O), vectors,
                optimize=True).real for O in stack[1:]])
            return None, (energies, diag, residual)

        _, (E, D, res) = jax.lax.scan(one_pass, None, Pz, unroll=1)
        return E, D, jnp.max(res)

    E, D, residual = _scan(stack)
    # C order over (x, y, z): pass index is z, plane row i*Ny + j.
    E = jnp.transpose(E[:, :n_plane].reshape(Nz, Nx, Ny, n_states),
                      (1, 2, 0, 3))
    D = jnp.transpose(D[:, :, :n_plane].reshape(Nz, -1, Nx, Ny, n_states),
                      (1, 2, 3, 0, 4))
    return (E.reshape(-1, n_states), D.reshape(D.shape[0], -1, n_states),
            residual)


def occupied_sums(energies, diagonals, n_electrons, kT):
    """FD ``(mu, sum_q w sum_n f <n|O|n>)`` with E_F fixed by the count.

    ``energies`` (nq, nb) Ry and ``diagonals`` (n_op, nq, nb) on a uniform
    grid (equal weights), host arrays.  ``gw.efermi`` owns the root.
    """
    from gw.efermi import solve_smearing_occupations
    nq = int(energies.shape[0])
    mu, f = solve_smearing_occupations(
        energies, np.full(nq, 1.0 / nq), float(n_electrons), float(kT),
        state_capacity=1.0, family="fd")
    f = np.asarray(f)
    return (float(mu), np.einsum('qn,oqn->o', f, diagonals) / nq,
            float(np.max(f[:, -1])))

