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

The orbital moment is the modern-theory one (``psp.orbital_response``): a
stored velocity of the WFN's own states (:func:`stored_velocity`) is
interpolated like any band operator and contracted on the path, and its
coarse-grid parent rows give the per-cell total (:func:`orbital_totals`).
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


def magnetization_axis(wfn):
    """``(n, source)``: the unit magnetization direction of the WFN's QE run.

    The schema that authenticates the WFN (``wfn.qe_symmetry_binding``) is
    read: ``output/magnetization/total_vec`` when nonzero (an SCF schema);
    otherwise, as in an NSCF schema, the input moment direction of the
    magnetic species (``spin_teta``/``spin_phi``, QE's angle1/angle2 in
    radians; absent means along z).  Without a schema or a magnetic
    species the axis is z, and the source says so.
    """
    import xml.etree.ElementTree as ET
    z = np.array([0.0, 0.0, 1.0])
    binding = getattr(wfn, "qe_symmetry_binding", None)
    if binding is None:
        return z, "default z (no QE schema authenticates this WFN)"
    path = binding.schema_path
    root = ET.parse(path).getroot()
    total = root.find("output/magnetization/total_vec")
    if total is not None:
        v = np.asarray(total.text.split(), dtype=np.float64)
        if np.linalg.norm(v) > 1e-6:
            return v / np.linalg.norm(v), f"{path} output total_vec"
    axes = set()
    species = root.find("input/atomic_species")
    for sp in (list(species) if species is not None else []):
        m = sp.findtext("starting_magnetization")
        if m is None or abs(float(m)) < 1e-12:
            continue
        th = float(sp.findtext("spin_teta") or 0.0)
        ph = float(sp.findtext("spin_phi") or 0.0)
        v = np.sign(float(m)) * np.array(
            [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        axes.add(tuple(np.round(v, 12)))
    if len(axes) == 1:
        return (np.asarray(axes.pop()),
                f"{path} input starting magnetization direction")
    return z, (f"default z ({path}: {len(axes)} distinct species "
               "directions, no output total)")


def _character_table(row_labels, channels):
    """(n_op, R, R) projectors on the atomic rows of each ``[element:]l``."""
    ops = []
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
    return np.stack(ops)


def band_operators(wfn, band_range, mesh, *, pseudos, channels=()):
    """``<psi_kn|O|psi_km>`` on the full BZ, ``(n_op, nk, nb_pad, nb_pad)``.

    Operators, in order: sigma_x, sigma_y, sigma_z; one projector per orbital
    ``channels`` entry, on the Loewdin-orthogonalized PP_PSWFC rows (QE
    projwfc's projector; radial functions j-averaged as in
    ``psp.hubbard_ops``).  psi is the loader's full-BZ unfold (the Galerkin
    fit's own source, so the band gauge is ctilde's).  Bands stay sharded
    over the mesh; one k's bands (and its atomic projections) are gathered at
    a time.
    """
    from common.collectives import device_put_process_local
    from common.wfn_layout import band_sphere_spec
    from psp import vnl_ops
    from psp.dft_operators import padded_gvectors
    from psp.hubbard_ops import build_atwfc_setup

    if int(wfn.nspinor) != 2:
        raise ValueError("band_operators: spin needs a spinor WFN")
    names = [f"sigma_{a}" for a in _AXES] + [f"char_{c}" for c in channels]
    setup, table = None, np.zeros((1,), dtype=np.complex128)
    if channels:
        upf = {el: getattr(p, "_source_path") for el, p in pseudos.items()}
        setup, row_labels = build_atwfc_setup(wfn, upf, nspinor=2)
        table = _character_table(row_labels, channels)

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
            if setup is None:
                return spin
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


def stored_velocity(path, *, wfn, wfn_path, sym, mesh):
    """``(parents, energies, label)`` of the WFN's stored velocity at ``path``.

    ``dipole.h5`` on a DFT WFN or an SC run's ``dipole_qsgw.h5`` on its
    ``WFN_qp.h5``, authenticated by ``file_io.dipole.velocity_stamp``.
    ``parents`` is ``(n_parent, 3, nb, nb)`` Ry Bohr (bra n, ket m, file band
    order, every stored band) at the file-wedge rows ``sym.kirr_fullids``,
    ``energies`` the file's ``band_energies`` there.  A QP velocity with no
    Sigma term refuses.  COLLECTIVE over ``mesh``.
    """
    from file_io.dipole import velocity_stamp
    from file_io.restart_bundle import read_dipole_parent_window
    label, basis, energies = velocity_stamp(path, wfn=wfn, wfn_path=wfn_path)
    if basis == "qp" and not label.startswith("v_DFT +"):
        raise ValueError(
            f"GATE qp_velocity_sigma_term: {path} is a QP-basis velocity "
            f"labelled {label!r}; want v_DFT + a Sigma term "
            "(sc_head_update = parallel_transport); why: the DFT velocity "
            "with QP energies is not d(H_QP)/dk (claim 3218)")
    rows = np.asarray(sym.kirr_fullids, dtype=np.int64)
    parents = read_dipole_parent_window(
        path, rows, 0, int(energies.shape[1]), nk_full=int(sym.nk_tot),
        mesh=mesh)
    return parents, energies[rows], label


def orbital_totals(parents, energies, sym, *, nelec, width_ry, deps_tol_ry,
                   n_ceilings=9):
    """Per-cell modern-theory orbital moment from parent rows, in mu_B.

    ``psp.orbital_response.orbital_magnetization`` at each parent k with the
    star weights of ``sym.irr_idx_k``, averaged over the group's axial
    time-odd action: the full-BZ sum without the unfold.  ``width_ry =
    None``: T = 0 at midgap of the ``nelec`` lowest bands; else the fixed-N
    Fermi-Dirac mu.  Returns ``(mu_ry, ceilings, E_c_ry, m (n_ceilings, 3))``
    for energy-ordered band ceilings from 0.6 nb to nb; ``E_c`` is the
    star-weighted mean energy of band ``c - 1``.
    """
    from gw.efermi import solve_smearing_occupations
    from psp.orbital_response import orbital_magnetization
    weights = np.bincount(np.asarray(sym.irr_idx_k),
                          minlength=len(energies)) / float(sym.nk_tot)
    order = np.argsort(energies, axis=1, kind="stable")
    E = np.take_along_axis(energies, order, axis=1)
    if width_ry is None:            # QE's count is a float: 129.99999 is 130
        n = int(round(float(nelec)))
        mu = 0.5 * (E[:, n - 1].max() + E[:, n].min())
    else:
        mu = float(solve_smearing_occupations(
            E, weights, float(nelec), float(width_ry), state_capacity=1.0,
            family="fd")[0])
    nb = E.shape[1]
    ceilings = np.unique(np.linspace(0.6 * nb, nb, n_ceilings).round()
                         ).astype(int)
    m = np.zeros((len(ceilings), 3))
    for p, perm in enumerate(order):
        v = jnp.asarray(parents[p][:, perm][:, :, perm])
        for i, c in enumerate(ceilings):
            m[i] += weights[p] * np.asarray(orbital_magnetization(
                v[:, :c, :c], E[p, :c], mu_ry=mu, width_ry=width_ry or 0.0,
                deps_tol_ry=deps_tol_ry))
    rows = np.asarray(sym.active_symmetry_rows, dtype=np.int32)
    projector = np.asarray(sym.cartesian_action(
        rows, axial=True, time_odd=True), dtype=np.float64).mean(axis=0)
    return float(mu), ceilings, weights @ E[:, ceilings - 1], m @ projector.T


def path_orbital_moments(parents, sym, window, source_coefficients,
                         path_coefficients, kpath, kgrid, mesh, energies,
                         deps_tol_ry):
    """Wavepacket moments ``(q, 3, band)`` mu_B of the path states.

    The ``window`` of the parent velocity, unfolded by the polar time-odd
    action (replicated, 48 nk nb^2 B), is interpolated like every band
    operator and contracted with the path ``energies`` (Ry).
    """
    from common.collectives import device_put_process_local, gather_to_host
    from psp.orbital_response import orbital_moments
    from symmetry_maps import unfold_file_wedge_polar_matrix
    velocity = device_put_process_local(np.moveaxis(
        unfold_file_wedge_polar_matrix(sym, parents[:, :, window, window]),
        1, 0), NamedSharding(mesh, P()))
    v_path = gather_to_host(interpolate_band_operator(
        velocity, source_coefficients, path_coefficients, np.asarray(kpath),
        kgrid, mesh))[:len(energies)]
    return np.asarray(orbital_moments(v_path, energies,
                                      deps_tol_ry=deps_tol_ry)[0])


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


def grid_moments(fH_R, f_params, operator_R_builders, kgrid, grid,
                 n_states, mesh):
    """Energies and band diagonals of the fitted states on a uniform q grid.

    Returns ``(E (Nq, n_states) Ry, D (n_op, Nq, n_states), newton
    residual)`` with q in C order over ``grid``.  One pass is one q_z plane:
    a separable phase sum ``z -> y -> x`` over the coarse lattice R (local on
    the ``(rank, rank)`` face; no dense-grid FFT carrier) and one face->q
    exchange.  The first ``lax.scan`` solves fH on every plane and keeps the
    fitted-state eigenvectors; ``fH_R`` is then deleted (consumed) and each
    operator's lattice image, built by its ``operator_R_builders`` entry
    only when needed, is read by a second scan.  At most one dense
    ``(nk, rank, rank)`` image is resident.
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
    plane_q = NamedSharding(mesh, P(None, ('x', 'y'), None))
    plane_v = NamedSharding(mesh, P(None, ('x', 'y'), None, None))

    def plane(pz, X):
        X = X.reshape(nx, ny, nz, rank, rank)
        A = jnp.einsum('z,xyzab->xyab', pz, X)
        A = jnp.einsum('jy,xyab->xjab', Py, A)
        A = 0.5 * jnp.einsum('ix,xjab->ijab', Px, A).reshape(
            n_plane, rank, rank)
        A = pad_axis(A, carrier, axis=0).array
        A = exchange(jax.lax.with_sharding_constraint(A, face))
        return A + jnp.conj(jnp.swapaxes(A, -1, -2))

    @partial(jax.jit, out_shardings=(plane_q, plane_v,
                                     NamedSharding(mesh, P())))
    def _solve(fH_R):
        def one_pass(_, pz):
            values, vectors = jax.vmap(jnp.linalg.eigh)(plane(pz, fH_R))
            energies, residual = newton_inv(
                a_f, n_f, shift, values[:, :n_states].real)
            return None, (energies, vectors[:, :, :n_states], residual)

        _, (E, V, res) = jax.lax.scan(one_pass, None, Pz, unroll=1)
        return E, V, jnp.max(res)

    @partial(jax.jit, out_shardings=plane_q)
    def _expect(O_R, V):
        def one_pass(_, xs):
            pz, v = xs
            return None, jnp.einsum('qan,qab,qbn->qn', jnp.conj(v),
                                    plane(pz, O_R), v, optimize=True).real

        _, D = jax.lax.scan(one_pass, None, (Pz, V), unroll=1)
        return D

    E, V, residual = _solve(fH_R)
    jax.block_until_ready(E)
    fH_R.delete()
    D = []
    for build in operator_R_builders:
        O_R = build()
        D.append(jax.block_until_ready(_expect(O_R, V)))
        O_R.delete()
    del V
    D = jnp.stack(D)
    # C order over (x, y, z): pass index is z, plane row i*Ny + j.
    E = jnp.transpose(E[:, :n_plane].reshape(Nz, Nx, Ny, n_states),
                      (1, 2, 0, 3))
    D = jnp.transpose(D[:, :, :n_plane].reshape(-1, Nz, Nx, Ny, n_states),
                      (0, 2, 3, 1, 4))
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

