"""First-order direct bulk photon head from the authenticated dipole vertex.

The six rows are three derivatives of the charge vertex followed by three
uniform current vertices.  Only the final 6 by 6 tensors are replicated;
band pairs remain tiled over both processor axes.  The metallic diagonal
response and the photon contact are separate inputs to the Γ-cell solve.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.bispinor_init import HALFALPHA
from gw.qsgw_head import _pad_head_band_manifold, _mesh_xy


def packed_gamma_vectors(photon_g0_vectors, layout, mesh):
    """Use the existing photon-layout owner for the four Γ plane-wave rows."""
    from common.collectives import device_put_process_local
    from gw.photon_layout import pack_photon_channel_vectors

    if photon_g0_vectors is None or len(photon_g0_vectors) != 4:
        raise ValueError("direct photon Γ needs four authenticated G=0 vectors")
    x = pack_photon_channel_vectors(tuple(photon_g0_vectors), layout,
                                    mesh, axis_name="x")[0]
    sh_y = NamedSharding(mesh, P(None, "y"))
    y = pack_photon_channel_vectors(tuple(
        device_put_process_local(row, sh_y) for row in photon_g0_vectors),
        layout, mesh, axis_name="y")[0]
    return x, y


def subtract_bare_tt_from_bank(packed_v, photon_g0_vectors, *, layout,
                                mesh, wfn, meta):
    """Keep the direct Γ bare TT exchange in V, outside the body W Dyson."""
    from vcoul import CoulombGeometry
    from gw.photon_layout import add_photon_q0_low_rank
    from gw.v_q_bispinor import _tt_head_tensor

    x, y = packed_gamma_vectors(photon_g0_vectors, layout, mesh)
    tensor = _tt_head_tensor(bvec=CoulombGeometry.from_wfn(wfn).bvec,
        cell_volume=float(meta.cell_volume), sys_dim=int(meta.sys_dim),
        kgrid=tuple(meta.kgrid))
    # V artifact already includes D_TT=-<v P_T>/Omega.  Subtract exactly
    # that rank-four q=0 tile from the screening root, retaining it for X.
    removal = np.zeros((4, 4), dtype=np.complex128)
    removal[1:, 1:] = tensor / float(meta.cell_volume)
    left = jax.lax.with_sharding_constraint(jnp.conj(x),
        NamedSharding(mesh, P(None, "x")))
    right = jax.lax.with_sharding_constraint(
        jnp.asarray(removal) @ y, NamedSharding(mesh, P(None, "y")))
    return add_photon_q0_low_rank(packed_v, layout, mesh,
        left_rows_X=left, right_rows_Y=right)


@lru_cache(maxsize=16)
def _contact_projection_program(mesh):
    ax_x, ax_y = _mesh_xy(mesh)

    def local(x, contact, y):
        reduced = jnp.einsum("am,mn,bn->ab", jnp.conj(x),
                             contact[0], y, optimize=True)
        return jax.lax.psum(reduced, (ax_x, ax_y))

    return jax.jit(shard_map(local, mesh=mesh,
        in_specs=(P(None,"x"), P(None,"x","y"), P(None,"y")),
        out_specs=P(None,None), check_vma=False))


def direct_photon_contact(contact_packed, photon_g0_vectors, *, layout, mesh):
    """Project the bank's one FD contact onto the four uniform vertices."""
    x, y = packed_gamma_vectors(photon_g0_vectors, layout, mesh)
    return _contact_projection_program(mesh)(x, contact_packed, y)


def add_direct_gamma_field(packed, coefficient, *, gamma_vectors, layout, mesh):
    """Inject one 4×4 direct Γ coefficient through the packed q0 owner."""
    from gw.photon_layout import add_photon_q0_low_rank

    x, y = gamma_vectors
    coefficient = jnp.asarray(coefficient, dtype=packed.dtype)
    if coefficient.shape != (4, 4):
        raise ValueError("direct photon Γ coefficient must be 4×4")
    left = jax.lax.with_sharding_constraint(jnp.conj(x),
        NamedSharding(mesh, P(None, "x")))
    right = jax.lax.with_sharding_constraint(coefficient @ y,
        NamedSharding(mesh, P(None, "y")))
    return add_photon_q0_low_rank(packed, layout, mesh,
        left_rows_X=left, right_rows_Y=right)


@lru_cache(maxsize=16)
def _interband_program(mesh: Mesh, nb_logical: int, degeneracy_ry: float):
    from jax.sharding import PartitionSpec as P

    ax_x, ax_y = _mesh_xy(mesh)

    def local(v, e_x, e_y, f_x, f_y, frequencies, scale):
        nx, ny = v.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        delta = e_y[:, None, :] - e_x[:, :, None]
        occupation = f_x[:, :, None] - f_y[:, None, :]
        active = ((ix[:, None] < nb_logical) &
                  (iy[None, :] < nb_logical))[None]
        offdiag = active & (ix[None, :, None] != iy[None, None, :])
        regular = offdiag & (jnp.abs(delta) > degeneracy_ry)
        inverse = jnp.where(regular, 1 / jnp.where(regular, delta, 1.0), 0.0)
        # In Ry and bohr units, ∂_k H has the dipole-file velocity.  The
        # packed Breit current uses Gamma_raw=(alpha_FS/2)*v_Ry.
        vertex = jnp.concatenate((v * inverse[None], HALFALPHA * v), axis=0)
        vertex = jnp.where(offdiag[None], vertex, 0.0)

        def contract(weight):
            value = jnp.einsum("akij,kij,bkij->ab", jnp.conj(vertex),
                                weight, vertex, optimize=True)
            return jax.lax.psum(value, (ax_x, ax_y))

        # One orientation per directed pair.  scale is half the incumbent
        # energy-ordered S-tensor prefactor because both directions appear.
        def one(_, z):
            denom = z - delta
            safe_denom = jnp.where(regular, denom, 1.0 + 0.0j)
            weight = jnp.where(regular,
                scale * occupation / safe_denom, 0.0)
            safe_z = jnp.where(z == 0, 1.0 + 0.0j, z)
            slope = jnp.where(regular & (z != 0),
                -scale * occupation /
                (2 * safe_z * safe_denom * safe_denom), 0.0)
            return None, (contract(weight), contract(slope))

        _, (value, slope) = jax.lax.scan(one, None, frequencies, unroll=1)
        moments = jnp.stack([contract(jnp.where(
            regular, scale * occupation * delta**power, 0.0))
            for power in range(4)])
        skipped = jax.lax.psum(jnp.sum(offdiag & ~regular &
                                     (jnp.abs(occupation) > 0)), (ax_x, ax_y))
        return value, slope, moments, skipped

    return jax.jit(shard_map(local, mesh=mesh,
        in_specs=(P(None,None,"x","y"), P(None,"x"), P(None,"y"),
                  P(None,"x"), P(None,"y"), P(None), P()),
        out_specs=(P(None,None,None), P(None,None,None),
                   P(None,None,None), P()), check_vma=False))


def direct_photon_interband_tensors(velocity_cart, energies_kn_ry,
                                    occupations_kn, frequencies_ry, *,
                                    mesh: Mesh, nb_logical: int,
                                    cell_volume: float, nk_tot: int,
                                    nspin: int, nspinor_wfn: int,
                                    degeneracy_ry: float = 1e-8):
    """Return ordered 6×6 value, d/d(z²), and 1/z..1/z⁴ coefficients.

    Rows 0:3 multiply the mini-BZ Cartesian q; rows 3:6 are current.  The
    declared first-order approximation drops interband charge vertices at
    exactly degenerate pairs, whose eigenstate derivative is undefined.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    z = jnp.atleast_1d(jnp.asarray(frequencies_ry, dtype=jnp.complex128))
    if (v.ndim != 4 or tuple(v.shape[:2]) != (3, nk_tot)
            or v.shape[-1] != v.shape[-2] or e.shape != f.shape
            or tuple(e.shape) != (nk_tot, nb_logical)
            or nb_logical > v.shape[-1]):
        raise ValueError("direct photon head requires aligned 3×Nk×Nb×Nb dipoles and Nk×Nb states")
    if not (cell_volume > 0 and nk_tot > 0 and nspin > 0 and
            nspinor_wfn > 0 and degeneracy_ry > 0):
        raise ValueError("direct photon head requires positive normalization and degeneracy threshold")
    points = np.asarray(frequencies_ry, dtype=np.complex128)
    if np.any((np.imag(points) < 0) |
              ((np.imag(points) == 0) & (np.real(points) != 0))):
        raise ValueError("direct photon head samples must be causal or exactly static")
    if v.shape[-1] > nb_logical:
        pad = int(v.shape[-1]) - int(nb_logical)
        e = jnp.pad(e, ((0, 0), (0, pad)))
        f = jnp.pad(f, ((0, 0), (0, pad)))
    v, e, f, _ = _pad_head_band_manifold(v, e, f, jnp.zeros_like(e), mesh=mesh)
    # Match the incumbent S_ab normalization for its charge-charge block.
    scale = 2.0 / (cell_volume * nk_tot * nspin * nspinor_wfn)
    result = _interband_program(mesh, int(nb_logical), float(degeneracy_ry))(
        v, e, e, f, f, z, jnp.asarray(scale, dtype=jnp.complex128))
    skipped = int(np.asarray(result[3]))
    if skipped:
        raise ValueError("GATE photon_direct_degenerate_occupation: "
                         f"{skipped} near-degenerate directed band pairs have "
                         "unequal occupations; first-order charge jets are "
                         "undefined for those pairs")
    return result


@jax.jit
def project_first_order_photon_response(tensor, q_cart):
    """Map the six first-order vertices onto C/T at each Γ-cell q."""
    q = jnp.asarray(q_cart, dtype=jnp.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_cart must have three Cartesian columns")
    rows = jnp.zeros((q.shape[0], 4, 6), dtype=jnp.float64)
    rows = rows.at[:, 0, :3].set(q)
    rows = rows.at[:, 1:, 3:].set(jnp.eye(3)[None])
    return jnp.einsum("qia,...ab,qjb->...qij", rows, tensor, rows,
                      optimize=True)


def metal_head_surface_tensors(velocity_cart, surface_weight_kn, *, mesh,
                               nb_logical, cell_volume, nk_tot, nspin,
                               nspinor_wfn):
    """Use the incumbent sharded Drude contraction and current-map FD DOS."""
    from gw.qsgw_head import head_drude_tensor_sharded

    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if surface.shape != (nk_tot, nb_logical):
        raise ValueError("metal head surface weights do not match the band manifold")
    stored = int(velocity_cart.shape[-1])
    if stored > nb_logical:
        surface_stored = jnp.pad(surface, ((0, 0), (0, stored-nb_logical)))
    else:
        surface_stored = surface
    drude = head_drude_tensor_sharded(velocity_cart, surface_stored,
        mesh=mesh, nb_logical=nb_logical, cell_volume=cell_volume,
        nk_tot=nk_tot, nspin=nspin, nspinor=nspinor_wfn)
    dos = (2.0 / (cell_volume * nk_tot * nspin * nspinor_wfn)
           * jnp.sum(surface))
    return drude, dos


@jax.jit
def metal_intraband_photon_response(q_cart, frequencies_ry, drude_tensor,
                                     static_dos):
    """Leading FD metal response with distinct dynamic and static limits.

    At ``z != 0`` the longitudinal intraband tensor is fixed by the
    Fermi-surface Drude tensor ``D_ab``: CC=qDq/z² and CT=qD·Gamma/z.
    At ``z=0`` the finite-q limit is Thomas–Fermi CC=-DOS and the normal
    current bubble is -Gamma·D·Gamma. Dynamic TT intraband starts at
    higher order in q; its separate uniform contact remains in the photon
    bank. No chemical-potential damping or artificial insulating gap enters.
    """
    q = jnp.asarray(q_cart, dtype=jnp.float64)
    z = jnp.atleast_1d(jnp.asarray(frequencies_ry, dtype=jnp.complex128))
    D = jnp.asarray(drude_tensor, dtype=jnp.complex128)
    if q.ndim != 2 or q.shape[1] != 3 or D.shape != (3, 3):
        raise ValueError("metal head requires q[nq,3] and Drude[3,3]")
    qD = q @ D
    qDq = jnp.einsum("qa,qa->q", qD, q)
    static = jnp.zeros((q.shape[0], 4, 4), dtype=jnp.complex128)
    static = static.at[:, 0, 0].set(-static_dos)
    static = static.at[:, 1:, 1:].set(-HALFALPHA**2 * D)

    def one(_, point):
        safe = jnp.where(point == 0, 1.0 + 0.0j, point)
        dynamic = jnp.zeros_like(static)
        dynamic = dynamic.at[:, 0, 0].set(qDq / safe**2)
        dynamic = dynamic.at[:, 0, 1:].set(HALFALPHA * qD / safe)
        dynamic = dynamic.at[:, 1:, 0].set(HALFALPHA * (D @ q.T).T / safe)
        return None, jnp.where(point == 0, static, dynamic)

    _, values = jax.lax.scan(one, None, z, unroll=1)
    return values


@jax.jit
def _direct_gamma_chunk(q, bare, weight, interband, slopes, coefficients,
                        frequencies, drude, dos, contact):
    """One small-matrix Γ cubature chunk; large centroid matrices never enter."""
    from gw.head_correction import _solve_photon_head

    identity = jnp.eye(4, dtype=jnp.complex128)[None]
    contact = jnp.asarray(contact, dtype=jnp.complex128)
    winf = jnp.linalg.solve(identity + bare @ contact, bare)
    projected = project_first_order_photon_response(coefficients, q)
    qD = q @ drude
    qDq = jnp.einsum("qa,qa->q", qD, q)
    r1 = projected[0].at[:, 0, 1:].add(HALFALPHA * qD)
    r1 = r1.at[:, 1:, 0].add(HALFALPHA * (drude @ q.T).T)
    r2 = projected[1].at[:, 0, 0].add(qDq)
    response_coefficients = (r1, r2, projected[2], projected[3])
    expansion = []
    for k, rk in enumerate(response_coefficients):
        rhs = rk @ winf
        for i in range(k):
            rhs = rhs + response_coefficients[i] @ expansion[k-i-1]
        expansion.append(winf @ rhs)
    moment_sum = jnp.stack([jnp.einsum("q,qab->ab", weight, x) / 2
                             for x in expansion])
    constant_sum = jnp.einsum("q,qab->ab", weight, winf - bare)
    bare_sum = jnp.einsum("q,qab->ab", weight, bare)

    def one(_, operands):
        z, tensor, derivative = operands
        pi = (project_first_order_photon_response(tensor, q)
              + metal_intraband_photon_response(q, z[None], drude, dos)[0])
        dpi = project_first_order_photon_response(derivative, q)
        safe = jnp.where(z == 0, 1.0 + 0.0j, z)
        dpi = dpi.at[:, 0, 0].add(-qDq / safe**4)
        dpi = dpi.at[:, 0, 1:].add(-HALFALPHA * qD / (2 * safe**3))
        dpi = dpi.at[:, 1:, 0].add(
            -HALFALPHA * (drude @ q.T).T / (2 * safe**3))
        dpi = jnp.where(z == 0, 0.0, dpi)
        def solve(response, derivative):
            W, lhs = _solve_photon_head(bare, response - contact)
            value = jnp.einsum("q,qab->ab", weight, W - winf)
            slope = jnp.einsum("q,qab->ab", weight, W @ derivative @ W)
            residual = lhs @ W - bare
            error = jnp.max(jnp.where(weight > 0,
                jnp.linalg.norm(residual, axis=(-2,-1)) /
                jnp.maximum(jnp.linalg.norm(bare, axis=(-2,-1)), 1e-300), 0.0))
            return value, slope, error

        value, slope, error = solve(pi, dpi)
        # The minus-q partner W_q(-conj z) conjugates the response at -q, then
        # uses the *same* V and contact. q reversal flips CT/TC, while the bare
        # transverse projector is even. Conjugating screened W would also
        # conjugate the Hall contact and break its shared moments.
        parity = jnp.diag(jnp.array((1, -1, -1, -1), dtype=pi.dtype))
        partner_value, partner_slope, partner_error = solve(
            jnp.conj(parity @ pi @ parity),
            jnp.conj(parity @ dpi @ parity))
        return None, (value, slope, partner_value, partner_slope,
                      jnp.maximum(error, partner_error))

    _, (value, slope, partner_value, partner_slope, errors) = jax.lax.scan(
        one, None, (frequencies, interband, slopes), unroll=1)
    return (value, slope, partner_value, partner_slope, constant_sum,
            moment_sum, bare_sum, jnp.max(errors))


def _bulk_sphere_rule(geometry, kgrid, analytic_bare, chunk_size):
    """Radial/angular screened cubature inside vcoul's excised bulk sphere.

    The weights are calibrated to the service's exact bare 8π/q² sphere
    integral.  Unlike adding the bare sphere tensor after Dyson, these points
    run through the same coupled 4×4 solve as every outer mini-BZ point.
    """
    from vcoul import (bulk_photon_D_raw, gauss_legendre_interval,
                       minibz_inscribed_sphere_r2)

    radius = np.sqrt(minibz_inscribed_sphere_r2(
        geometry.bvec, kgrid, is_2d=False))
    radial, wr = gauss_legendre_interval(8, 0.0, 1.0)
    cosine, wc = gauss_legendre_interval(12, -1.0, 1.0)
    phi = 2.0 * np.pi * np.arange(24) / 24
    transverse = np.sqrt(1.0 - cosine**2)[:, None]
    direction = np.stack(np.broadcast_arrays(
        transverse * np.cos(phi), transverse * np.sin(phi),
        cosine[:, None]), axis=-1)
    q = (radius * radial[:, None, None, None] * direction[None]).reshape(-1, 3)
    bare, q2 = bulk_photon_D_raw(q)
    weight = (float(analytic_bare) * q2 / (8.0 * np.pi)
              * np.broadcast_to(wr[:, None, None] * wc[None, :, None]
                                / (2.0 * len(phi)),
                                (len(radial), len(cosine), len(phi))).ravel())
    if len(q) > chunk_size:
        raise ValueError("bulk photon sphere rule exceeds direct Γ chunk")
    q_out = np.zeros((chunk_size, 3), np.float64)
    bare_out = np.zeros((chunk_size, 4, 4), np.float64)
    weight_out = np.zeros(chunk_size, np.float64)
    q_out[:len(q)], bare_out[:len(q)], weight_out[:len(q)] = q, bare, weight
    if not np.isclose(np.dot(weight, bare[:, 0, 0]), analytic_bare,
                      rtol=1e-13):
        raise ValueError("GATE photon_direct_sphere_bare: sphere rule lost bare normalization")
    return q_out, bare_out, weight_out


def build_direct_photon_head(velocity_cart, wfns, occupation_state, *,
                             contact_packed, photon_g0_vectors, layout,
                             mesh, meta, wfn, frequencies_ry, print_fn=print):
    """Sample the direct bulk Γ 4×4 Dyson once, then hand sectors small data.

    This first-order model uses dipole charge jets, a velocity-based
    approximation to the raw Breit current, and the existing FD contact.
    It does not fold wings or microscopic body fields. The coupled screened
    sphere is integrated radially and angularly; Sobol samples its exterior.
    """
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry, get_kernel, iter_minibz_photon_samples
    from common.collectives import device_put_process_local

    if int(meta.sys_dim) != 3:
        raise ValueError("GATE photon_direct_head_bulk: first-order direct Γ requires sys_dim=3")
    if occupation_state is None or occupation_state.smearing_family != "fd":
        raise ValueError("GATE photon_direct_head_fd: metallic direct Γ requires current-map Fermi-Dirac occupations")
    nb = int(meta.b_id_4_chi_user) - int(meta.b_id_0)
    if velocity_cart.shape[-1] < nb or wfns.enk.shape[1] < nb:
        raise ValueError("GATE photon_direct_head_bands: dipole and response manifolds differ")
    z = np.asarray(frequencies_ry, dtype=np.complex128).reshape(-1)
    e = wfns.enk[:, :nb]
    f = occupation_state.f_kn[:, :nb]
    tensors = direct_photon_interband_tensors(velocity_cart, e, f, z,
        mesh=mesh, nb_logical=nb, cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
        nspinor_wfn=int(meta.nspinor_wfnfile))
    width = float(occupation_state.smearing_width_ry)
    surface = f * (1.0 - f) / width
    drude, dos = metal_head_surface_tensors(velocity_cart, surface,
        mesh=mesh, nb_logical=nb, cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
        nspinor_wfn=int(meta.nspinor_wfnfile))
    contact = direct_photon_contact(contact_packed, photon_g0_vectors,
        layout=layout, mesh=mesh)
    geometry = CoulombGeometry.from_wfn(wfn)
    kgrid = tuple(int(n) for n in meta.kgrid)
    chunk_size = 2**13
    samples = iter_minibz_photon_samples(get_kernel(3), geometry,
        kgrid, nsamples=2**17,
        qmc_reps=4, chunk_size=chunk_size, analytic_sphere=True)
    fields = ((len(z),4,4), (len(z),4,4), (len(z),4,4),
              (len(z),4,4), (4,4), (4,4,4), (4,4))
    total = [jnp.zeros(shape, dtype=jnp.complex128) for shape in fields]
    replicate = [jnp.zeros(shape, dtype=jnp.complex128) for shape in fields]
    spreads = []
    count = 0
    last_rep = 0
    analytic_bare = None
    max_error = jnp.asarray(0.0)
    replicated = NamedSharding(mesh, P())
    points_device = device_put_process_local(z, replicated)

    def finish_rep():
        nonlocal replicate, count
        if count == 0:
            raise ValueError("GATE photon_direct_head_cubature: empty replicate")
        normalized = [x / count for x in replicate]
        spreads.append(normalized[0])
        for i, value in enumerate(normalized):
            total[i] = total[i] + value
        replicate = [jnp.zeros(shape, dtype=jnp.complex128) for shape in fields]
        count = 0

    for rep, _start, _stop, q, D, valid, weight, analytic_D in samples:
        if analytic_bare is None:
            analytic_bare = float(analytic_D[0, 0])
        if rep != last_rep:
            finish_rep()
            last_rep = rep
        result = _direct_gamma_chunk(
            device_put_process_local(q, replicated),
            device_put_process_local(D, replicated),
            device_put_process_local(weight, replicated),
            tensors[0], tensors[1], tensors[2], points_device,
            drude, dos, contact)
        for i in range(len(fields)):
            replicate[i] = replicate[i] + result[i]
        max_error = jnp.maximum(max_error, result[-1])
        count += int(valid)
    finish_rep()
    if len(spreads) != 4:
        raise ValueError("GATE photon_direct_head_cubature: missing Sobol replicate")
    if analytic_bare is None or analytic_bare <= 0:
        raise ValueError("GATE photon_direct_head_cubature: missing analytic sphere")
    sq, sD, sw = _bulk_sphere_rule(geometry, kgrid, analytic_bare, chunk_size)
    sphere = _direct_gamma_chunk(
        device_put_process_local(sq, replicated),
        device_put_process_local(sD, replicated),
        device_put_process_local(sw, replicated),
        tensors[0], tensors[1], tensors[2], points_device,
        drude, dos, contact)
    max_error = jnp.maximum(max_error, sphere[-1])
    fields_mean = [(np.asarray(total[i]) / len(spreads) + np.asarray(sphere[i]))
                   / float(meta.cell_volume) for i in range(len(fields))]
    spread_rows = np.asarray(jnp.stack(spreads))
    spread = float(np.max(np.abs(spread_rows - np.mean(spread_rows, axis=0)))) / float(meta.cell_volume)
    if not all(np.all(np.isfinite(value)) for value in fields_mean):
        raise ValueError("GATE photon_direct_head_nonfinite: direct Γ cell average is not finite")
    if jax.process_index() == 0:
        print_fn("  direct photon Γ: first-order CC/CT/TC/TT, FD Drude; "
                 "4×131072 Sobol exterior + 8×12×24 screened sphere; "
                 f"max replicate spread={spread:.3e} Ry, "
                 f"Dyson residual={float(max_error):.3e}", flush=True)
    return dict(zip(("Wc", "dWc_ds", "Wc_minus_q", "dWc_minus_q_ds",
                     "constant", "moments", "bare"),
                    fields_mean))
