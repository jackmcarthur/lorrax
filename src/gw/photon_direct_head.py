"""First-order direct bulk photon head from the authenticated dipole vertex.

The six rows are three derivatives of the charge vertex followed by three
uniform current vertices.  Only the final 6 by 6 tensors are replicated;
band pairs remain tiled over both processor axes.  The metallic diagonal
response and the photon contact are separate inputs to the Γ-cell solve.
"""
from functools import lru_cache

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.bispinor_init import HALFALPHA
from gw.qsgw_head import _pad_head_band_manifold, _mesh_xy


def cartesian_gamma_rows(photon_g0_vectors, current_basis_rows):
    """The four Γ head rows as per-block vectors: ``rows[a][i]`` in Lorentz block i.

    The G=0 one-leg vectors are diagonal in the fit basis (channel c is
    component c's ζ at G=0).  The photon operator keeps Cartesian blocks
    (``v_q_bispinor._cartesian_tt_results``), so Cartesian head row a places
    ``Σ_c conj(B_ca) B_ci g^c`` in current block i: then
    ``Σ_ab conj(row_a) M_ab row_b`` equals the fit-basis head with
    ``B M Bᴴ``, for every Cartesian 4×4 head tensor M.  ``B`` None
    (Cartesian fits): row a is ``g^a`` in block a alone.
    """
    g = tuple(photon_g0_vectors)
    zero = [None] * 4
    if current_basis_rows is None:
        return tuple(tuple(v if i == a else zero[i] for i in range(4))
                     for a, v in enumerate(g))
    B = np.asarray(current_basis_rows, dtype=np.complex128)
    rows = [(g[0], None, None, None)]
    for a in range(3):
        rows.append((None,) + tuple(
            sum(complex(np.conj(B[c, a]) * B[c, i]) * g[1 + c]
                for c in range(3) if abs(B[c, a] * B[c, i]) > 0)
            for i in range(3)))
    return tuple(rows)


def packed_gamma_vectors(photon_g0_vectors, layout, mesh, *, current_basis_rows):
    """Use the existing photon-layout owner for the four Γ plane-wave rows.

    ``current_basis_rows`` is ``meta.current_basis_rows``: the rows are
    :func:`cartesian_gamma_rows`, each packed block by block.
    """
    from common.collectives import device_put_process_local
    from gw.photon_layout import pack_photon_channel_vectors

    if photon_g0_vectors is None or len(photon_g0_vectors) != 4:
        raise ValueError("direct photon Γ needs four authenticated G=0 vectors")
    blocks = cartesian_gamma_rows(photon_g0_vectors, current_basis_rows)
    sh_y = NamedSharding(mesh, P(None, "y"))

    def pack(axis_name, place):
        out = []
        for a, row in enumerate(blocks):
            filled = tuple(place(v) if v is not None else
                           jnp.zeros_like(place(photon_g0_vectors[i]))
                           for i, v in enumerate(row))
            out.append(pack_photon_channel_vectors(
                filled, layout, mesh, axis_name=axis_name)[0].sum(axis=0))
        return jnp.stack(out)

    if current_basis_rows is None:
        x = pack_photon_channel_vectors(tuple(photon_g0_vectors), layout,
                                        mesh, axis_name="x")[0]
        y = pack_photon_channel_vectors(tuple(
            device_put_process_local(row, sh_y) for row in photon_g0_vectors),
            layout, mesh, axis_name="y")[0]
        return x, y
    x = pack("x", lambda v: v)
    y = pack("y", lambda v: device_put_process_local(v, sh_y))
    return x, y


def subtract_bare_tt_from_bank(packed_v, photon_g0_vectors, *, layout,
                                mesh, wfn, meta):
    """Keep the direct Γ bare TT exchange in V, outside the body W Dyson."""
    from vcoul import CoulombGeometry
    from gw.photon_layout import add_photon_q0_low_rank
    from gw.v_q_bispinor import _tt_head_tensor

    x, y = packed_gamma_vectors(photon_g0_vectors, layout, mesh,
                                current_basis_rows=meta.current_basis_rows)
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


def direct_photon_contact(contact_packed, photon_g0_vectors, *, layout, mesh,
                          current_basis_rows):
    """Project the bank's one FD contact onto the four uniform vertices."""
    x, y = packed_gamma_vectors(photon_g0_vectors, layout, mesh,
                                current_basis_rows=current_basis_rows)
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


def hall_projection(tensor):
    """Block-wise Cartesian antisymmetric part of ``(..., 6, 6)`` jet/current tensors.

    Rows 0:3 are charge jets and 3:6 currents.  Each 3x3 block ``B`` keeps
    ``(B - B^T)/2``; the jet-jet block is set to zero because
    ``q.A.q = 0`` for antisymmetric ``A`` and its entries carry the pair's
    ``1/Delta`` jets (``docs/theory/metal-q0-head.md`` section 2).
    """
    shape = tensor.shape
    blocks = tensor.reshape(shape[:-2] + (2, 3, 2, 3))
    swapped = jnp.swapaxes(blocks, -3, -1)
    hall = (0.5 * (blocks - swapped)).reshape(shape)
    return hall.at[..., :3, :3].set(0.0)


@lru_cache(maxsize=16)
def _interband_program(mesh: Mesh, nb_logical: int):
    from jax.sharding import PartitionSpec as P
    from gw.degen_average import TOL_DEGENERACY_RY
    from gw.fermi_surface import intraband_pair_fraction

    ax_x, ax_y = _mesh_xy(mesh)

    def local(v, e_x, e_y, f_x, f_y, frequencies, scale, d_x, d_y, w_x, w_y,
              moment):
        nx, ny = v.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        delta = e_y[:, None, :] - e_x[:, :, None]
        occupation = f_x[:, :, None] - f_y[:, None, :]
        active = ((ix[:, None] < nb_logical) &
                  (iy[None, :] < nb_logical))[None]
        offdiag = active & (ix[None, :, None] != iy[None, None, :])
        # One pair split with the scalar head: each pair keeps 1 - phi of its
        # first-order tensor, and its Fermi-surface share phi keeps only the
        # Hall (antisymmetric) part here; exact multiplets leave entirely.
        share = intraband_pair_fraction(v, delta, d_x, d_y, w_x, w_y, moment,
                                        TOL_DEGENERACY_RY)
        exact = jnp.abs(delta) < TOL_DEGENERACY_RY
        jet = offdiag & ~exact
        regular = jnp.where(jet, 1.0 - share, 0.0)
        hall_pairs = jnp.where(jet, share, 0.0)
        inverse = jnp.where(jet, 1 / jnp.where(jet, delta, 1.0), 0.0)
        # In Ry and bohr units, ∂_k H has the dipole-file velocity.  The
        # packed Breit current uses Gamma_raw=(alpha_FS/2)*v_Ry.
        vertex = jnp.concatenate((v * inverse[None], HALFALPHA * v), axis=0)
        vertex = jnp.where(offdiag[None], vertex, 0.0)

        def contract(weight):
            value = jnp.einsum("akij,kij,bkij->ab", jnp.conj(vertex),
                                weight, vertex, optimize=True)
            return jax.lax.psum(value, (ax_x, ax_y))

        def both(pairs_weight):
            """Interband tensor plus the Hall part of the intraband pairs."""
            return (contract(regular * pairs_weight)
                    + hall_projection(contract(hall_pairs * pairs_weight)))

        # One orientation per directed pair.  scale is half the incumbent
        # energy-ordered S-tensor prefactor because both directions appear.
        def one(_, z):
            denom = z - delta
            safe_denom = jnp.where(jet, denom, 1.0 + 0.0j)
            weight = scale * occupation / safe_denom
            safe_z = jnp.where(z == 0, 1.0 + 0.0j, z)
            slope = jnp.where(z != 0,
                -scale * occupation /
                (2 * safe_z * safe_denom * safe_denom), 0.0)
            return None, (both(weight), both(slope))

        _, (value, slope) = jax.lax.scan(one, None, frequencies, unroll=1)
        moments = jnp.stack([both(scale * occupation * delta**power)
                             for power in range(4)])
        return value, slope, moments

    return jax.jit(shard_map(local, mesh=mesh,
        in_specs=(P(None,None,"x","y"), P(None,"x"), P(None,"y"),
                  P(None,"x"), P(None,"y"), P(None), P(),
                  P(None,None,"x"), P(None,None,"y"), P(None,"x"), P(None,"y"),
                  P(None,None)),
        out_specs=(P(None,None,None), P(None,None,None),
                   P(None,None,None)), check_vma=False))


def direct_photon_interband_tensors(velocity_cart, energies_kn_ry,
                                    occupations_kn, frequencies_ry, *,
                                    surface_kn, pair_split,
                                    mesh: Mesh, nb_logical: int,
                                    cell_volume: float, nk_tot: int,
                                    nspin: int, nspinor_wfn: int):
    """Return ordered 6×6 value, d/d(z²), and 1/z..1/z⁴ coefficients.

    Rows 0:3 multiply the mini-BZ Cartesian q; rows 3:6 are current.  Pairs
    are split once with the scalar head (``pair_split``,
    ``gw.qsgw_head.metal_pair_split``; ``surface_kn`` is the tetrahedron
    table): each pair enters with ``1 - phi`` and its Fermi-surface share
    ``phi`` only through :func:`hall_projection`; exact multiplets enter not
    at all (their content is the Fermi-surface atoms').
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    w = jnp.asarray(surface_kn, dtype=jnp.float64)
    z = jnp.atleast_1d(jnp.asarray(frequencies_ry, dtype=jnp.complex128))
    if (v.ndim != 4 or tuple(v.shape[:2]) != (3, nk_tot)
            or v.shape[-1] != v.shape[-2] or e.shape != f.shape
            or tuple(e.shape) != (nk_tot, nb_logical) or w.shape != e.shape
            or nb_logical > v.shape[-1]):
        raise ValueError("direct photon head requires aligned 3×Nk×Nb×Nb dipoles and Nk×Nb states")
    if not (cell_volume > 0 and nk_tot > 0 and nspin > 0 and nspinor_wfn > 0):
        raise ValueError("direct photon head requires positive normalization")
    points = np.asarray(frequencies_ry, dtype=np.complex128)
    if np.any((np.imag(points) < 0) |
              ((np.imag(points) == 0) & (np.real(points) != 0))):
        raise ValueError("direct photon head samples must be causal or exactly static")
    if v.shape[-1] > nb_logical:
        pad = int(v.shape[-1]) - int(nb_logical)
        e = jnp.pad(e, ((0, 0), (0, pad)))
        f = jnp.pad(f, ((0, 0), (0, pad)))
        w = jnp.pad(w, ((0, 0), (0, pad)))
    v, e, f, w = _pad_head_band_manifold(v, e, f, w, mesh=mesh)
    d_x, d_y, moment = pair_split.operands(v)
    # Match the incumbent S_ab normalization for its charge-charge block.
    scale = 2.0 / (cell_volume * nk_tot * nspin * nspinor_wfn)
    return _interband_program(mesh, int(nb_logical))(
        v, e, e, f, f, z, jnp.asarray(scale, dtype=jnp.complex128),
        d_x, d_y, w, w, moment)


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


def metal_intraband_photon_response(q_cart, z, drude_tensor, static_dos,
                                     atom_w, atom_u):
    """Intraband 4×4 response and its z² derivative at one frequency.

    At ``z != 0`` the anisotropic Fermi-surface Lindhard function of the
    velocity atoms (``gw.fermi_surface.four_current_intraband_response``):
    CC, CT = (alpha/2) sum W g u, TT = (alpha/2)^2 sum W g u u with
    ``g = q.u/(z - q.u)``; Drude ``q.D.q/z^2`` for ``|z| >> q u``.  At
    ``z = 0`` the static limit: Thomas–Fermi CC=-N0, TT=-(alpha/2)^2 D.
    No chemical-potential damping or artificial insulating gap enters.
    """
    from gw.fermi_surface import four_current_intraband_response

    q = jnp.asarray(q_cart, dtype=jnp.float64)
    D = jnp.asarray(drude_tensor, dtype=jnp.complex128)
    static = jnp.zeros((q.shape[0], 4, 4), dtype=jnp.complex128)
    static = static.at[:, 0, 0].set(-static_dos)
    static = static.at[:, 1:, 1:].set(-HALFALPHA**2 * D)
    value, derivative = four_current_intraband_response(
        q, z, atom_w, atom_u, HALFALPHA)
    return (jnp.where(z == 0, static, value),
            jnp.where(z == 0, 0.0, derivative))


@jax.jit
def _direct_gamma_chunk(q, bare, weight, interband, slopes, coefficients,
                        frequencies, drude, dos, contact, atom_w, atom_u):
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
        intra, dintra = metal_intraband_photon_response(
            q, z, drude, dos, atom_w, atom_u)
        pi = project_first_order_photon_response(tensor, q) + intra
        dpi = project_first_order_photon_response(derivative, q) + dintra
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


#: Sobol samples per direct-Γ replicate (four replicates; the sphere rule is one more call).
_GAMMA_SAMPLES = 2**17
#: Smallest samples per rank per call: below this the call is launch-bound.
_GAMMA_MIN_LOCAL = 2**10
#: Share of the stage room the chunk's compiled footprint may take.
_GAMMA_ROOM_FRACTION = 0.5
#: Points of the screened sphere rule (``_bulk_sphere_rule``: 8 radial x 12 polar x 24 azimuthal).
_GAMMA_SPHERE_POINTS = 8 * 12 * 24
#: The exterior Sobol stream by (cell, k grid, chunk).  It is seeded geometry,
#: so a hit is the stream a redraw makes, bit for bit; a metallic bispinor
#: bank asks for the same stream at every SC map (0.4-0.6 s of host draw and
#: bare-kernel work per map on Fe 4^3).  One entry, host float64:
#: 4 x 2^17 x (3 + 16 + 1) x 8 B = 84 MB per process.
_GAMMA_STREAM: dict = {}


def _gamma_sample_stream(geometry, kgrid, chunk_size):
    """The four-replicate exterior stream of ``iter_minibz_photon_samples``, held."""
    from vcoul import get_kernel, iter_minibz_photon_samples

    key = (np.asarray(geometry.bvec, dtype=np.float64).tobytes(),
           float(geometry.cell_volume), tuple(int(n) for n in kgrid),
           int(chunk_size))
    held = _GAMMA_STREAM.get(key)
    if held is None:
        held = tuple(iter_minibz_photon_samples(get_kernel(3), geometry,
            kgrid, nsamples=_GAMMA_SAMPLES,
            qmc_reps=4, chunk_size=chunk_size, analytic_sphere=True))
        _GAMMA_STREAM.clear()
        _GAMMA_STREAM[key] = held
    return held


def direct_gamma_chunk_plan(mesh, operands, *, nsamples):
    """(chunk, sharding) of the direct-Γ cubature calls.

    The samples of one call are split over every rank (``P(('x', 'y'))``: the
    q sums in :func:`_direct_gamma_chunk` become one small all-reduce), so no
    rank repeats another's samples.  The chunk starts at the floor
    (``_GAMMA_MIN_LOCAL`` samples per rank, and at least the sphere rule's
    points) and doubles while it stays within ``nsamples`` (one call per
    replicate) and its compiled footprint, priced at the floor and scaled per
    sample, fits ``_GAMMA_ROOM_FRACTION`` of the stage room
    (``common.gpu_utils.device_room_bytes``).  Every process enters.
    """
    from common.gpu_utils import device_budget_bytes, device_room_bytes, record_stage_price
    n_dev = int(mesh.devices.size)
    split = NamedSharding(mesh, P(tuple(mesh.axis_names)))
    probe = n_dev * _GAMMA_MIN_LOCAL
    sds = jax.ShapeDtypeStruct
    compiled = _direct_gamma_chunk.lower(
        sds((probe, 3), jnp.float64, sharding=split),
        sds((probe, 4, 4), jnp.float64, sharding=split),
        sds((probe,), jnp.float64, sharding=split), *operands).compile()
    memory = compiled.memory_analysis()
    # Per global sample, per rank: the compiled temporaries plus the sample's own
    # q, D and weight rows (the fixed operands and the 4x4 outputs do not scale).
    per_sample = max(float(getattr(memory, "temp_size_in_bytes", 0)) / probe, 0.0) \
        + 8.0 * (3 + 16 + 1) / n_dev
    live_room = float(device_room_bytes())
    room = _GAMMA_ROOM_FRACTION * live_room
    floor = max(probe, _GAMMA_SPHERE_POINTS)
    chunk = floor
    while chunk * 2 <= int(nsamples) and chunk * 2 * per_sample <= room:
        chunk *= 2
    chunk = min(max(chunk, floor), max(int(nsamples), floor))
    record_stage_price(f"direct Gamma head, chunk {chunk} over {n_dev} ranks",
                       device_budget_bytes() - live_room + chunk * per_sample)
    return int(chunk), split


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
    from vcoul import CoulombGeometry
    from common.collectives import device_put_process_local

    if int(meta.sys_dim) != 3:
        raise ValueError("GATE photon_direct_head_bulk: first-order direct Γ requires sys_dim=3")
    if occupation_state is None or occupation_state.smearing_family != "fd":
        raise ValueError("GATE photon_direct_head_fd: metallic direct Γ requires current-map Fermi-Dirac occupations")
    nb = int(meta.b_id_4_chi_user) - int(meta.b_id_0)
    if velocity_cart.shape[-1] < nb or wfns.enk.shape[1] < nb:
        raise ValueError("GATE photon_direct_head_bands: dipole and response manifolds differ")
    z = np.asarray(frequencies_ry, dtype=np.complex128).reshape(-1)
    e = np.asarray(wfns.enk[:, :nb], dtype=np.float64)
    f = np.asarray(occupation_state.f_kn[:, :nb], dtype=np.float64)
    geometry = CoulombGeometry.from_wfn(wfn)
    kgrid = tuple(int(n) for n in meta.kgrid)
    # One Fermi-surface owner with the scalar head: the tetrahedron table,
    # its multiplet rule, the Taylor-radius pair split and the velocity atoms
    # (docs/theory/metal-q0-head.md).
    from gw.fermi_surface import metal_head_surface_weights
    from gw.qsgw_head import metal_intraband_model
    surface = metal_head_surface_weights(
        e, float(occupation_state.mu_ry), sym=wfn.symmetry(), kgrid=wfn.kgrid,
        bvec_cart=geometry.bvec)
    stored = int(velocity_cart.shape[-1])
    pad = ((0, 0), (0, stored - nb))
    drude, atoms, split = metal_intraband_model(
        velocity_cart, np.pad(surface, pad), np.pad(e, pad), mesh=mesh,
        nb_logical=nb, cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
        nspinor=int(meta.nspinor_wfnfile), bvec_cart=geometry.bvec, kgrid=kgrid)
    dos = atoms.dos
    tensors = direct_photon_interband_tensors(velocity_cart, e, f, z,
        surface_kn=surface, pair_split=split,
        mesh=mesh, nb_logical=nb, cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
        nspinor_wfn=int(meta.nspinor_wfnfile))
    contact = direct_photon_contact(contact_packed, photon_g0_vectors,
        layout=layout, mesh=mesh, current_basis_rows=meta.current_basis_rows)
    replicated = NamedSharding(mesh, P())
    points_device = device_put_process_local(z, replicated)
    atom_w, atom_u = (device_put_process_local(np.asarray(x), replicated)
                      for x in atoms.device_operands())
    operands = (tensors[0], tensors[1], tensors[2], points_device,
                jnp.asarray(drude), jnp.asarray(dos, jnp.float64), contact,
                atom_w, atom_u)
    if jax.process_index() == 0:
        from gw.qsgw_head import drude_report
        print("  direct photon Γ metal head: tetrahedron Fermi surface, "
              + drude_report(atoms)
              + f"; kappa_TF^2 = {8.0 * np.pi * dos:.6f} bohr^-2; "
              + atoms.describe(), file=sys.stderr, flush=True)
    chunk_size, split = direct_gamma_chunk_plan(mesh, operands, nsamples=_GAMMA_SAMPLES)
    samples = _gamma_sample_stream(geometry, kgrid, chunk_size)
    fields = ((len(z),4,4), (len(z),4,4), (len(z),4,4),
              (len(z),4,4), (4,4), (4,4,4), (4,4))
    total = [jnp.zeros(shape, dtype=jnp.complex128) for shape in fields]
    replicate = [jnp.zeros(shape, dtype=jnp.complex128) for shape in fields]
    spreads = []
    count = 0
    last_rep = 0
    analytic_bare = None
    max_error = jnp.asarray(0.0)

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
            device_put_process_local(q, split),
            device_put_process_local(D, split),
            device_put_process_local(weight, split), *operands)
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
        device_put_process_local(sq, split),
        device_put_process_local(sD, split),
        device_put_process_local(sw, split), *operands)
    max_error = jnp.maximum(max_error, sphere[-1])
    fields_mean = [(np.asarray(total[i]) / len(spreads) + np.asarray(sphere[i]))
                   / float(meta.cell_volume) for i in range(len(fields))]
    spread_rows = np.asarray(jnp.stack(spreads))
    spread = float(np.max(np.abs(spread_rows - np.mean(spread_rows, axis=0)))) / float(meta.cell_volume)
    if not all(np.all(np.isfinite(value)) for value in fields_mean):
        raise ValueError("GATE photon_direct_head_nonfinite: direct Γ cell average is not finite")
    if jax.process_index() == 0:
        print_fn("  direct photon Γ: first-order CC/CT/TC/TT + Fermi-surface Lindhard cell; "
                 "4×131072 Sobol exterior + 8×12×24 screened sphere; "
                 f"max replicate spread={spread:.3e} Ry, "
                 f"Dyson residual={float(max_error):.3e}", flush=True)
        origin = int(np.argmin(np.abs(z)))
        print(f"  direct photon Γ origin: z={complex(z[origin]):.6g} Ry, "
                 f"<W_h - W_inf>_CC/Omega={complex(fields_mean[0][origin][0, 0]):.9g}, "
                 f"TT trace={complex(np.trace(fields_mean[0][origin][1:, 1:])):.6g}", file=sys.stderr, flush=True)
    return dict(zip(("Wc", "dWc_ds", "Wc_minus_q", "dWc_minus_q_ds",
                     "constant", "moments", "bare"),
                    fields_mean))
