"""Full-density snapshot bubble and bounded joint Keldysh screening.

The density-space approximation is explicit: ``F[mu,r]`` is a Coulomb
factor, ``v ~= F F.H``.  The bubble is projected as ``F.H P F``, so its
Dyson equation has bare interaction ``I``.  Physical correlation/noise
operators are ``F WcR F.H`` and ``F Wless F.H``.  Electronic bands are
never truncated by this projection.  Frozen bands enter through their
diagonal occupations in the same full density matrix as the active block.

The finite-lag model uses finite complex supports, with a common full-block Loewner
realization of the stacked one-sided ``[P<; P>]``.  No frequency integral,
passive residue constraint, or dense all-transition state space is used.
The fit is admitted only after independent full-matrix sample checks.
The separate lossless transition-bath model explicitly pinches the screening
density and uses signed Hermitian Galerkin/Krylov response reduction with
same-subspace statistical weights and algebraically paired q/−q banks.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from . import keldysh_rational as kr


def mirror(system):
    """Return ``-conj(H(-conj(z)))``, the one-sided bosonic crossing map.

    For a physical q/−q pair in conjugate Coulomb factor gauges,
    ``Pgreater_minus=mirror(Plesser_q)`` and conversely.  Unlike sharp,
    this map does not transpose input/output axes.  Its poles stay in the
    same half-plane.  It follows directly from
    ``Xgreater_q(t)=-conj(Xlesser_minus(t))`` for positive lag.
    """
    A, B, C, D = (np.asarray(x, dtype=complex) for x in system)
    return -A.conj(), B.conj(), C.conj(), -D.conj()


def projected_coulomb_reciprocity(gram_q, gram_minus_q, cross, *, tolerance=1e-7):
    """Certify projected bare reciprocity from reduced Gram contractions.

    Given factors ``Fq[mu,r]`` and ``Fm[mu,r]``, inputs are the small
    replicated reductions ``gram_q=Fq.H Fq``, ``gram_minus_q=Fm.H Fm`` and
    ``cross=Fm.H conj(Fq)``.  The identity for the Frobenius distance of
    two Gram factorizations measures ``Fm Fm.H - conj(Fq Fq.H)`` without
    forming a centroid-square matrix or requiring matching factor gauges.
    The q/−q rule is a bare-Coulomb identity even for magnetic states; it
    does not impose equilibrium reciprocity on nonequilibrium W.
    """
    gq, gm, cross = (np.asarray(x, dtype=complex) for x in (gram_q, gram_minus_q, cross))
    if (gq.ndim != 2 or gm.ndim != 2 or gq.shape[0] != gq.shape[1]
            or gm.shape[0] != gm.shape[1] or cross.shape != (gm.shape[0], gq.shape[0])
            or not all(np.isfinite(x).all() for x in (gq, gm, cross)) or tolerance <= 0):
        raise ValueError("GATE snapshot_coulomb_reciprocity: invalid factor Gram contractions")
    norm_q2, norm_m2 = np.linalg.norm(gq) ** 2, np.linalg.norm(gm) ** 2
    distance2 = float(norm_q2 + norm_m2 - 2.0 * np.linalg.norm(cross) ** 2)
    scale2 = max(float(norm_q2), float(norm_m2), 1e-300)
    if distance2 < -1e-12 * scale2:
        raise ValueError("GATE snapshot_coulomb_reciprocity: inconsistent factor Gram contractions")
    error = float(np.sqrt(max(distance2, 0.0) / scale2))
    if error > tolerance:
        raise ValueError(f"GATE snapshot_coulomb_reciprocity: paired projected bare error "
                         f"{error:.3e} exceeds {tolerance:.3e}")
    return {"projected_bare_reciprocity_error": error, "tolerance": float(tolerance)}


def embed_active_density(occupations, active_rho, start):
    """Embed ``[k,a,a]`` active rho in frozen ``[k,band]`` occupations.

    Arrays are small replicated host band tables.  No off-diagonal active
    entry is discarded, and the frozen/active cross transitions remain live.
    """
    occupations = np.asarray(occupations, dtype=float)
    active_rho = np.asarray(active_rho, dtype=np.complex128)
    if occupations.ndim != 2 or active_rho.ndim != 3:
        raise ValueError("GATE snapshot_density_shape: expected [k,b] and [k,a,a]")
    nk, nb = occupations.shape
    na = active_rho.shape[-1]
    if active_rho.shape != (nk, na, na) or start < 0 or start + na > nb:
        raise ValueError("GATE snapshot_density_shape: active block outside band support")
    rho = np.zeros((nk, nb, nb), dtype=np.complex128)
    index = np.arange(nb)
    rho[:, index, index] = occupations
    rho[:, start:start + na, start:start + na] = active_rho
    if not np.isfinite(rho).all() or np.max(abs(rho - rho.conj().transpose(0, 2, 1))) > 1e-10:
        raise ValueError("GATE snapshot_density_hermitian: rho must be finite Hermitian")
    return rho


def pair_vertices(psi_k, psi_kq, rho_k, rho_kq, factor, b, d, *, xp=np):
    """Project ordered bubble vertices, retaining the MGKBA matrix ordering.

    ``psi`` is ``[band,spin,mu]``, rho ``[band,band]`` and the Coulomb
    factor ``[mu,r]``.  ``b,d`` select the energy columns on the two legs.
    Returned ``(less,greater,right)`` have shape ``[r,b,d]``.  ``xp`` may
    be NumPy for a tiny oracle or JAX inside a distributed caller.  The
    latter must reduce the mu contraction over its named mesh axis.
    """
    left_h = xp.einsum("asm,ab->bsm", psi_k, rho_k)
    right_h = xp.einsum("asm,ab->bsm", psi_kq, rho_kq)
    left_p, right_p = psi_k - left_h, psi_kq - right_h

    def project(left, right):
        return xp.einsum("mr,bsm,dsm->rbd", xp.conj(factor), left[b],
                         xp.conj(right[d]), optimize=True)

    return (project(left_h, right_p), project(left_p, right_h),
            project(psi_k, psi_kq))


def polarization_samples(psi, energy, rho, factor, kmq, sites, *,
                         adiabatic_ry, spin_weight=1.0, band_block=32):
    """Evaluate exact projected ``P</>(z)`` by blocked ordered band pairs.

    This host implementation is for small arrays; the distributed factory
    below computes the same equation without moving centroid faces to host.
    Inputs have shapes ``psi[k,b,s,mu]``, ``energy[k,b]`` (Ry),
    ``rho[k,b,b]``, ``factor[mu,r]``, ``kmq[k]`` and ``sites[support]``
    (Ry).  Output ``[2,support,r,r]`` is lesser then greater.  The declared
    positive finite lag decay shifts transition poles by ``-i gamma``;
    this is not a validated physical adiabatic statistical closure.
    It is independent of support heights and the collision evaluation eta.
    Normalization is ``spin_weight / nk``; no cell-volume factor is added.
    """
    psi, energy, rho, factor = map(np.asarray, (psi, energy, rho, factor))
    sites = np.asarray(sites, dtype=complex)
    nk, nb = energy.shape
    if adiabatic_ry <= 0 or band_block < 1:
        raise ValueError("GATE snapshot_regulator: declare positive finite lag decay adiabatic_ry and band_block")
    if psi.shape[:2] != (nk, nb) or rho.shape != (nk, nb, nb):
        raise ValueError("GATE snapshot_band_support: inconsistent full-band arrays")
    kmq = np.asarray(kmq, dtype=int)
    if kmq.shape != (nk,) or not np.array_equal(np.sort(kmq), np.arange(nk)):
        raise ValueError("GATE snapshot_k_partner: k-q must be an exact full-grid permutation")
    rank = factor.shape[1]
    result = np.zeros((2, sites.size, rank, rank), dtype=complex)
    for k in range(nk):
        partner = kmq[k]
        for blo in range(0, nb, band_block):
            b = slice(blo, min(nb, blo + band_block))
            for dlo in range(0, nb, band_block):
                d = slice(dlo, min(nb, dlo + band_block))
                less, greater, right = pair_vertices(
                    psi[k], psi[partner], rho[k], rho[partner], factor, b, d)
                delta = energy[k, b, None] - energy[partner, None, d]
                resolvent = 1.0 / (sites[:, None, None] - delta + 1j * adiabatic_ry)
                for channel, left in enumerate((less, greater)):
                    result[channel] += np.einsum(
                        "sbd,rbd,tbd->srt", resolvent, left, right.conj(), optimize=True)
    return result * (float(spin_weight) / nk)


def polarization_from_vertices(vertices, energy, rho, kmq, sites, *,
                               adiabatic_ry, spin_weight=1.0, band_block=32):
    """Sample the full-rho bubble from preprojected charge-pair vertices.

    ``vertices[k,a,c,r] = sum_mu,s conj(F[mu,r]) psi[k,a,s,mu]
    conj(psi[k-q,c,s,mu])``.  Only this reduced array is a host operand;
    large centroid faces stay distributed at its producer.  Energy and
    density are replicated ``[k,b]`` and ``[k,b,b]`` tables.  All bands,
    including frozen ones, must be present on both vertex axes.  The
    output and regulator convention match ``polarization_samples``.
    """
    vertices, energy, rho = map(np.asarray, (vertices, energy, rho))
    sites = np.asarray(sites, dtype=complex)
    nk, nb = energy.shape
    kmq = np.asarray(kmq, dtype=int)
    if (vertices.ndim != 4 or vertices.shape[:3] != (nk, nb, nb)
            or rho.shape != (nk, nb, nb) or kmq.shape != (nk,)
            or not np.array_equal(np.sort(kmq), np.arange(nk))):
        raise ValueError("GATE snapshot_projected_vertices: incompatible band/partner tables")
    if adiabatic_ry <= 0 or band_block < 1:
        raise ValueError("GATE snapshot_regulator: declare positive regulator and block")
    rank = vertices.shape[-1]
    result = np.zeros((2, sites.size, rank, rank), dtype=complex)
    identity = np.eye(nb)
    for k in range(nk):
        partner = kmq[k]
        h, hq = rho[k], rho[partner]
        # rho_ab is on the energy-column b (the ordered MGKBA line).
        # Conjugating the second orbital line conjugates its coefficient.
        weighted = (
            np.einsum("ab,cd,acr->bdr", h, (identity - hq).conj(), vertices[k], optimize=True),
            np.einsum("ab,cd,acr->bdr", identity - h, hq.conj(), vertices[k], optimize=True))
        for blo in range(0, nb, band_block):
            b = slice(blo, min(nb, blo + band_block))
            for dlo in range(0, nb, band_block):
                d = slice(dlo, min(nb, dlo + band_block))
                denominator = 1.0 / (sites[:, None, None] - energy[k, b, None]
                                      + energy[partner, None, d] + 1j * adiabatic_ry)
                right = vertices[k, b, d].conj()
                for channel, left in enumerate(weighted):
                    result[channel] += np.einsum("sbd,bdr,bdt->srt", denominator,
                                                   left[b, d], right, optimize=True)
    return result * (float(spin_weight) / nk)


def make_polarization_sampler(mesh_xy, *, band_block=32):
    """Build the sharded blocked bubble sampler, using no centroid-square tile.

    The returned callable takes ``(psi, energy, rho, factor, kmq, sites,
    adiabatic_ry, spin_weight)`` with the shapes of ``polarization_samples``.
    psi and factor are faces at ``P(None,None,None,'x')`` and ``P('x',None)``;
    small band tables and ``[2,support,r,r]`` outputs are replicated.  Bands
    must be padded to ``band_block`` with zero psi (density pad = zero).
    Every loop in the jit is a scan inside shard_map.  Only the projected
    vertices are reduced; no whole centroid matrix is formed or gathered.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    block = int(band_block)
    if block < 1:
        raise ValueError("band_block must be positive")

    def body(psi, energy, rho, factor, kmq, sites, gamma, spin_weight):
        nk, nb = energy.shape
        if nb % block:
            raise ValueError("GATE snapshot_band_padding: band support must be block padded")
        holes = jnp.einsum("kasm,kab->kbsm", psi, rho)
        particles = psi - holes
        rank = factor.shape[1]
        zero = jnp.zeros((2, sites.size, rank, rank), dtype=psi.dtype)
        nblocks = nb // block

        def step(carry, index):
            k = index // (nblocks * nblocks)
            ib = (index // nblocks) % nblocks
            jd = index % nblocks
            partner = kmq[k]

            def rows(array, row, col):
                return jax.lax.dynamic_slice(array, (row, col * block, 0, 0),
                                             (1, block, array.shape[2], array.shape[3]))[0]

            def project(left, right):
                local = jnp.einsum("mr,bsm,dsm->rbd", factor.conj(), left,
                                    right.conj(), optimize=True)
                return jax.lax.psum(local, "x")

            right = project(rows(psi, k, ib), rows(psi, partner, jd))
            less = project(rows(holes, k, ib), rows(particles, partner, jd))
            greater = project(rows(particles, k, ib), rows(holes, partner, jd))
            eb = jax.lax.dynamic_slice(energy, (k, ib * block), (1, block))[0]
            ed = jax.lax.dynamic_slice(energy, (partner, jd * block), (1, block))[0]
            weights = 1.0 / (sites[:, None, None] - eb[None, :, None]
                             + ed[None, None, :] + 1j * gamma)
            value = jnp.einsum("sbd,crbd,tbd->csrt", weights,
                                jnp.stack((less, greater)), right.conj(), optimize=True)
            return carry + value, None

        result, _ = jax.lax.scan(step, zero, jnp.arange(nk * nblocks * nblocks))
        return result * (spin_weight / nk)

    return jax.jit(jax.shard_map(
        body, mesh=mesh_xy,
        in_specs=(P(None, None, None, "x"), P(), P(), P("x", None), P(), P(), P(), P()),
        out_specs=P(), check_vma=False))


def make_projected_sampler(mesh_xy, *, band_block=32):
    """Build a GPU-capable bubble sampler for one q's projected vertices.

    The callable takes ``(T, energy, rho, kmq, sites, gamma, spin_weight)``.
    ``T[k,b,d,r]`` is distributed at ``P(('x','y'),None,None,None)``;
    its k carrier may be zero-padded to the mesh size.  Small band tables
    ``energy[k,b]`` (Ry), ``rho[k,b,b]``, exact ``kmq[k]`` and supports
    ``sites[s]`` (Ry) are replicated.  The returned ``[2,s,r,r]`` is
    replicated.  ``gamma>0`` is the declared finite lag decay,
    independent of both support height and collision eta.

    Local k and band-pair loops are scans inside shard_map.  The full-rho
    vertices use three band GEMMs per k, followed by blocked residue/support
    GEMMs.  There is only one collective, on the final small response table;
    no per-transition or per-block collective is performed.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    block = int(band_block)
    if block < 1:
        raise ValueError("band_block must be positive")
    axes = ("x", "y")

    def body(T, energy, rho, kmq, sites, gamma, spin_weight):
        nk, nb = energy.shape
        rank = T.shape[-1]
        if T.shape[1:3] != (nb, nb) or rho.shape != (nk, nb, nb) or kmq.shape != (nk,):
            raise ValueError("GATE snapshot_projected_vertices: inconsistent projected band support")
        nblocks = (nb + block - 1) // block
        padded = nblocks * block
        zero = jnp.zeros((2, sites.size, rank, rank), dtype=T.dtype)
        offset = jax.lax.axis_index(axes) * T.shape[0]

        def k_step(carry, local_k):
            k = offset + local_k

            def accumulate(carry):
                partner = kmq[k]
                vertex = T[local_k]
                # H^T T Q* and Q^T T H* share the H^T T H* product.
                left_h = jnp.einsum("ab,acr->bcr", rho[k], vertex, optimize=True)
                right_h = jnp.einsum("cd,acr->adr", rho[partner].conj(), vertex, optimize=True)
                both = jnp.einsum("ab,adr->bdr", rho[k], right_h, optimize=True)
                weighted = jnp.stack((left_h - both, right_h - both))
                weighted = jnp.pad(weighted, ((0, 0), (0, padded - nb), (0, padded - nb), (0, 0)))
                vertex = jnp.pad(vertex.conj(), ((0, padded - nb), (0, padded - nb), (0, 0)))
                eb = jnp.pad(energy[k], ((0, padded - nb),))
                ed = jnp.pad(energy[partner], ((0, padded - nb),))

                def pair_step(acc, index):
                    b, d = (index // nblocks) * block, (index % nblocks) * block
                    left = jax.lax.dynamic_slice(weighted, (0, b, d, 0), (2, block, block, rank))
                    right = jax.lax.dynamic_slice(vertex, (b, d, 0), (block, block, rank))
                    delta = (jax.lax.dynamic_slice(eb, (b,), (block,))[:, None]
                             - jax.lax.dynamic_slice(ed, (d,), (block,))[None])
                    weights = 1.0 / (sites[:, None, None] - delta + 1j * gamma)
                    residue = jnp.einsum("cbdr,bdt->cbdrt", left, right)
                    values = jnp.matmul(weights.reshape(sites.size, block * block),
                        residue.reshape(2, block * block, rank * rank))
                    return acc + values.reshape(2, sites.size, rank, rank), None

                return jax.lax.scan(pair_step, carry, jnp.arange(nblocks * nblocks))[0]

            carry = jax.lax.cond(k < nk, accumulate, lambda x: x, carry)
            return carry, None

        local = jax.lax.scan(k_step, zero, jnp.arange(T.shape[0]))[0]
        return jax.lax.psum(local, axes) * (spin_weight / nk)

    return jax.jit(jax.shard_map(body, mesh=mesh_xy,
        in_specs=(P(axes, None, None, None), P(), P(), P(), P(), P(), P()),
        out_specs=P(), check_vma=False))


def joint_loewner(sites, samples, *, max_order=96, rank_tolerance=1e-11,
                  max_pencil_bytes=128 * 1024 ** 2):
    """Fit a common strictly proper realization to stacked ``[P<;P>]``.

    ``samples[2,support,r,r]`` and complex ``sites[support]`` are small
    replicated host arrays; energies use Ry.  Real boundary supports are
    allowed when the declared finite lag decay makes their exact samples
    finite.  All matrix entries enter the
    full-block Loewner pencil.  Its SVD selects at most ``max_order`` modal
    candidates; the strictly lower-half-plane candidate subset defines a
    stable approximation basis.  Joint output amplitudes are then fitted
    against ALL support entries by linear least squares with common right
    directions. No pole is shifted, no residue sign is constrained, and
    noise-only modes remain eligible.  This is a new stable basis fit, not
    causal projection of an already accepted transfer.  Independent exact
    response and dressed-noise checks remain mandatory in screen_snapshot.

    ``max_pencil_bytes`` bounds each complex block pencil before allocation;
    factorization workspace and copies require approximately eight such
    arrays.  State count is independent of the electronic transition count.
    """
    sites = np.asarray(sites, dtype=complex)
    samples = np.asarray(samples, dtype=complex)
    if (samples.ndim != 4 or samples.shape[0] != 2
            or samples.shape[1] != sites.size or samples.shape[2] != samples.shape[3]
            or sites.size < 4 or max_order < 1 or not 0 < rank_tolerance < 1
            or not np.isfinite(sites).all() or not np.isfinite(samples).all()
            or np.unique(sites).size != sites.size):
        raise ValueError("GATE snapshot_fit_input: finite unique supports and joint square samples required")
    rank = samples.shape[2]
    values = samples.transpose(1, 0, 2, 3).reshape(sites.size, 2 * rank, rank)
    nl, nr = (sites.size + 1) // 2, sites.size // 2
    shape = (nl * 2 * rank, nr * rank)
    pencil_bytes = 16 * shape[0] * shape[1]
    if pencil_bytes > int(max_pencil_bytes):
        raise ValueError(f"GATE snapshot_pencil_budget: {pencil_bytes} bytes per block pencil "
                         f"exceeds explicit {max_pencil_bytes} byte limit; reduce supports/density rank")
    scale = max(float(np.max(abs(sites))), 1e-12)
    mu, lam = sites[::2] / scale, sites[1::2] / scale
    hl, hr = values[::2], values[1::2]
    denominator = (mu[:, None] - lam[None, :])[:, :, None, None]
    L = ((hl[:, None] - hr[None]) / denominator).transpose(0, 2, 1, 3).reshape(shape)
    Ls = ((mu[:, None, None, None] * hl[:, None]
           - lam[None, :, None, None] * hr[None]) / denominator)
    Ls = Ls.transpose(0, 2, 1, 3).reshape(shape)
    U, singular, Vh = sla.svd(L, full_matrices=False)
    if singular[0] == 0:
        empty = kr.pole_residue_realization([], np.zeros((0, rank, rank)))
        return {"lesser": empty, "greater": empty, "order": 0, "singular_values": singular,
                "pencil_shape": shape, "pencil_bytes": pencil_bytes, "refit_error": 0.0}
    order = min(int(max_order), int(np.count_nonzero(singular > singular[0] * rank_tolerance)))
    requested_order = order
    Y, X = U[:, :order], Vh[:order].conj().T
    A = scale * ((Y.conj().T @ Ls @ X) / singular[:order, None])
    B = -scale * (Y.conj().T @ hl.reshape(nl * 2 * rank, rank)) / singular[:order, None]
    poles, vectors = sla.eig(A)
    if np.linalg.cond(vectors) > 1e12:
        raise ValueError("GATE snapshot_fit_pencil: defective Loewner modal candidate basis")
    modal_right = sla.solve(vectors, B)
    stable = poles.imag < -1e-12
    poles, modal_right = poles[stable], modal_right[stable]
    if poles.size == 0:
        raise ValueError("GATE snapshot_causal_poles: no causal input modal candidates")
    design = ((1.0 / (sites[:, None] - poles[None]))[:, :, None]
              * modal_right[None]).transpose(0, 2, 1).reshape(sites.size * rank, -1)
    norms = np.linalg.norm(design, axis=0)
    live = norms > 0.0
    design, poles, modal_right, norms = design[:, live], poles[live], modal_right[live], norms[live]
    design /= norms[None]
    rhs = values.transpose(0, 2, 1).reshape(sites.size * rank, 2 * rank)
    coefficients = sla.lstsq(design, rhs, cond=rank_tolerance, lapack_driver="gelsd")[0]
    A, B, C = np.diag(poles), modal_right / norms[:, None], coefficients.T
    order = poles.size
    reconstructed = (design @ coefficients).reshape(sites.size, rank, 2 * rank).transpose(0, 2, 1)
    refit_error = _relative_error(reconstructed, values)
    zero = np.zeros((rank, rank), dtype=complex)
    return {"lesser": (A, B, C[:rank], zero),
            "greater": (A, B, C[rank:], zero.copy()),
            "order": order, "requested_order": requested_order,
            "stable_rank_reduction": requested_order - order, "singular_values": singular,
            "refit_error": refit_error, "pencil_shape": shape, "pencil_bytes": pencil_bytes}


def _relative_error(actual, reference):
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return float("inf")
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


def compact_causal_realization(system, sites, *, tolerance=1e-9):
    """Remove canceled stable states by square-root balanced truncation.

    This changes numerical coordinates, never pole signs or physical
    residues.  Stable controllability/observability Gramians use generator
    ``-i A``.  A full-matrix evaluation check certifies the compact result
    at the caller's finite sites before it can be consumed by Sigma.  If
    truncation fails that check the original exact realization is retained;
    downstream pole/conditioning gates still apply, and no fit tolerance is
    relaxed merely to obtain a smaller state space.
    All operands are bounded reduced host matrices, not centroid arrays.
    """
    A, B, C, D = system
    n = A.shape[0]
    if n == 0:
        return system
    generator = -1j * A

    def gram_root(generator, forcing):
        value = sla.solve_continuous_lyapunov(generator, -forcing)
        value = 0.5 * (value + value.conj().T)
        eig, vec = sla.eigh(value)
        largest = max(float(eig[-1]), 1e-300)
        if eig[0] < -1e-8 * largest:
            raise ValueError("GATE snapshot_compact_gramian: unstable numerical Gramian")
        keep = eig > largest * 1e-13
        return vec[:, keep] * np.sqrt(eig[keep])[None, :]

    Rc = gram_root(generator, B @ B.conj().T)
    Ro = gram_root(generator.conj().T, C.conj().T @ C)
    if Rc.shape[1] == 0 or Ro.shape[1] == 0:
        compact = (np.zeros((0, 0), complex), np.zeros((0, B.shape[1]), complex),
                   np.zeros((C.shape[0], 0), complex), D.copy())
    else:
        U, singular, Vh = sla.svd(Ro.conj().T @ Rc, full_matrices=False)
        keep = singular > max(float(singular[0]) * tolerance * 0.01, 1e-300)
        invsqrt = 1.0 / np.sqrt(singular[keep])
        right = (Rc @ Vh[keep].conj().T) * invsqrt[None]
        left = invsqrt[:, None] * (U[:, keep].conj().T @ Ro.conj().T)
        compact = (left @ A @ right, left @ B, C @ right, D.copy())
    if np.any(np.linalg.eigvals(compact[0]).imag >= -1e-12):
        return system
    expected = np.stack([kr.evaluate(system, z) - D for z in sites])
    actual = np.stack([kr.evaluate(compact, z) - D for z in sites])
    if _relative_error(actual, expected) > tolerance:
        return system
    return compact


def screen_snapshot(sampler, fit_sites, check_sites, *, max_order=96,
                    fit_tolerance=1e-3, rank_tolerance=1e-11,
                    pole_tolerance=1e-12, max_projection_order=400,
                    self_negative=False, max_pencil_bytes=128 * 1024 ** 2,
                    real_fit_sites=None):
    """Build and certify a reduced joint screened snapshot from a callback.

    ``sampler(sites)`` returns exact ``[2,support,r,r]`` bubble values in
    Coulomb coordinates.  Its declared electronic support, density rank,
    and finite lag decay belong to the refresh receipt. Fit
    and check supports (Ry) must be disjoint and strictly above the axis.
    A plain dict returns ``retarded`` (total W), ``correlation`` (WcR),
    ``lesser`` (one-sided W<), and sample/error/pole receipts.  Causal
    projection is exact on this fitted rational model, using the reference
    Schur/Sylvester owner.  It retains the lesser 1/z tail.  It refuses
    unresolved or growing fitted poles, rather than moving them.
    ``self_negative=True`` imposes crossing on a self-inverse q, in a real
    Coulomb factor gauge.  It averages the two fitted one-sided functions;
    the independent exact bubble/noise checks still bound this operation.
    ``max_projection_order`` bounds causal-projection states and
    ``max_pencil_bytes`` bounds each block Loewner pencil before allocation;
    increasing these capacity limits does not change fit acceptance.
    Optional ``real_fit_sites`` are exact, finite boundary training supports
    for the same joint fit, not frequency integration nodes.  The sampler's
    positive finite lag decay makes these samples finite. It is not a
    validated physical adiabatic statistical closure. They
    must be disjoint from ``check_sites.real``, which remains an independent
    dressed-noise oracle; no check values enter training.
    """
    fit_sites, check_sites = (np.asarray(x, dtype=complex) for x in (fit_sites, check_sites))
    if (check_sites.size == 0 or not np.isfinite(fit_sites).all()
            or not np.isfinite(check_sites).all() or np.any(fit_sites.imag <= 0)
            or np.any(check_sites.imag <= 0) or np.intersect1d(fit_sites, check_sites).size
            or not np.isfinite(fit_tolerance) or fit_tolerance <= 0):
        raise ValueError("GATE snapshot_check_support: positive disjoint fit/check supports required")
    real_fit_sites = (np.empty(0, complex) if real_fit_sites is None
                      else np.asarray(real_fit_sites, dtype=complex))
    if (real_fit_sites.ndim != 1 or not np.isfinite(real_fit_sites).all()
            or np.any(real_fit_sites.imag != 0)
            or np.unique(real_fit_sites).size != real_fit_sites.size
            or np.intersect1d(real_fit_sites, check_sites.real).size):
        raise ValueError("GATE snapshot_check_support: finite unique real training supports "
                         "must be disjoint from real noise checks")
    data = np.asarray(sampler(fit_sites), dtype=complex)
    joint_sites = fit_sites
    if real_fit_sites.size:
        joint_sites = np.concatenate((fit_sites, real_fit_sites))
        data = np.concatenate((data, np.asarray(sampler(real_fit_sites), dtype=complex)), axis=1)
    fit = joint_loewner(joint_sites, data, max_order=max_order, rank_tolerance=rank_tolerance,
                        max_pencil_bytes=max_pencil_bytes)
    pl, pg = fit["lesser"], fit["greater"]
    if self_negative:
        A, B, Cl, D = pl
        Cg = pg[2]
        rank = D.shape[0]
        # Shared realization of .5*(Pl+mirror(Pg)) and its exact mirror;
        # retaining the shared A/B avoids a redundant four-block PR model.
        joint = (sla.block_diag(A, -A.conj()), np.vstack((B, B.conj())),
                 0.5 * np.vstack((np.hstack((Cl, Cg.conj())),
                                  np.hstack((Cg, Cl.conj())))),
                 np.zeros((2 * rank, rank), complex))
        joint = compact_causal_realization(joint, check_sites)
        Aj, Bj, Cj, _ = joint
        pl, pg = ((Aj, Bj, Cj[:rank], D.copy()), (Aj, Bj, Cj[rank:], D.copy()))
        fit["order"] = Aj.shape[0]
    A, B, Cl, D = pl
    pr = (A, B, pg[2] - Cl, D.copy())
    rank = D.shape[0]
    wr = kr.screen_retarded(pr, np.eye(rank))
    poles = np.linalg.eigvals(wr[0])
    input_poles = np.linalg.eigvals(A)
    if np.any(input_poles.imag >= -pole_tolerance):
        raise ValueError("GATE snapshot_causal_poles: unstable/unresolved fitted input poles; "
                         "exact damped bubble input has only lower-half-plane poles")
    if np.any(poles.imag >= -pole_tolerance):
        raise ValueError("GATE snapshot_causal_poles: unstable/unresolved collective feedback poles")
    exact = np.asarray(sampler(check_sites), dtype=complex)
    approx = np.stack([[kr.evaluate(system, z) for z in check_sites] for system in (pl, pg)])
    bubble_error = _relative_error(approx, exact)
    exact_w = np.stack([np.linalg.solve(np.eye(rank) - (exact[1, i] - exact[0, i]),
                                        np.eye(rank)) for i in range(check_sites.size)])
    approx_w = np.stack([kr.evaluate(wr, z) for z in check_sites])
    screening_error = _relative_error(approx_w - np.eye(rank), exact_w - np.eye(rank))
    refit_error = float(fit.get("refit_error", 0.0))
    if max(refit_error, bubble_error, screening_error) > fit_tolerance:
        raise ValueError(f"GATE snapshot_fit_error: bubble={bubble_error:.3e}, "
                         f"screening={screening_error:.3e}, train={refit_error:.3e}; "
                         f"tolerance={fit_tolerance:.3e}; "
                         "increase supports/order or revise density-space rank")
    projection_order = 4 * fit["order"]
    if projection_order > max_projection_order:
        raise ValueError(f"GATE snapshot_projection_budget: {projection_order} states exceeds "
                         f"{max_projection_order}; lower fit order or increase explicit budget")
    # Remove exactly canceled noise states from W^R before composing the
    # Keldysh product; they otherwise produce defective unobservable blocks.
    wr = compact_causal_realization(wr, check_sites)
    wl = kr.screened_lesser(wr, pl, pole_tolerance=pole_tolerance)
    wl = compact_causal_realization(wl, check_sites)
    wc = (wr[0], wr[1], wr[2], np.zeros_like(wr[3]))
    # This independent check sees the dressed noise, rather than inferring
    # its accuracy from separate P and retarded W residuals. The declared
    # finite lag decay in the sampler makes real support evaluation finite.
    real_sites = check_sites.real.astype(complex)
    noise_data = np.asarray(sampler(real_sites), dtype=complex)
    noise_exact, noise_fit = [], []
    for i, z in enumerate(real_sites):
        wp = np.linalg.solve(np.eye(rank) - noise_data[1, i] + noise_data[0, i], np.eye(rank))
        phat = noise_data[0, i] - noise_data[0, i].conj().T
        noise_exact.append(wp @ phat @ wp.conj().T)
        wless = kr.evaluate(wl, z)
        noise_fit.append(wless - wless.conj().T)
    noise_error = _relative_error(np.asarray(noise_fit), np.asarray(noise_exact))
    if noise_error > fit_tolerance:
        raise ValueError(f"GATE snapshot_noise_error: dressed noise={noise_error:.3e}; "
                         f"tolerance={fit_tolerance:.3e}; increase supports/order")
    return {"retarded": wr, "correlation": wc, "lesser": wl,
            "polarization_lesser": pl, "polarization_greater": pg,
            "receipt": {"density_rank": rank, "fit_order": fit["order"],
                        "projection_order": projection_order,
                        "bubble_check_error": bubble_error,
                        "screening_check_error": screening_error,
                        "noise_check_error": noise_error,
                        "fit_tolerance": float(fit_tolerance),
                        "stable_rank_reduction": int(fit.get("stable_rank_reduction", 0)),
                        "fit_method": "full_block_loewner_causal_basis_refit",
                        "training_refit_error": refit_error,
                        "fit_pencil_shape": tuple(fit["pencil_shape"]),
                        "fit_pencil_bytes": int(fit["pencil_bytes"]),
                        "self_negative_crossing": bool(self_negative),
                        "fit_sites_ry": fit_sites.copy(), "check_sites_ry": check_sites.copy(),
                        "real_fit_sites_ry": real_fit_sites.copy(),
                        "retarded_poles_ry": np.linalg.eigvals(wr[0]),
                        "lesser_order": wl[0].shape[0],
                        "noise_only_poles_allowed": True,
                        "polarization_lesser_tail": Cl @ B,
                        "lesser_tail": wl[2] @ wl[1]}}


def screen_snapshot_partner(representative, sampler, check_sites, *, fit_tolerance=None):
    """Derive a q partner algebraically and certify its exact heldout bubble.

    Factors must obey ``F_minus=conj(F_q)`` in the same reduced gauge.
    No independent partner fit or second causal Schur projection is made:
    ``Wc_minus=-mirror(Wc_q)`` and
    ``Wless_minus=mirror(Wc_q+Wless_q)``.  These identities follow from
    the retarded Dyson equation and the two-sided Keldysh product; they
    hold for the full-matrix ordered snapshot bubble, including coherence.
    Independent partner bubble, retarded response and dressed-noise gates
    prevent enforcing crossing outside the accepted fit error.
    """
    sites = np.asarray(check_sites, dtype=complex)
    if sites.size == 0 or not np.isfinite(sites).all() or np.any(sites.imag <= 0):
        raise ValueError("GATE snapshot_check_support: positive partner check supports required")
    tolerance = (float(representative["receipt"]["fit_tolerance"])
                 if fit_tolerance is None else float(fit_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("GATE snapshot_check_support: finite positive partner fit tolerance required")
    pl = mirror(representative["polarization_greater"])
    pg = mirror(representative["polarization_lesser"])
    wr = kr.scale(mirror(representative["retarded"]), -1.0)
    wc = kr.scale(mirror(representative["correlation"]), -1.0)
    wl = mirror(kr.add(representative["correlation"], representative["lesser"]))
    wl = compact_causal_realization(wl, sites)
    rank = pl[3].shape[0]
    exact = np.asarray(sampler(sites), dtype=complex)
    approx = np.stack([[kr.evaluate(h, z) for z in sites] for h in (pl, pg)])
    bubble_error = _relative_error(approx, exact)
    exact_w = np.stack([np.linalg.solve(np.eye(rank) - exact[1, i] + exact[0, i],
                                        np.eye(rank)) for i in range(sites.size)])
    approx_w = np.stack([kr.evaluate(wr, z) for z in sites])
    screening_error = _relative_error(approx_w - np.eye(rank), exact_w - np.eye(rank))
    real_sites = sites.real.astype(complex)
    noise_data = np.asarray(sampler(real_sites), dtype=complex)
    noise_exact, noise_fit = [], []
    for i, z in enumerate(real_sites):
        wp = np.linalg.solve(np.eye(rank) - noise_data[1, i] + noise_data[0, i], np.eye(rank))
        phat = noise_data[0, i] - noise_data[0, i].conj().T
        noise_exact.append(wp @ phat @ wp.conj().T)
        value = kr.evaluate(wl, z)
        noise_fit.append(value - value.conj().T)
    noise_error = _relative_error(np.asarray(noise_fit), np.asarray(noise_exact))
    if max(bubble_error, screening_error, noise_error) > tolerance:
        raise ValueError(f"GATE snapshot_partner_error: bubble={bubble_error:.3e}, "
                         f"screening={screening_error:.3e}, noise={noise_error:.3e}; "
                         f"tolerance={tolerance:.3e}; improve representative fit or check factor gauge")
    receipt = dict(representative["receipt"], derived_by_crossing=True,
                   self_negative_crossing=False, bubble_check_error=bubble_error,
                   screening_check_error=screening_error, noise_check_error=noise_error,
                   check_sites_ry=sites.copy(), retarded_poles_ry=np.linalg.eigvals(wr[0]),
                   lesser_order=wl[0].shape[0], polarization_lesser_tail=pl[2] @ pl[1],
                   lesser_tail=wl[2] @ wl[1])
    return {"retarded": wr, "correlation": wc, "lesser": wl,
            "polarization_lesser": pl, "polarization_greater": pg, "receipt": receipt}


def retarded_reference_snapshot(sampler, fit_sites, check_sites, *, max_order=48,
                                fit_tolerance=0.02, max_pencil_bytes=128 * 1024**2,
                                representative=None, self_negative=False, collision_eta=0.0):
    """Fit only full-density retarded screening and assign reference branches.

    ``sampler(sites)`` supplies ordered [P<,P>] in reduced Coulomb
    coordinates, but ONLY their difference PR enters the fit, checks, and
    retarded Dyson equation. The lesser array returned for the existing
    Sigma executor is the equilibrium-style T=0 frequency branch of WR,
    with no independent fluctuation fit or noise/PSD acceptance gate.
    Frozen and updated modes can therefore use precisely the same
    retarded approximation while retaining full rho in electronic lines.
    Supports and pole frequencies use Ry; the sampler owns its explicitly
    declared positive response lag decay, independent of collision eta.
    A growing theta(t) response is admitted only when both the declared
    collision_eta and every Laplace sample height exceed its pole growth.
    """
    fit_sites, check_sites = (np.asarray(x, complex) for x in (fit_sites, check_sites))
    if (fit_sites.ndim != 1 or check_sites.ndim != 1 or fit_sites.size < 4
            or not check_sites.size or not np.isfinite(fit_sites).all()
            or not np.isfinite(check_sites).all() or np.any(fit_sites.imag <= 0)
            or np.any(check_sites.imag <= 0) or np.intersect1d(fit_sites, check_sites).size
            or not np.isfinite(fit_tolerance) or fit_tolerance <= 0):
        raise ValueError("GATE snapshot_check_support: positive disjoint retarded fit/check supports required")
    exact_channels = np.asarray(sampler(check_sites))
    exact = exact_channels[1] - exact_channels[0]
    rank = exact.shape[-1]
    fit = None
    if representative is None:
        channels = np.asarray(sampler(fit_sites))
        target = channels[1] - channels[0]
        fit = joint_loewner(fit_sites, np.stack((np.zeros_like(target), target)),
            max_order=max_order, max_pencil_bytes=max_pencil_bytes)
        pr = fit["greater"]
        if self_negative:
            pr = kr.scale(kr.add(pr, kr.scale(mirror(pr), -1.0)), 0.5)
            pr = compact_causal_realization(pr, check_sites)
        wr = kr.screen_retarded(pr, np.eye(rank))
        wc = (*wr[:3], np.zeros((rank, rank), complex))
        # Bound the retarded/branch/common-Sigma dense work before modal
        # allocation; each pole receives one state, not rank states.
        n = wc[0].shape[0]
        dense_bytes = 16 * (8 * n**2 + 16 * n * rank)
        if dense_bytes > max_pencil_bytes:
            raise ValueError("GATE snapshot_reference_workspace: retarded/branch union exceeds max_pencil_bytes")
        branches = kr.retarded_reference_branches(wc, evaluation_height=collision_eta)
        wc, wl = branches["correlation"], branches["lesser"]
        wr = (*wc[:3], np.eye(rank, dtype=complex))
    else:
        pr = kr.scale(mirror(representative["polarization_retarded"]), -1.0)
        wc = kr.scale(mirror(representative["correlation"]), -1.0)
        wr = (*wc[:3], np.eye(rank, dtype=complex))
        # The symmetric frequency branch rule commutes with crossing.
        branches = kr.retarded_reference_branches(wc, evaluation_height=collision_eta)
        wc, wl = branches["correlation"], branches["lesser"]
        wr = (*wc[:3], np.eye(rank, dtype=complex))
    growth = branches["maximum_pole_growth"]
    support_margin = float(min(np.min(fit_sites.imag), np.min(check_sites.imag)) - growth)
    if support_margin <= 1e-12:
        raise ValueError(f"GATE snapshot_reference_support_height: Laplace supports must exceed "
                         f"retarded pole growth {growth:.9g}; margin={support_margin:.3e}")
    approximate = np.stack([kr.evaluate(pr, z) for z in check_sites])
    bubble_error = _relative_error(approximate, exact)
    expected_w = np.stack([np.linalg.solve(np.eye(rank) - p, np.eye(rank)) - np.eye(rank)
                           for p in exact])
    screened_error = _relative_error(np.stack([kr.evaluate(wc, z) for z in check_sites]), expected_w)
    training_error = float(fit["refit_error"]) if fit is not None else 0.0
    if max(bubble_error, screened_error, training_error) > fit_tolerance:
        raise ValueError(f"GATE snapshot_reference_fit: bubble={bubble_error:.3e}, "
                         f"response={screened_error:.3e}, train={training_error:.3e}; "
                         f"tolerance={fit_tolerance:.3e}")
    inherited = representative["receipt"] if representative is not None else {}
    receipt = dict(inherited, density_rank=rank, fit_order=pr[0].shape[0],
        projection_order=wc[0].shape[0] + wl[0].shape[0], bubble_check_error=bubble_error,
        screening_check_error=screened_error, noise_check_error=0.0,
        fit_tolerance=float(fit_tolerance), training_refit_error=training_error,
        fit_method="retarded_only_loewner_vacuum_reference", noise_check_kind="not_applicable_retarded_reference",
        screening_density_approximation="full_matrix_ordered_snapshot",
        noise_approximation="no_independent_noise_retarded_T0_reference_branches",
        lag_decay_used=True, derived_by_crossing=representative is not None,
        self_negative_crossing=bool(self_negative),
        reference_zero_frequency_count=branches["zero_frequency_count"],
        reference_signed_gain_pole_count=branches["signed_gain_pole_count"],
        reference_complex_residue_pole_count=branches["complex_residue_pole_count"],
        reference_maximum_pole_growth_ry=growth,
        reference_collision_eta_ry=branches["evaluation_height"],
        reference_laplace_margin_ry=branches["laplace_margin"],
        reference_support_margin_ry=support_margin,
        reference_modal_condition=branches["condition"],
        reference_modal_error=branches["reconstruction_error"],
        retarded_poles_ry=branches["poles"], lesser_order=wl[0].shape[0],
        fit_pencil_shape=tuple(fit["pencil_shape"]) if fit is not None else inherited.get("fit_pencil_shape"),
        fit_pencil_bytes=int(fit["pencil_bytes"]) if fit is not None else inherited.get("fit_pencil_bytes"))
    return dict(retarded=wr, correlation=wc, lesser=wl, polarization_retarded=pr, receipt=receipt)


def _lossless_bath_projection(energy, coupling, sites, order, budget):
    """Project a diagonal transition Hamiltonian on an orthonormal Krylov basis."""
    n, rank = coupling.shape
    bin_index, bin_weights = None, None
    original_count = n
    # A preliminary disjoint-bin Galerkin space bounds host work for the
    # production rank-one case. Its local vector is the normalized coupling,
    # its Hamiltonian is the coupling-weighted mean energy, and its bath
    # averages Bose WEIGHTS at the queried collective frequency, never beta.
    # Both reductions are subsequently checked against original transitions.
    bin_count = max(256, 16 * int(order))
    if rank == 1 and n > 2 * bin_count and np.ptp(energy) > 0:
        raw_index = np.minimum(((energy - energy.min()) / np.ptp(energy)
                               * bin_count).astype(int), bin_count - 1)
        weight = abs(coupling[:, 0]) ** 2
        sums = np.bincount(raw_index, weights=weight, minlength=bin_count)
        live = sums > 0
        mapping = np.cumsum(live) - 1
        bin_index = mapping[raw_index]
        bin_weights = weight / sums[raw_index]
        energy = (np.bincount(raw_index, weights=weight * energy,
                              minlength=bin_count) / np.where(live, sums, 1))[live]
        coupling = np.sqrt(sums[live])[:, None].astype(complex)
        n = energy.size
    capacity = min(n, int(order))
    required = 16 * n * (capacity + 4 * rank) + (16 * original_count if bin_index is not None else 0)
    if required > budget:
        raise ValueError(f"GATE snapshot_bath_workspace: {required} bytes exceeds {budget}; "
                         "increase max_pencil_bytes or reduce max_order")
    basis = np.empty((n, capacity), complex)
    used, candidate = 0, coupling
    sorted_sites = np.asarray(sites)[np.argsort(np.asarray(sites).real)]
    shifts = sorted_sites[np.linspace(0, sorted_sites.size - 1, capacity, dtype=int)]
    while used < capacity:
        for _ in range(2):
            if used:
                candidate -= basis[:, :used] @ (basis[:, :used].conj().T @ candidate)
        q, singular, _ = sla.svd(candidate, full_matrices=False)
        live = singular > max(float(np.max(singular, initial=0.0)) * 1e-12,
                              np.linalg.norm(coupling) * 1e-14)
        q = q[:, live][:, :capacity - used]
        if not q.shape[1]:
            break
        basis[:, used:used + q.shape[1]] = q
        used += q.shape[1]
        shift = shifts[(used // max(rank, 1)) % shifts.size]
        candidate = q / (energy[:, None] - shift)
    basis = basis[:, :used]
    h = basis.conj().T @ (energy[:, None] * basis)
    h = 0.5 * (h + h.conj().T)
    c = basis.conj().T @ coupling
    return dict(h=h, c=c, basis=basis, energy=energy, workspace_bytes=required,
                bin_index=bin_index, bin_weights=bin_weights,
                original_transition_count=original_count, projected_transition_count=n)


def _bath_bose(beta, frequency):
    """Evaluate Bose weights with explicit zero-temperature signed limits."""
    argument = beta * frequency
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = 1.0 / np.expm1(argument)
    result = np.where(argument > 700, 0.0, result)
    result = np.where(argument < -700, -1.0, result)
    if not np.isfinite(result).all():
        raise ValueError("GATE snapshot_bath_temperature: unresolved zero pair temperature")
    return result


def _lossless_response_check(correlation, energy, difference, vertices, sites):
    """Compare reduced Dyson response with independent microscopic samples."""
    rank = vertices.shape[1]
    exact, approximate = [], []
    for z in sites:
        p = np.einsum("j,jr,js->rs", difference / (z - energy),
                      vertices, vertices.conj(), optimize=True)
        exact.append(np.linalg.solve(np.eye(rank) - p, np.eye(rank)) - np.eye(rank))
        approximate.append(kr.evaluate(correlation, z))
    return _relative_error(np.asarray(approximate), np.asarray(exact))


def _lossless_dense_workspace(n_response, n_noise, rank, *, self_negative,
                              basis_bytes, feedback_order, budget):
    """Bound canonical banks and their downstream common-union allocation."""
    nc = rank * n_response * (2 if self_negative else 1)
    nl = rank * ((3 * n_response + 2 * n_noise) if self_negative
                 else (2 * n_response + n_noise))
    # Include a reciprocal partner's larger lesser union, the final Sigma
    # union, construction copies, and small feedback/metric work arrays.
    dense_bytes = 16 * (2 * nc**2 + 2 * nl**2 + (nc + nl)**2
                        + 4 * feedback_order**2 + 8 * rank * (nc + nl))
    required = int(basis_bytes + dense_bytes)
    if required > int(budget):
        raise ValueError(f"GATE snapshot_bath_dense_workspace: {required} bytes exceeds {budget}; "
                         "reduce density rank/order or increase max_pencil_bytes")
    return required


def lossless_bath_snapshot(vertices, energy, density, partner, fit_sites, check_sites, *,
                           spin_weight=1.0, max_order=48, fit_tolerance=0.02,
                           max_pencil_bytes=128 * 1024 ** 2, representative=None,
                           self_negative=False):
    """Construct a secular lossless colored-bath RPA screening approximation.

    All inputs are replicated host arrays for ONE q: vertices [k,b,d,r],
    energies [k,b] in Ry, and full density [k,b,b]. Screening explicitly
    pinches density between distinct energy eigenspaces; coherences inside
    exactly degenerate blocks are diagonalized with their vertices. The
    electronic self-energy must continue to use the original full density.

    A signed Hermitian rational Krylov projection replaces the transition
    Hamiltonian, and the colored bath is projected on the SAME subspace.
    Collective lesser residues are R S_beta R. Independent zero-response
    noise supplies both collective tails and its dressed bare poles.
    Gaps within the declared 1e-12 Ry screening resolution use an aggregated static
    Wick transition reservoir, with collective tails and a bare-zero
    nullspace remnant. Equal-population zero-gap noise is a separate exact
    static delta contribution and has no collective tails. Nonzero gaps,
    above that resolution retain their frequency-dependent colored bath weights.
    Exact thermal KMS refers to the resolved screening Hamiltonian. The
    response check still uses original electronic gaps, and Sigma phases
    retain the original energies.
    Real poles use only the subsequent electronic collision eta; no lag
    decay, real-frequency grid, or positive-residue clipping is used.
    """
    T, energy, rho = (np.asarray(x) for x in (vertices, energy, density))
    fit_sites, check_sites = (np.asarray(x, complex) for x in (fit_sites, check_sites))
    if (fit_sites.ndim != 1 or check_sites.ndim != 1 or not fit_sites.size or not check_sites.size
            or not np.isfinite(fit_sites).all() or not np.isfinite(check_sites).all()
            or np.any(fit_sites.imag <= 0) or np.any(check_sites.imag <= 0)
            or np.intersect1d(fit_sites, check_sites).size
            or not np.isfinite(fit_tolerance) or fit_tolerance <= 0):
        raise ValueError("GATE snapshot_check_support: positive disjoint bath fit/check supports required")
    partner = np.asarray(partner, int)
    nk, nb = energy.shape
    rank = T.shape[-1]
    if (T.shape[:3] != (nk, nb, nb) or rho.shape != (nk, nb, nb)
            or partner.shape != (nk,) or not np.array_equal(np.sort(partner), np.arange(nk))
            or max_order < 2 or not np.isfinite(T).all() or not np.isfinite(energy).all()
            or not np.isfinite(rho).all()
            or np.max(abs(rho - rho.conj().transpose(0, 2, 1))) > 1e-10):
        raise ValueError("GATE snapshot_bath_input: incompatible transition tables")
    # Exact degeneracy is a gauge freedom. Rotate only within those blocks,
    # then discard matrix entries between different frozen energy values.
    rotations = np.broadcast_to(np.eye(nb, dtype=complex), rho.shape).copy()
    pinched = rho * (abs(energy[:, :, None] - energy[:, None, :]) <= 1e-12)
    discarded_coherence = float(np.linalg.norm(rho - pinched))
    f = np.real(np.diagonal(rho, axis1=-2, axis2=-1)).copy()
    for k in range(nk):
        remaining = set(range(nb))
        while remaining:
            first = min(remaining)
            block = np.flatnonzero(abs(energy[k] - energy[k, first]) <= 1e-12)
            remaining.difference_update(block.tolist())
            if block.size > 1:
                values, vectors = sla.eigh(rho[k][np.ix_(block, block)])
                rotations[k][np.ix_(block, block)] = vectors
                f[k, block] = values
    if np.any(f < -1e-12) or np.any(f > 1 + 1e-12):
        raise ValueError("GATE snapshot_bath_density: screening populations outside [0,1]")
    bounded_f = np.minimum(1.0, np.maximum(0.0, f))
    population_roundoff = float(np.max(abs(bounded_f - f)))
    f = bounded_f
    T = np.einsum("kab,kacr->kbcr", rotations, T, optimize=True)
    T = np.einsum("kdcr,kce->kder", T, rotations[partner].conj(), optimize=True)
    e = (energy[:, :, None] - energy[partner, None, :]).reshape(-1)
    fb = np.broadcast_to(f[:, :, None], (nk, nb, nb)).reshape(-1)
    fd = np.broadcast_to(f[partner, None, :], (nk, nb, nb)).reshape(-1)
    d = fd - fb
    v = T.reshape(-1, rank) * np.sqrt(float(spin_weight) / nk)
    vertex_live = np.linalg.norm(v, axis=1) > 0
    original_e = e
    gap_resolution = 1e-12
    resolved_zero = abs(original_e) <= gap_resolution
    gap_adjustment = float(np.max(abs(original_e[resolved_zero & vertex_live]), initial=0.0))
    projected_gap_count = int(np.count_nonzero(resolved_zero & (original_e != 0) & vertex_live))
    e = np.where(resolved_zero, 0.0, original_e)
    if representative is not None:
        n_response = representative["correlation"][0].shape[0] // rank
        n_noise = representative["lesser"][0].shape[0] // rank - n_response
        dense_workspace_bytes = _lossless_dense_workspace(n_response, n_noise, rank,
            self_negative=False, basis_bytes=0, feedback_order=0, budget=max_pencil_bytes)
        wr = kr.scale(mirror(representative["retarded"]), -1.0)
        wc = kr.scale(mirror(representative["correlation"]), -1.0)
        wl = mirror(kr.add(representative["correlation"], representative["lesser"]))
        error = _lossless_response_check(wc, original_e, d, v, check_sites)
        if error > fit_tolerance:
            raise ValueError(f"GATE snapshot_bath_partner_fit: response={error:.3e} "
                             f"exceeds {fit_tolerance:.3e}")
        receipt = dict(representative["receipt"], derived_by_crossing=True,
            self_negative_crossing=False, bubble_check_error=error,
            screening_check_error=error, discarded_screening_coherence_norm=discarded_coherence,
            lesser_order=wl[0].shape[0], screening_population_roundoff_correction=population_roundoff,
            dense_workspace_bytes=dense_workspace_bytes,
            screening_gap_resolution_ry=gap_resolution,
            screening_gap_projection_count=projected_gap_count,
            maximum_screening_gap_adjustment_ry=gap_adjustment,
            final_screened_model="reciprocal_partner_of_projected_bath")
        return dict(retarded=wr, correlation=wc, lesser=wl, receipt=receipt)
    scale = max(float(np.ptp(energy)), 1.0)
    static = (e == 0) & vertex_live
    static_response = np.einsum("j,jr,js->rs", d[static], v[static], v[static].conj(), optimize=True)
    static_reservoir = static & (d != 0)
    static_exact = static & (d == 0)
    static_noise = np.einsum("j,jr,js->rs", fb[static_reservoir] * (1 - fd[static_reservoir]),
                             v[static_reservoir], v[static_reservoir].conj(), optimize=True)
    static_exact_noise = np.einsum("j,jr,js->rs", fb[static_exact] * (1 - fd[static_exact]),
                                   v[static_exact], v[static_exact].conj(), optimize=True)
    static_response = 0.5 * (static_response + static_response.conj().T)
    static_noise = 0.5 * (static_noise + static_noise.conj().T)
    static_total_noise = static_noise + static_exact_noise
    static_absolute_response = np.einsum("j,jr,js->rs", abs(d[static]),
        v[static], v[static].conj(), optimize=True)
    static_values, static_vectors = sla.eigh(static_response)
    static_scale = max(np.linalg.norm(static_absolute_response), 1e-300)
    static_roundoff = 64 * np.finfo(float).eps * static_scale
    static_live = abs(static_values) > static_roundoff
    static_null = static_vectors[:, ~static_live]
    static_rank_error = float(np.linalg.norm(static_values[~static_live]) / static_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (np.log1p(d / fb) + np.log1p(d / (1 - fd))) / e
    branches = []
    order = max(1, int(max_order) // 2)
    for sign in (1, -1):
        live = (d * sign > 0) & vertex_live & ~static
        if not np.any(live):
            continue
        branch = _lossless_bath_projection(e[live], np.sqrt(abs(d[live]))[:, None] * v[live].conj(),
            fit_sites, order, int(max_pencil_bytes) - sum(b["workspace_bytes"] for b in branches))
        branch.update(sign=sign, beta=beta[live])
        choices = (np.inf, -np.inf, branch["beta"][0])
        baseline = max(choices, key=lambda value: np.count_nonzero(branch["beta"] == value))
        branch.update(baseline_beta=baseline, exceptional_beta=branch["beta"] != baseline)
        branches.append(branch)
    zero = (d == 0) & (fb > 0) & (fb < 1) & vertex_live & ~static
    noise = None
    if np.any(zero):
        noise = _lossless_bath_projection(e[zero], np.sqrt(fb[zero] * (1 - fb[zero]))[:, None]
            * v[zero].conj(), fit_sites, int(max_order), int(max_pencil_bytes)
            - sum(b["workspace_bytes"] for b in branches))
    for sign in (1, -1):
        live = static_live & (static_values * sign > 0)
        if not np.any(live):
            continue
        count = int(np.count_nonzero(live))
        c = np.sqrt(abs(static_values[live]))[:, None] * static_vectors[:, live].conj().T
        branches.append(dict(h=np.zeros((count, count), complex), c=c, sign=sign,
            static_aggregate=True, workspace_bytes=16 * (count**2 + count * rank),
            original_transition_count=int(np.count_nonzero(static)), projected_transition_count=count))
    planned_basis_bytes = sum(b["workspace_bytes"] for b in branches) + (noise["workspace_bytes"] if noise else 0)
    planned_feedback_order = sum(b["h"].shape[0] for b in branches)
    planned_noise_count = (noise["h"].shape[0] if noise else 0) + int(
        bool(np.any(static_total_noise) and static_null.shape[1]))
    _lossless_dense_workspace(planned_feedback_order, planned_noise_count, rank,
        self_negative=self_negative, basis_bytes=planned_basis_bytes,
        feedback_order=planned_feedback_order, budget=max_pencil_bytes)
    if branches:
        A = sla.block_diag(*(b["h"] for b in branches))
        B = np.vstack([b["c"] for b in branches])
        C = np.hstack([b["sign"] * b["c"].conj().T for b in branches])
        pr = (A, B, C, np.zeros((rank, rank), complex))
    else:
        pr = kr.pole_residue_realization([], np.zeros((0, rank, rank)))
    wr = kr.screen_retarded(pr, np.eye(rank))
    metric = sla.block_diag(*(b["sign"] * np.eye(b["h"].shape[0]) for b in branches)) if branches else None
    poles, residues, transition_factors = kr.lossless_residues(wr, metric=metric,
                                                            transition_factors=True)
    basis_bytes = sum(b["workspace_bytes"] for b in branches) + (noise["workspace_bytes"] if noise else 0)
    static_bare_noise = bool(np.any(static_total_noise) and static_null.shape[1])
    noise_count = (noise["h"].shape[0] if noise else 0) + int(static_bare_noise)
    dense_workspace_bytes = _lossless_dense_workspace(len(poles), noise_count, rank,
        self_negative=self_negative, basis_bytes=basis_bytes, feedback_order=pr[0].shape[0],
        budget=max_pencil_bytes)
    wc = kr.pole_residue_realization(poles, residues)
    wr = (*wc[:3], np.eye(rank, dtype=complex))
    # Independent exact upper-half-plane response checks certify projection.
    error = _lossless_response_check(wc, original_e, d, v, check_sites)
    if error > fit_tolerance:
        raise ValueError(f"GATE snapshot_bath_fit: response={error:.3e} exceeds {fit_tolerance:.3e}; "
                         "increase max_order or supports")
    lesser = []
    minimum, normalization_error = 0.0, 0.0
    for pole, residue, transition in zip(poles, residues, transition_factors):
        if abs(pole) < 1e-10 * scale:
            raise ValueError("GATE snapshot_bath_zero_mode: collective zero mode unresolved")
        value = np.zeros((rank, rank), complex)
        derivative = np.zeros_like(value)
        offset = 0
        for b in branches:
            stop = offset + b["h"].shape[0]
            reduced = transition[offset:stop]
            offset = stop
            norm = reduced.conj().T @ reduced
            derivative += b["sign"] * norm
            if b.get("static_aggregate", False):
                continue
            baseline = float(_bath_bose(np.asarray(b["baseline_beta"]), pole))
            value += b["sign"] * baseline * norm
            exceptional = b["exceptional_beta"]
            if np.any(exceptional):
                weight = _bath_bose(b["beta"][exceptional], pole) - baseline
                if b["bin_index"] is None:
                    g = b["basis"][exceptional] @ reduced
                else:
                    weight = np.bincount(b["bin_index"][exceptional],
                        weights=b["bin_weights"][exceptional] * weight,
                        minlength=b["projected_transition_count"])
                    g = b["basis"] @ reduced
                value += b["sign"] * g.conj().T @ (weight[:, None] * g)
        value += residue @ static_noise @ residue / pole**2
        normalization_error = max(normalization_error,
            _relative_error(derivative, residue))
        if normalization_error > 1e-6:
            raise ValueError("GATE snapshot_bath_normalization: collective derivative normalization failed")
        if noise is not None:
            h, c = noise["h"], noise["c"]
            if np.min(abs(pole - sla.eigvalsh(h))) < 1e-10 * scale:
                raise ValueError("GATE snapshot_bath_coincidence: collective and noise poles coincide")
            g = sla.solve(pole * np.eye(h.shape[0]) - h, c)
            value += residue @ (g.conj().T @ (h @ g) / pole) @ residue
        value = 0.5 * (value + value.conj().T)
        for component in (value, value + residue):
            eig = sla.eigvalsh(component)
            minimum = min(minimum, float(eig[0]))
            if eig[0] < -1e-8 * max(np.linalg.norm(component), np.linalg.norm(residue), 1e-30):
                raise ValueError("GATE snapshot_bath_noise_psd: negative lesser/greater collective residue")
        lesser.append(value)
    lesser_poles = list(poles)
    if static_bare_noise:
        finite_p0 = np.zeros((rank, rank), complex)
        for b in branches:
            if b.get("static_aggregate", False):
                continue
            if np.linalg.cond(b["h"]) > 1e12:
                raise ValueError("GATE snapshot_bath_zero_limit: unresolved finite-gap response at zero")
            finite_p0 += b["sign"] * b["c"].conj().T @ sla.solve(-b["h"], b["c"])
        restricted = static_null.conj().T @ (np.eye(rank) - finite_p0) @ static_null
        if np.linalg.cond(restricted) > 1e12:
            raise ValueError("GATE snapshot_bath_zero_limit: singular zero-gap nullspace screening")
        w0 = static_null @ sla.solve(restricted, static_null.conj().T)
        lesser_poles.append(0.0)
        lesser.append(w0 @ static_total_noise @ w0.conj().T)
    if noise is not None:
        values, vectors = sla.eigh(noise["h"])
        outputs = noise["c"].conj().T @ vectors
        for pole, output in zip(values, outputs.T):
            if poles.size and np.min(abs(poles - pole)) < 1e-10 * scale:
                raise ValueError("GATE snapshot_bath_coincidence: independent noise and collective poles coincide")
            dressed = kr.evaluate(wr, pole) @ output
            lesser_poles.append(float(pole))
            lesser.append(np.outer(dressed, dressed.conj()))
    wl = kr.pole_residue_realization(np.asarray(lesser_poles),
        np.asarray(lesser).reshape(-1, rank, rank))
    # A self-inverse q uses the convex mixture of a projected bath and its
    # reciprocal partner. This final screened-model reduction preserves
    # noise PSD/KMS and enforces crossing exactly, even at finite order.
    if self_negative:
        lp = np.asarray(lesser_poles)
        lr = np.asarray(lesser).reshape(-1, rank, rank)
        wc = kr.pole_residue_realization(np.concatenate((poles, -poles)),
            0.5 * np.concatenate((residues, -residues.conj())))
        wl = kr.pole_residue_realization(np.concatenate((lp, -lp, -poles)),
            0.5 * np.concatenate((lr, lr.conj(), residues.conj())))
        wr = (*wc[:3], np.eye(rank, dtype=complex))
        error = _lossless_response_check(wc, original_e, d, v, check_sites)
        if error > fit_tolerance:
            raise ValueError(f"GATE snapshot_bath_self_crossing_fit: response={error:.3e} "
                             f"exceeds {fit_tolerance:.3e}")
    receipt = dict(density_rank=rank, fit_order=pr[0].shape[0],
        projection_order=wc[0].shape[0] + wl[0].shape[0], bubble_check_error=error,
        screening_check_error=error, noise_check_error=normalization_error,
        fit_tolerance=float(fit_tolerance), fit_method="lossless_hermitian_rational_krylov_bath",
        training_refit_error=0.0, fit_pencil_shape=(0, 0),
        fit_pencil_bytes=basis_bytes, dense_workspace_bytes=dense_workspace_bytes,
        preliminary_transition_projection="rank_one_disjoint_energy_bin_Galerkin",
        original_transition_counts=[b["original_transition_count"] for b in branches],
        projected_transition_counts=[b["projected_transition_count"] for b in branches],
        screening_density_approximation="spectral_pinching_in_instantaneous_H",
        discarded_screening_coherence_norm=discarded_coherence,
        screening_population_roundoff_correction=population_roundoff,
        noise_approximation="same_subspace_weak_colored_bath",
        noise_check_kind="collective_lesser_greater_residue_psd",
        minimum_noise_residue=minimum, lag_decay_used=False,
        collective_normalization_error=normalization_error, zero_gap_noise="static_wick",
        zero_gap_statistics="aggregated_static_Wick_transition_reservoir",
        zero_gap_transition_count=int(np.count_nonzero(static)),
        zero_gap_unequal_population_count=int(np.count_nonzero(static & (d != 0))),
        zero_gap_equal_population_static_count=int(np.count_nonzero(static_exact)),
        zero_gap_response_rank=int(np.count_nonzero(static_live)),
        zero_gap_response_roundoff_residual=static_rank_error,
        zero_gap_discarded_roundoff_response_norm=float(np.linalg.norm(static_values[~static_live])),
        near_zero_unequal_population_count=int(np.count_nonzero(vertex_live & (d != 0)
            & (original_e != 0) & resolved_zero)),
        minimum_nonzero_transition_gap_ry=float(np.min(abs(original_e[vertex_live & (original_e != 0)]), initial=np.inf)),
        screening_gap_resolution_ry=gap_resolution,
        screening_gap_projection_count=projected_gap_count,
        maximum_screening_gap_adjustment_ry=gap_adjustment,
        retarded_poles_ry=np.concatenate((poles, -poles)) if self_negative else poles,
        lesser_order=wl[0].shape[0], noise_only_poles_allowed=True,
        self_negative_crossing=bool(self_negative),
        final_screened_model=("selfnegative_screened_model_convex_crossing" if self_negative
                              else "projected_transition_Dyson"))
    return dict(retarded=wr, correlation=wc, lesser=wl, receipt=receipt)
