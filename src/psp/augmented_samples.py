"""Project atom-local reconstruction channels and correct ISDF orbital samples.

The reconstruction is linear on individual Pauli orbitals.  It remains
separable after the normalized RKB lift: psi_aug(mu)=psi_smooth(mu)+c_i
[U delta_phi_i](mu).  Atomic duals are formed from the authenticated OCEAN
pseudo partial waves, not from Kleinman--Bylander operator projectors.
"""
from __future__ import annotations

import numpy as np


def _dual_radial(data):
    """The single overlap inversion used by direct and cached Fourier duals."""
    r = np.asarray(data['r'], dtype=np.float64)
    w = np.asarray(data['weights_dr'], dtype=np.float64)
    ps = np.asarray(data['ps_u'], dtype=np.complex128)
    ell = np.asarray(data['l'], dtype=np.int32)
    kappa = np.asarray(data['kappa'], dtype=np.int32)
    if (r.ndim != 1 or ps.shape != (len(r), len(ell)) or w.shape != r.shape
            or kappa.shape != ell.shape or not np.all(np.isfinite(ps))
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(w))
            or np.any(r <= 0) or np.any(w <= 0) or np.any(np.diff(r) <= 0)):
        raise ValueError("invalid atomic radial projection data")
    dual = np.zeros_like(ps)
    for ki in np.unique(kappa):
        columns = np.flatnonzero(kappa == ki)
        waves = ps[:, columns]
        overlap = (waves.conj().T * w) @ waves
        if np.linalg.eigvalsh(overlap).min() <= 128*np.finfo(float).eps * np.trace(overlap).real:
            raise ValueError("atomic reconstruction pseudo dual has an unresolved null mode")
        dual[:, columns] = np.linalg.solve(overlap.T, waves.T).T
    return r, w, ell, kappa, dual


def _projection_identity(data):
    import hashlib
    digest = hashlib.sha256()
    for name in ('r', 'weights_dr', 'ps_u', 'l', 'kappa'):
        value = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_projection_radial_cache(data, *, momentum_max, momentum_points,
                                  relative_tolerance=1e-10, absolute_tolerance=1e-12,
                                  validation_points=64):
    """Tabulate species dual transforms once; independently check interpolation.

    Only the radial amplitude depends on |K|. Atom phases and spin harmonics
    are evaluated directly for each Bloch momentum. This replaces repeated
    native-radial by full-G Bessel matrices with a small reusable species
    table. The direct transform at deterministic off-grid momenta supplies
    a bounded interpolation receipt; solid observable convergence is separate.
    """
    from psp.augmentation_spinors import build_pauli_fourier_cache

    r, w, ell, kappa, dual = _dual_radial(data)
    cache = build_pauli_fourier_cache(dual/r[:, None], r, w, ell, kappa,
        momentum_max=momentum_max, momentum_points=momentum_points,
        relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
        validation_points=validation_points)
    cache['source_identity'] = _projection_identity(data)
    return cache


def atomic_projection_table(data, wavevectors_cart, *, center_cart,
                            cell_volume, normalized_rkb_source=False, radial_cache=None):
    r"""One atom's Fourier dual table, shape (N_function,2,N_G).

    ``data`` is a loaded atomic-reconstruction sidecar; its ps_u columns
    use u=rR and share the AE transformation.  The local pseudo overlap is
    inverted explicitly, even when a PCA produced nearly orthonormal OPFs.
    The returned conjugated table contracts directly against source c_G,
    giving c_i=<dual_i|psi_physical> in the physical radial convention.
    The table's 1/sqrt(Omega) is already the physical plane-wave orbital
    normalization; callers must not divide the resulting overlap again.

    With ``normalized_rkb_source=True``, the caller supplies the *upper*
    two components of the smooth normalized four-component carrier.  The
    table includes the inverse of its existing carrier owner's r(K), so
    projection still acts on the original Pauli orbital.  It does not
    normalize the atomic correction a second time.

    Optional ``radial_cache`` is an independently checked species transform
    from build_projection_radial_cache. Its source and momentum range are
    authenticated; no extrapolation is permitted.
    """
    from psp.augmentation_spinors import atomic_pauli_fourier, spinor_function_labels

    r, w, ell, kappa, dual = _dual_radial(data)
    K = np.asarray(wavevectors_cart, dtype=np.float64)
    center = np.asarray(center_cart, dtype=np.float64)
    volume = float(cell_volume)
    if (not np.isfinite(volume) or volume <= 0 or K.ndim != 2 or K.shape[1] != 3
            or not np.all(np.isfinite(K)) or center.shape != (3,)
            or not np.all(np.isfinite(center))):
        raise ValueError("invalid atomic projection data or cell volume")
    # Different kappa channels are angularly orthogonal.  Solve each radial
    # overlap block; never mix labels merely because radial shapes overlap.
    labels = spinor_function_labels(ell, kappa)
    if radial_cache is None:
        table = atomic_pauli_fourier(dual/r[:, None], r, w, ell, kappa, K,
                                     center_cart=center_cart)
    else:
        from psp.augmentation_spinors import evaluate_pauli_fourier_cache
        magnitude = np.linalg.norm(K, axis=1)
        if (radial_cache['source_identity'] != _projection_identity(data)
                or not np.array_equal(radial_cache['ell'], ell)
                or not np.array_equal(radial_cache['kappa'], kappa)
                or np.any(magnitude > radial_cache['momentum'][-1])):
            raise ValueError("atomic projection radial cache source or momentum range mismatch")
        table = evaluate_pauli_fourier_cache(radial_cache, K, center_cart=center)
    table = table.conj()/np.sqrt(volume)
    if table.shape != (len(labels), 2, len(K)):
        raise ValueError("atomic Fourier dual lost its spinor channel ordering")
    if normalized_rkb_source:
        from common.bispinor_init import _normalized_rkb_factor
        table = table / np.asarray(_normalized_rkb_factor(K))[None, None, :]
    return table


def make_atomic_projection(mesh_xy):
    r"""Band contraction c_i=sum_Gs P_i(Gs)c_n(Gs), one atom at a time.

    Returns a compiled function ``(source,table)->coefficients``.  Source is
    (parent,band,2,N_G), table (parent,N_function,2,N_G), G sharded over
    ('x','y').  Coefficients (parent,band,N_function) are replicated: one
    atom's constant-size channel count, never all atoms' projector stack.
    Stream parent/band tiles through this same program at a fixed width.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    def local(source, table):
        if (source.ndim != 4 or table.ndim != 4 or source.shape[0] != table.shape[0]
                or source.shape[2] != 2 or table.shape[2:] != source.shape[2:]):
            raise ValueError("atomic projection requires matching parent/spin/G axes")
        result = jnp.einsum('pnsg,pisg->pni', source, table)
        return jax.lax.psum(result, ('x', 'y'))

    return jax.jit(shard_map(local, mesh=mesh_xy,
        in_specs=(P(None, None, None, ('x', 'y')),) * 2,
        out_specs=P(), check_vma=False))


def make_sample_correction(mesh_xy, *, face_sharding):
    r"""Return the device contraction psi_aug=psi_smooth+c_i Udelta_phi_i.

    Smooth/output arrays use the caller's existing (parent,band,4,mu) face
    sharding.  Coefficients (parent,band,N_function) are one atom's small
    replicated projection result.  Corrections (N_function,4,mu) share the
    face's centroid partition.  No full-k or band-pair carrier is formed.
    Image phases (parent,mu) are exp(2pi i k.L) for the contributing
    periodic image of this atom, sharing the centroid partition.  The
    caller streams atoms; phases are not absorbed into atomic coefficients.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    if not isinstance(face_sharding, NamedSharding) or face_sharding.mesh != mesh_xy:
        raise ValueError("atomic sample correction requires the run's NamedSharding")
    spec = face_sharding.spec
    if len(spec) != 4 or spec[0] is not None or spec[2] is not None:
        raise ValueError("atomic sample correction requires (parent,band,spin,mu) face layout")
    rep = NamedSharding(mesh_xy, P())
    delta_sh = NamedSharding(mesh_xy, P(None, None, spec[3]))
    phase_sh = NamedSharding(mesh_xy, P(None, spec[3]))

    @jax.jit(in_shardings=(face_sharding, rep, delta_sh, phase_sh), out_shardings=face_sharding,
             donate_argnums=(0,))
    def correct(smooth, coefficients, delta, image_phases):
        if (smooth.ndim != 4 or smooth.shape[2] != 4 or coefficients.ndim != 3
                or coefficients.shape[:2] != smooth.shape[:2]
                or delta.shape != (coefficients.shape[-1], 4, smooth.shape[-1])
                or image_phases.shape != (smooth.shape[0], smooth.shape[-1])):
            raise ValueError("atomic sample correction carrier shapes disagree")
        return smooth + jnp.einsum('pni,ism,pm->pnsm', coefficients, delta, image_phases)

    return correct


def atomic_image_geometry(points_frac, center_frac, lattice_cart):
    """Nearest periodic atom image and its EXACT integer Bloch phase offset.

    Returns ``(relative_cart, image_lattice)`` each (N_point,3).  The
    image convention is r-R_A-L; the sample correction therefore carries
    exp(2pi i k.L).  A caller using compact support must independently show
    that only one image contributes and that its normalized-RKB tail is
    converged; this geometric helper does not impose a cutoff.
    """
    x = np.asarray(points_frac, dtype=np.float64)
    center = np.asarray(center_frac, dtype=np.float64)
    lattice = np.asarray(lattice_cart, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 3 or center.shape != (3,)
            or lattice.shape != (3, 3) or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(lattice))):
        raise ValueError("invalid atomic sample geometry")
    singular = np.linalg.svd(lattice, compute_uv=False)
    if singular[-1] <= 128*np.finfo(float).eps*singular[0]:
        raise ValueError("atomic image geometry requires a nonsingular resolved lattice")
    # Start with the 27 neighbors of the nearest fractional integer.  The
    # smallest singular value certifies how far an unvisited image can be:
    # outside [-n,n]^3 its Cartesian distance is >= sigma_min*(n+1/2).
    # Expand only when this bound cannot yet certify the current closest
    # image; highly skew cells therefore never receive a wrong 27-cell map.
    first = np.rint(x-center).astype(np.int64)
    offsets = np.asarray([(i, j, k) for i in (-1, 0, 1)
                          for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
    images = first[:, None, :] + offsets[None, :, :]
    relative = np.einsum('pai,ij->paj', x[:, None, :]-center-images, lattice)
    distance2 = np.sum(relative*relative, axis=-1)
    chosen = np.argmin(distance2, axis=1)
    rows = np.arange(len(x))
    best_relative = relative[rows, chosen].copy()
    best_images = images[rows, chosen].copy()
    best_distance2 = distance2[rows, chosen].copy()
    if len(x) == 0:
        return best_relative, best_images
    needed = max(1, int(np.ceil(np.sqrt(best_distance2.max())/singular[-1] - 0.5)))
    for i in range(-needed, needed+1):
        for j in range(-needed, needed+1):
            for k in range(-needed, needed+1):
                if max(abs(i), abs(j), abs(k)) <= 1:
                    continue
                image = first + np.asarray((i, j, k), dtype=np.int64)
                candidate = (x-center-image) @ lattice
                norm2 = np.sum(candidate*candidate, axis=1)
                improve = norm2 < best_distance2
                best_relative[improve] = candidate[improve]
                best_images[improve] = image[improve]
                best_distance2[improve] = norm2[improve]
    return best_relative, best_images
