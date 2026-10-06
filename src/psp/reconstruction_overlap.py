"""Explicit reconstruction metric and finite-WFN symmetric Lowdin convention.

U=[I,X](I+X^dagger X)^(-1/2) is isometric. The full reconstructed Gram
therefore follows from Pauli overlaps with raw atomic differences, without
four-spinor quadrature or a normalized-tail cutoff. Atomic spheres must
not overlap; the caller authenticates geometry and atomic source identity.

Lowdin is opt-in at the fitting owner. Retaining the original energy labels
after nonunitary band mixing defines an effective vertex model, not exact
all-electron eigenstate equivalence. One factor must rotate the smooth
carrier, all atomic coefficients, and every sample endpoint coherently.
"""
from __future__ import annotations

import hashlib
import numpy as np

from psp.augmentation_spinors import (
    atomic_pauli_fourier, build_pauli_fourier_cache,
    evaluate_pauli_fourier_cache, spinor_function_labels,
)


def _delta_radial(data):
    r = np.asarray(data['r'], dtype=np.float64)
    w = np.asarray(data['weights_dr'], dtype=np.float64)
    u = np.asarray(data['delta_u'], dtype=np.complex128)
    ell, kappa = np.asarray(data['l']), np.asarray(data['kappa'])
    spinor_function_labels(ell, kappa)
    if (r.ndim != 1 or len(r) < 2 or w.shape != r.shape
            or u.shape != (len(r), len(ell)) or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(w)) or not np.all(np.isfinite(u))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(w <= 0)):
        raise ValueError("invalid raw atomic correction radial arrays")
    return r, w, u, ell.astype(int), kappa.astype(int)


def _delta_identity(data):
    digest = hashlib.sha256()
    for name in ('r', 'weights_dr', 'delta_u', 'l', 'kappa'):
        value = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_delta_gram(data):
    """Raw Pauli B_ij=<delta_i|delta_j>, with exact angular selection.

    Radial data use u=rR. The returned constant-size block includes all mj
    members, and never mixes distinct kappa even when radial shapes overlap.
    """
    _, w, u, ell, kappa = _delta_radial(data)
    labels = spinor_function_labels(ell, kappa)
    opf, m2 = labels[:, 0], labels[:, 1]
    radial = (u.conj().T*w) @ u
    allowed = (kappa[opf, None] == kappa[None, opf]) & (m2[:, None] == m2[None, :])
    return radial[opf[:, None], opf[None, :]]*allowed


def build_delta_radial_cache(data, **controls):
    """Species raw-delta transform cache, authenticated against its arrays."""
    r, w, u, ell, kappa = _delta_radial(data)
    cache = build_pauli_fourier_cache(u/r[:, None], r, w, ell, kappa, **controls)
    cache['source_identity'] = _delta_identity(data)
    return cache


def atomic_delta_overlap_table(data, wavevectors_cart, *, center_cart,
                               cell_volume, normalized_rkb_source=False,
                               radial_cache=None):
    """Conjugated Fourier table for d_ni=<delta_i|psi_n>, (function,2,G).

    Cell-volume normalization is included exactly once. For smooth normalized
    RKB upper components, the inverse canonical r(K) restores the original
    Pauli source before projection. The delta is never dualized or normalized.
    """
    r, w, u, ell, kappa = _delta_radial(data)
    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("atomic delta overlap requires positive cell volume")
    if radial_cache is None:
        table = atomic_pauli_fourier(u/r[:, None], r, w, ell, kappa,
                                     wavevectors_cart, center_cart=center_cart)
    else:
        if (radial_cache.get('source_identity') != _delta_identity(data)
                or not np.array_equal(radial_cache['ell'], ell)
                or not np.array_equal(radial_cache['kappa'], kappa)):
            raise ValueError("atomic delta Fourier cache source mismatch")
        table = evaluate_pauli_fourier_cache(radial_cache, wavevectors_cart,
                                             center_cart=center_cart)
    table = table.conj()/np.sqrt(volume)
    if normalized_rkb_source:
        from common.bispinor_init import _normalized_rkb_factor
        table = table/np.asarray(_normalized_rkb_factor(wavevectors_cart))[None, None, :]
    return table


def reconstruction_gram_correction(coefficients, delta_overlaps, delta_gram):
    r"""One atom's full band-Gram correction, including the smooth residual.

    C and D have shape (...,band,function), D_ni=<delta_i|psi_n>, and
    B_ij=<delta_i|delta_j>. Returns D* C^T + C* D^T + C* B C^T in band-row
    convention. Stream atomic blocks; no orbital sample cloud is required.
    """
    c = np.asarray(coefficients, dtype=np.complex128)
    d = np.asarray(delta_overlaps, dtype=np.complex128)
    b = np.asarray(delta_gram, dtype=np.complex128)
    if (c.ndim < 2 or d.shape != c.shape or b.shape != (c.shape[-1],)*2
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(d))
            or not np.all(np.isfinite(b))):
        raise ValueError("reconstruction Gram requires paired coefficients and an atomic Gram block")
    _hermitian(b, 'atomic delta Gram')
    cross = np.einsum('...ni,...mi->...nm', d.conj(), c)
    return (cross+cross.conj().swapaxes(-1, -2)
            + np.einsum('...ni,ij,...mj->...nm', c.conj(), b, c, optimize=True))


def reconstruction_gram(source_gram, coefficients, delta_overlaps, delta_gram):
    """Add one atomic correction to the actual smooth-source Gram.

    Pass the previous accumulated Gram when streaming several nonoverlapping
    atoms. The source Gram is measured from source coefficients, not assumed
    to be identity, so physical source drift and zero padding remain visible.
    """
    source = np.asarray(source_gram, dtype=np.complex128)
    correction = reconstruction_gram_correction(coefficients, delta_overlaps, delta_gram)
    if source.shape != correction.shape or not np.all(np.isfinite(source)):
        raise ValueError("smooth-source and reconstructed band Gram shapes differ")
    _hermitian(source, 'smooth-source Gram')
    return source+correction


def _hermitian(value, name):
    if (value.ndim < 2 or value.shape[-1] != value.shape[-2]
            or not value.shape[-1] or not np.all(np.isfinite(value))):
        raise ValueError(f"{name} must be a finite nonempty square matrix")
    scale = np.maximum(1., np.max(np.abs(value), axis=(-2, -1)))
    error = np.max(np.abs(value-value.conj().swapaxes(-1, -2)), axis=(-2, -1))
    if np.any(error > 128*np.finfo(float).eps*value.shape[-1]*scale):
        raise ValueError(f"{name} is not Hermitian within floating-point accumulation tolerance")
    return error


def lowdin_factor(gram, *, physical_bands):
    """Explicit symmetric inverse-root and finite-WFN metric receipts.

    Physical bands form the leading block. Padded source rows/columns must
    be zero; they receive identity only in the factorization matrix and
    rotation factor. Nonpositive or numerically unresolved physical metrics
    refuse rather than truncate an orbital or choose a hidden rank threshold.
    Returned arrays have the same parent batch axes as the input Gram.
    """
    full = np.asarray(gram, dtype=np.complex128)
    hermitian_error = _hermitian(full, 'reconstructed band Gram')
    n = int(physical_bands)
    nb = full.shape[-1]
    if n != physical_bands or n <= 0 or n > nb:
        raise ValueError("physical band count must be a positive integer within the carrier")
    scale = max(1., float(np.max(np.abs(full))))
    tolerance = 128*np.finfo(float).eps*nb*scale
    if n < nb and (np.max(np.abs(full[..., n:, :])) > tolerance
                   or np.max(np.abs(full[..., :, n:])) > tolerance):
        raise ValueError("padded reconstruction Gram rows must remain zero")
    block = (full[..., :n, :n]+full[..., :n, :n].conj().swapaxes(-1, -2))/2
    eigenvalue, eigenvector = np.linalg.eigh(block)
    minimum, maximum = eigenvalue[..., 0], eigenvalue[..., -1]
    resolution = 128*np.finfo(float).eps*n*np.maximum(1., maximum)
    if np.any(minimum <= resolution):
        raise ValueError("reconstructed physical band metric is nonpositive or numerically unresolved")
    factor = np.broadcast_to(np.eye(nb, dtype=np.complex128), full.shape).copy()
    factor[..., :n, :n] = ((eigenvector*eigenvalue[..., None, :]**-.5)
                           @ eigenvector.conj().swapaxes(-1, -2))
    padded_metric = np.zeros_like(full)
    padded_metric[..., :n, :n] = block
    if n < nb:
        padded_metric[..., n:, n:] = np.eye(nb-n)
    restored = factor.conj().swapaxes(-1, -2) @ padded_metric @ factor
    defect = block-np.eye(n)
    diagonal = np.diagonal(defect, axis1=-2, axis2=-1)
    offdiagonal = defect.copy()
    index = np.arange(n)
    offdiagonal[..., index, index] = 0
    return dict(gram=full, inverse_sqrt=factor, physical_bands=n,
                eigenvalue_min=minimum, eigenvalue_max=maximum,
                condition_number=maximum/minimum,
                max_diagonal_defect=np.max(np.abs(diagonal), axis=-1),
                max_offdiagonal_defect=np.max(np.abs(offdiagonal), axis=(-2, -1)),
                hermitian_error=hermitian_error,
                factor_isometry_error=np.max(np.abs(restored-np.eye(nb)), axis=(-2, -1)),
                convention='full_wfn_lowdin_effective_vertex')


def rotate_band_rows(values, inverse_sqrt, *, band_axis=1, conjugated=False):
    r"""JAX rotation out_n=sum_m values_m A_mn, preserving caller sharding.

    Parent is axis zero; A is (parent,band,band), or a shared (band,band)
    matrix. Band axis may also be the last axis of complementary sample
    faces. Conjugated orbital faces require A* and explicitly set that flag.
    The caller owns compilation, layouts and donation; no host gather occurs.
    """
    import jax.numpy as jnp

    value, factor = jnp.asarray(values), jnp.asarray(inverse_sqrt)
    if value.ndim < 2 or int(band_axis) != band_axis or not -value.ndim <= band_axis < value.ndim:
        raise ValueError("band rotation requires a valid parent/band array axis")
    axis = int(band_axis) % value.ndim
    if axis == 0:
        raise ValueError("band rotation requires a parent axis and a distinct band axis")
    bands = value.shape[axis]
    if (factor.ndim not in (2, 3) or factor.shape[-2:] != (bands, bands)
            or (factor.ndim == 3 and factor.shape[0] != value.shape[0])):
        raise ValueError("band rotation factor disagrees with parent/band carrier shape")
    if conjugated:
        factor = factor.conj()
    row = jnp.moveaxis(value, axis, 1)
    if factor.ndim == 2:
        rotated = jnp.einsum('mn,pm...->pn...', factor, row)
    else:
        rotated = jnp.einsum('pmn,pm...->pn...', factor, row)
    return jnp.moveaxis(rotated, 1, axis)
