"""Frozen-static screened finite-q crossing block.

This module is deliberately a row-at-a-time builder.  Its only dense
``N_mu x N_mu`` input is the already distributed static remainder solve and
its only objects of that class that survive are the ordinary pole slabs the
MPA store already owns.  The pair-space eigenproblem is replicated; no rank
gathers a centroid-space matrix.

For ``chi1 = P D(z) P^H`` and ``W0bar = W0(0)`` it constructs

``H = diag(u**2) + 2 diag(w) P^H W0bar P``

and the exact modal expansion of

``W0bar P (H-z**2 I)^-1 diag(-2w) P^H W0bar``.

Contiguous residue-weighted clusters are then collapsed element by element.
The two stored moments are the exact static value and the exact ``z^-2``
coefficient.  Consequently each compressed element has

``Cbar = sum C_m`` and ``Omega_bar**2 = Cbar / sum(C_m/lambda_m)``;
``Bbar = -Cbar/(2 Omega_bar)`` is exactly the pole-store convention.  The
imaginary part selected by this complex second moment is the block's
intrinsic Landau width; no executor eta enters here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


MODEL = "intraband_eigenmode_v1"

# A numerical acceptance, not an input or a physical parameter.  It is the
# design's stated ~4e-3 ceiling for the known block on its stamped support.
SAMPLE_REL_TOL = 4.0e-3
STATIC_REL_TOL = 2.0e-11
MIN_CLUSTERS = 3
MAX_CLUSTERS = 6
# Claim 0282's dense eigenproblem is explicitly sized at N_s ~ 1--4k.
# Crossing this ceiling is not an invitation to allocate until the GPU dies:
# it means the construction's scaling premise is false for this q row.
MAX_DENSE_MODES = 4096

_MODE_KERNELS = {}


@dataclass(frozen=True)
class IntrabandRow:
    """One compressed wedge row, still in native distributed sharding."""

    Omega_p: jax.Array
    B_p: jax.Array
    n_poles: int
    n_modes: int
    sample_max_rel_error: float
    static_max_rel_error: float
    certified: bool
    folded_modes: int
    dropped_modes: int
    folded_elements: int
    dropped_elements: int
    cluster_width_max_ry: float


def _mesh_of(value, where):
    mesh = getattr(getattr(value, "sharding", None), "mesh", None)
    if mesh is None:
        raise ValueError(f"{where} requires a globally NamedSharded array")
    return mesh


def _host_replicated(value):
    """Host view of a deliberately replicated small array."""
    return np.asarray(jax.device_get(value.addressable_data(0)))


def _mode_kernel(mesh):
    cached = _MODE_KERNELS.get(id(mesh))
    if cached is not None:
        return cached
    w_shard = NamedSharding(mesh, P("x", "y"))
    px_shard = NamedSharding(mesh, P(None, "x"))
    py_shard = NamedSharding(mesh, P(None, "y"))
    rep1 = NamedSharding(mesh, P(None))
    left_shard = NamedSharding(mesh, P("x", None))
    right_shard = NamedSharding(mesh, P(None, "y"))

    @jax.jit
    def build(W0bar, u, w, p_x, p_y):
        # W0bar@P and P^H@W0bar remain sharded on their one centroid axis.
        Wp = jax.lax.with_sharding_constraint(
            W0bar @ jnp.transpose(p_y), left_shard)
        pW = jax.lax.with_sharding_constraint(
            jnp.conj(p_x) @ W0bar, right_shard)
        A = jax.lax.with_sharding_constraint(
            jnp.conj(p_x) @ Wp, NamedSharding(mesh, P(None, None)))
        H = jnp.diag(u * u) + (2.0 * w[:, None]) * A
        eigenvalues, X = jnp.linalg.eig(H)
        Xinv = jnp.linalg.inv(X)
        left = jax.lax.with_sharding_constraint(Wp @ X, left_shard)
        right = jax.lax.with_sharding_constraint(
            Xinv @ ((-2.0 * w)[:, None] * pW), right_shard)
        left_norm = jnp.sqrt(jnp.sum(jnp.abs(left) ** 2, axis=0))
        right_norm = jnp.sqrt(jnp.sum(jnp.abs(right) ** 2, axis=1))
        weights = jax.lax.with_sharding_constraint(
            left_norm * right_norm, rep1)
        return eigenvalues, left, right, weights

    _MODE_KERNELS[id(mesh)] = build
    return build


def _retarded_modes(eigenvalues, left, right, weights):
    """Choose the retarded sheet, preserving the static modal sum."""
    lam = np.asarray(eigenvalues, dtype=np.complex128)
    wt = np.asarray(weights, dtype=np.float64)
    roots = np.sqrt(lam)
    negative = roots.real < 0.0
    roots[negative] *= -1.0
    # A real-positive eigenvalue routinely returns a root with an O(eps)
    # imaginary part after a non-Hermitian eigensolve.  Snap that roundoff
    # to the real axis: an anomaly count is evidence, not a floating-point
    # sign-bit census.  Material upper-half-plane roots are still folded and
    # remain visible in the named count.
    root_scale = max(
        float(np.max(np.abs(roots))) if roots.size else 0.0, 1.0)
    sheet_tol = 64.0 * np.finfo(np.float64).eps * root_scale
    near_real = np.abs(roots.imag) <= sheet_tol
    roots[near_real] = roots[near_real].real + 0.0j
    upper = roots.imag > sheet_tol
    folded = int(np.count_nonzero(upper))
    roots[upper] = np.conj(roots[upper])
    folded_lambda = roots * roots

    floor = np.finfo(np.float64).eps * root_scale
    weight_scale = max(float(np.max(wt)) if wt.size else 0.0, 1.0)
    live = wt > np.finfo(np.float64).eps * weight_scale
    keep = np.isfinite(roots) & (roots.real > floor)
    material_drop = (~keep) & live
    if np.any(material_drop):
        # The design permits a named drop, but it may never be silent.  The
        # caller receives the count and its sample/static certificate will
        # fail if the lost residue was material.
        pass
    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        raise ValueError(
            "GATE intraband_retarded_sheet: no crossing eigenmode remains "
            "on Re Omega > 0, Im Omega <= 0")
    ratio = np.ones(lam.shape, dtype=np.complex128)
    changed = upper & (lam != 0.0)
    ratio[changed] = folded_lambda[changed] / lam[changed]
    # C -> C * lambda_fold/lambda keeps C/lambda, hence z=0, exact.
    right = right * jnp.asarray(ratio[:, None])
    return (
        roots[keep_idx],
        left[:, keep_idx],
        right[keep_idx, :],
        wt[keep_idx],
        folded,
        int(np.count_nonzero(~keep)),
    )


def _weighted_clusters(omega, weights, n_cluster):
    """Deterministic residue-weighted Lloyd clustering on complex Omega."""
    points = np.column_stack((omega.real, omega.imag))
    n = int(points.shape[0])
    k = int(n_cluster)
    if not 1 <= k <= n:
        raise ValueError(f"cluster count {k} is outside [1,{n}]")
    weight = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("intraband modal residue weights must be finite >= 0")
    if not np.any(weight > 0.0):
        weight = np.ones_like(weight)
    order = np.lexsort((points[:, 1], points[:, 0]))
    cumulative = np.cumsum(weight[order])
    targets = (np.arange(k, dtype=np.float64) + 0.5) * cumulative[-1] / k
    seeds = order[np.searchsorted(cumulative, targets, side="left")]
    centers = points[seeds].copy()
    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(32):
        distance = np.sum((points[:, None, :] - centers[None, :, :]) ** 2,
                          axis=2)
        updated = np.argmin(distance, axis=1).astype(np.int32)
        if np.array_equal(updated, labels):
            break
        labels = updated
        nearest = np.min(distance, axis=1)
        for group in range(k):
            members = labels == group
            if np.any(members):
                centers[group] = np.average(
                    points[members], axis=0, weights=weight[members])
            else:
                pick = int(np.argmax(weight * nearest))
                centers[group] = points[pick]
                labels[pick] = group
    # Stable pole order is increasing real center, then width.
    center_order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = np.empty(k, dtype=np.int32)
    remap[center_order] = np.arange(k, dtype=np.int32)
    return remap[labels]


def _compress(mesh, omega, left, right, weights, labels, n_cluster):
    """Per-element static/z^-2 two-moment compression."""
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    lam = jnp.asarray(omega * omega)
    Omega_rows, B_rows = [], []
    folded_elements = 0
    dropped_elements = 0
    cluster_widths = []
    for group in range(int(n_cluster)):
        idx_host = np.flatnonzero(labels == group)
        idx = jnp.asarray(idx_host, dtype=jnp.int32)
        l = left[:, idx]
        r = right[idx, :]
        C = jax.lax.with_sharding_constraint(l @ r, matrix_shard)
        C0 = jax.lax.with_sharding_constraint(
            (l / lam[idx][None, :]) @ r, matrix_shard)
        scale = jnp.maximum(jnp.max(jnp.abs(C0)), 1.0)
        live = jnp.abs(C0) > np.finfo(np.float64).eps * scale
        lambda_bar = jnp.where(live, C / C0, 1.0 + 0.0j)
        root = jnp.sqrt(lambda_bar)
        root = jnp.where(jnp.real(root) < 0.0, -root, root)
        fold = live & (jnp.imag(root) > 0.0)
        root = jnp.where(fold, jnp.conj(root), root)
        invalid = live & (
            ~jnp.isfinite(root) | (jnp.real(root) <= 0.0))
        active = live & ~invalid
        # Retarded folding changes lambda.  Re-form C from the exact static
        # moment so the z=0 anchor remains an identity even on an anomalous
        # element; the separately measured z^-2 error exposes the change.
        C_retarded = C0 * root * root
        root = jnp.where(active, root, 1.0 + 0.0j)
        B = jnp.where(active, -C_retarded / (2.0 * root), 0.0 + 0.0j)
        Omega_rows.append(jax.lax.with_sharding_constraint(root, matrix_shard))
        B_rows.append(jax.lax.with_sharding_constraint(B, matrix_shard))
        folded_elements += int(jax.device_get(jnp.count_nonzero(fold)))
        dropped_elements += int(jax.device_get(jnp.count_nonzero(invalid)))

        wg = np.asarray(weights)[idx_host]
        og = np.asarray(omega)[idx_host]
        if np.sum(wg) > 0.0:
            center = np.sum(wg * og) / np.sum(wg)
            cluster_widths.append(float(np.sqrt(
                np.sum(wg * np.abs(og - center) ** 2) / np.sum(wg))))
        else:
            cluster_widths.append(0.0)
    return (
        jax.lax.with_sharding_constraint(jnp.stack(Omega_rows), pole_shard),
        jax.lax.with_sharding_constraint(jnp.stack(B_rows), pole_shard),
        folded_elements,
        dropped_elements,
        max(cluster_widths, default=0.0),
    )


def evaluate_pole_sum(Omega_p, B_p, z):
    """Evaluate poles in the store's own ``2 Omega B/(z^2-Omega^2)`` form."""
    zc = jnp.asarray(complex(z), dtype=jnp.complex128)
    return jnp.sum(
        2.0 * Omega_p * B_p / (zc * zc - Omega_p * Omega_p), axis=0)


def _evaluate_modes(left, right, omega, z):
    lam = jnp.asarray(omega * omega)
    return (left / (lam - complex(z) ** 2)[None, :]) @ right


def _relative_error(model, exact):
    numerator = jnp.real(jnp.vdot(model - exact, model - exact))
    denominator = jnp.real(jnp.vdot(exact, exact))
    floor = np.finfo(np.float64).tiny
    return float(jax.device_get(jnp.sqrt(numerator / jnp.maximum(
        denominator, floor))))


def build_row(W0bar, pair_block, z_samples, *, sample_rel_tol=SAMPLE_REL_TOL):
    """Build and certify one frozen-static crossing row.

    The smallest residue-weighted cluster count meeting ``sample_rel_tol``
    is selected, starting at the design floor of three and ending at the
    fixed implementation ceiling of six.  The tolerance is an API keyword
    only for synthetic red/green twins; production calls never source it
    from configuration.
    """
    mesh = _mesh_of(W0bar, "build_row")
    u, w, vertices = pair_block
    p_x, p_y = vertices
    n_pair = int(u.shape[0])
    if n_pair == 0:
        raise ValueError("build_row is not called for the empty Gamma block")
    if n_pair > MAX_DENSE_MODES:
        h_gib = n_pair * n_pair * np.dtype(np.complex128).itemsize / 2**30
        raise ValueError(
            "GATE intraband_dense_eigenproblem_size: exact crossing "
            f"selection has {n_pair} modes, above the design ceiling "
            f"{MAX_DENSE_MODES}; H alone would occupy {h_gib:.3f} GiB "
            "before eigenvectors, inverse, residues, or solver workspace. "
            "Refusing rather than truncating the certified selection or "
            "attempting an uncertified dense eigensolve outside the "
            "design's priced size regime")
    eigenvalues, left, right, weights = _mode_kernel(mesh)(
        W0bar, u, w, p_x, p_y)
    eigen_host = _host_replicated(eigenvalues)
    weights_host = _host_replicated(weights)
    omega, left, right, weights_host, folded, dropped = _retarded_modes(
        eigen_host, left, right, weights_host)

    z = tuple(complex(value) for value in np.asarray(z_samples).reshape(-1))
    candidates = range(
        min(MIN_CLUSTERS, len(omega)),
        min(MAX_CLUSTERS, len(omega)) + 1,
    )
    selected = None
    for n_cluster in candidates:
        labels = _weighted_clusters(omega, weights_host, n_cluster)
        Om, Bp, fold_el, drop_el, width = _compress(
            mesh, omega, left, right, weights_host, labels, n_cluster)
        errors = [
            _relative_error(
                evaluate_pole_sum(Om, Bp, value),
                _evaluate_modes(left, right, omega, value),
            )
            for value in z
        ]
        static_error = _relative_error(
            evaluate_pole_sum(Om, Bp, 0.0j),
            _evaluate_modes(left, right, omega, 0.0j),
        )
        selected = (n_cluster, Om, Bp, max(errors, default=0.0),
                    static_error, fold_el, drop_el, width)
        if selected[3] <= float(sample_rel_tol) \
                and static_error <= STATIC_REL_TOL:
            break
    if selected is None:
        raise ValueError("GATE intraband_cluster_support: no live modes")
    n_cluster, Om, Bp, error, static_error, fold_el, drop_el, width = selected
    return IntrabandRow(
        Omega_p=Om,
        B_p=Bp,
        n_poles=int(n_cluster),
        n_modes=int(len(omega)),
        sample_max_rel_error=float(error),
        static_max_rel_error=float(static_error),
        certified=bool(error <= float(sample_rel_tol)
                       and static_error <= STATIC_REL_TOL),
        folded_modes=int(folded),
        dropped_modes=int(dropped),
        folded_elements=int(fold_el),
        dropped_elements=int(drop_el),
        cluster_width_max_ry=float(width),
    )


def pad_row(row, n_poles):
    """Pad a shorter certified q row with causal, exactly dark poles."""
    target = int(n_poles)
    if target < row.n_poles:
        raise ValueError(
            f"cannot pad {row.n_poles} intraband poles to {target}")
    if target == row.n_poles:
        return row.Omega_p, row.B_p
    n_mu = int(row.Omega_p.shape[-1])
    mesh = _mesh_of(row.Omega_p, "pad_row")
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    extra = target - row.n_poles
    sentinel = jnp.ones((extra, n_mu, n_mu), dtype=jnp.complex128)
    dark = jnp.zeros((extra, n_mu, n_mu), dtype=jnp.complex128)
    return (
        jax.lax.with_sharding_constraint(
            jnp.concatenate((row.Omega_p, sentinel), axis=0), sharding),
        jax.lax.with_sharding_constraint(
            jnp.concatenate((row.B_p, dark), axis=0), sharding),
    )


__all__ = [
    "IntrabandRow",
    "MAX_CLUSTERS",
    "MAX_DENSE_MODES",
    "MODEL",
    "SAMPLE_REL_TOL",
    "STATIC_REL_TOL",
    "build_row",
    "evaluate_pole_sum",
    "pad_row",
]
