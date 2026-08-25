"""Real-space image geometry for Fourier interpolation on a k mesh.

The flat-k FFT remains owned by :mod:`common.fft_helpers`.  Its inverse
transform returns one coefficient for each residue class
``R mod diag(kgrid)``.  This module owns the separate interpolation question:
which lattice-equivalent images represent that residue class between the
coarse nodes?

For each residue ``r`` the Wigner--Seitz plan keeps every physically shortest
image ``R = r + diag(kgrid) l`` and divides the residue coefficient equally
among them.  At an unshifted coarse point ``k_i = m_i / N_i`` all those images
have the same phase, because ``exp(-2 pi i m.l) = 1``.  The plan therefore
changes only off-grid interpolation and preserves every coarse node exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


WS_DISTANCE_RTOL = 1.0e-10
WS_DISTANCE_ATOL = 1.0e-12
WS_MAX_SEARCH_SHELL = 16


@dataclass(frozen=True)
class WignerSeitzPhasePlan:
    """Host-built, dense-padded nearest-image plan in flat-FFT residue order."""

    images: np.ndarray       # (nk, max_degeneracy, 3), integer lattice coords
    weights: np.ndarray      # (nk, max_degeneracy), 1 / degeneracy or zero pad
    degeneracies: np.ndarray # (nk,)
    canonical_changed: int   # classes whose centered FFT representative is absent
    redistributed: int       # changed or multiply represented classes
    search_shell: int        # largest certified integer-translation shell


def build_R_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Centered representatives in the C-order used by flat-k FFT helpers."""

    shape = tuple(int(n) for n in kgrid)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"kgrid must contain three positive extents, got {kgrid!r}")

    def _shift(n: int) -> np.ndarray:
        a = np.arange(n, dtype=np.int64)
        return np.where(a >= (n + 1) // 2, a - n, a)

    return np.stack(
        np.meshgrid(*(_shift(n) for n in shape), indexing="ij"), axis=-1
    ).reshape(-1, 3)


def build_uniform_k_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Unshifted fractional coarse points in canonical flat-k FFT order."""

    shape = tuple(int(n) for n in kgrid)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"kgrid must contain three positive extents, got {kgrid!r}")
    axes = tuple(np.arange(n, dtype=np.float64) / float(n) for n in shape)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _translation_cube(shell: int) -> np.ndarray:
    axis = np.arange(-shell, shell + 1, dtype=np.int64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def build_wigner_seitz_phase_plan(
    kgrid: tuple[int, int, int],
    avec: np.ndarray,
    *,
    distance_rtol: float = WS_DISTANCE_RTOL,
    distance_atol: float = WS_DISTANCE_ATOL,
    max_search_shell: int = WS_MAX_SEARCH_SHELL,
) -> WignerSeitzPhasePlan:
    """Return equal-weight shortest images for every flat-FFT residue class.

    ``avec`` has direct-lattice vectors as rows, matching ``WfnLoader.avec``.
    Search termination is certified rather than inferred from a minimizer not
    touching the current cube: for ``B = diag(kgrid) @ avec``, any translation
    outside shell ``s`` obeys

    ``|r @ avec + l @ B| >= sigma_min(B) * (s + 1) - |r @ avec|``.

    The search stops only when that lower bound cannot tie or beat the best
    image already present.  This keeps the host geometry correct for oblique
    cells without hard-coding the ``[-1, 1]^3`` convention used by common
    high-symmetry examples.
    """

    shape = tuple(int(n) for n in kgrid)
    residues = build_R_grid_np(shape)
    lattice = np.asarray(avec, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"avec must be a finite (3, 3) row-vector matrix, got {lattice.shape}")
    supercell = np.asarray(shape, dtype=np.float64)[:, None] * lattice
    sigma_min = float(np.linalg.svd(supercell, compute_uv=False)[-1])
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError("Wigner-Seitz interpolation requires a nonsingular direct lattice")
    if distance_rtol < 0.0 or distance_atol < 0.0:
        raise ValueError("Wigner-Seitz distance tolerances must be non-negative")
    if max_search_shell < 1:
        raise ValueError("max_search_shell must be at least one")

    nearest: list[np.ndarray] = []
    largest_shell = 0
    for residue in residues:
        residue_cart = residue @ lattice
        residue_norm = float(np.linalg.norm(residue_cart))
        chosen = None
        for shell in range(1, int(max_search_shell) + 1):
            translations = _translation_cube(shell)
            candidates = residue[None, :] + translations * np.asarray(shape)[None, :]
            cart = candidates @ lattice
            distance2 = np.einsum("ri,ri->r", cart, cart)
            d2_min = float(np.min(distance2))

            # An image outside this translation cube has ||l||_2 >= shell+1.
            # Require strict separation including the same tolerance used to
            # collect degenerate shortest images.
            outside_lower = max(0.0, sigma_min * float(shell + 1) - residue_norm)
            tied_upper = d2_min * (1.0 + distance_rtol) + distance_atol
            if outside_lower * outside_lower > tied_upper:
                mask = np.isclose(
                    distance2, d2_min, rtol=distance_rtol, atol=distance_atol
                )
                chosen = candidates[mask]
                largest_shell = max(largest_shell, shell)
                break
        if chosen is None:
            raise RuntimeError(
                "Wigner-Seitz image search was not certified within shell "
                f"{max_search_shell} for residue {residue.tolist()} on kgrid {shape}; "
                "increase WS_MAX_SEARCH_SHELL only after inspecting the cell metric"
            )
        nearest.append(np.asarray(chosen, dtype=np.int64))

    degeneracies = np.asarray([len(images) for images in nearest], dtype=np.int32)
    max_degeneracy = int(np.max(degeneracies))
    images_padded = np.zeros((len(nearest), max_degeneracy, 3), dtype=np.float64)
    weights = np.zeros((len(nearest), max_degeneracy), dtype=np.float64)
    canonical_changed = 0
    redistributed = 0
    for idx, (residue, images) in enumerate(zip(residues, nearest, strict=True)):
        count = int(images.shape[0])
        images_padded[idx, :count] = images
        weights[idx, :count] = 1.0 / float(count)
        canonical_present = bool(np.any(np.all(images == residue[None, :], axis=1)))
        canonical_changed += int(not canonical_present)
        redistributed += int((not canonical_present) or count > 1)

    return WignerSeitzPhasePlan(
        images=images_padded,
        weights=weights,
        degeneracies=degeneracies,
        canonical_changed=canonical_changed,
        redistributed=redistributed,
        search_shell=largest_shell,
    )


def effective_fourier_phase(
    q_frac: jax.Array, images: jax.Array, weights: jax.Array
) -> jax.Array:
    """Traceable ``sum_image weight * exp(-2 pi i q.R_image)`` per residue.

    ``q_frac`` may have any leading batch shape ending in three.  ``images``
    and ``weights`` are the dense-padded arrays from
    :func:`build_wigner_seitz_phase_plan`; zero-weight pad images contribute
    nothing.  The returned final axis is the canonical flat-FFT residue axis.
    """

    phase = jnp.exp(
        -2j * jnp.pi * jnp.einsum("...a,rda->...rd", q_frac, images)
    )
    return jnp.sum(phase * weights, axis=-1)
