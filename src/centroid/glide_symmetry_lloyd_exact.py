#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


def wrap01(u: Array) -> Array:
    return np.mod(u, 1.0)


class GlideSymmetry2D:
    """2D affine symmetry group with identity and a glide reflection.

    Row-vector fractional convention:
        u' = u @ W[s].T + tau[s]   (mod 1)

    We use the glide
        g(x, y) = (x + 1/2, -y) mod 1.
    """

    def __init__(self) -> None:
        self.W = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, -1]],
        ], dtype=int)
        self.tau = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
        ], dtype=float)
        self.nsym = 2

    def apply(self, s: int, u: Array) -> Array:
        return wrap01(np.asarray(u, dtype=float) @ self.W[s].T + self.tau[s])

    def canonicalize(self, u: Array, tol: float = 1e-12) -> Array:
        u = wrap01(np.asarray(u, dtype=float))
        v = self.apply(1, u)
        if u[0] < v[0] - tol:
            return u
        if u[0] > v[0] + tol:
            return v
        if u[1] <= v[1] + tol:
            return u
        return v


@dataclass
class DemoConfig:
    grid_shape: tuple[int, int] = (24, 24)
    n_reps: int = 8
    n_iter: int = 25
    seed: int = 4
    n_init: int = 8
    sigma: float = 0.16
    tie_tol: float = 1.0e-12
    centroid_tol: float = 1.0e-13
    centroid_max_inner: int = 100
    image_search_nmax: int = 2
    lattice: Array = field(default_factory=lambda: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    center: Array = field(default_factory=lambda: np.array([0.23, 0.31], dtype=float))
    edge_center: Array = field(default_factory=lambda: np.array([0.00, 0.31], dtype=float))


@dataclass
class AssignResult:
    assign_mu: Array
    dmin: Array
    d2: Array
    shift_idx: Array
    disp: Array
    imgs: Array


def frac_to_cart(u: Array, lattice: Array) -> Array:
    return np.asarray(u) @ lattice


def metric_sq_from_frac(dfrac: Array, lattice: Array) -> Array:
    cart = np.asarray(dfrac) @ lattice
    return np.sum(cart * cart, axis=-1)


def make_grid(shape: tuple[int, int]) -> Array:
    n1, n2 = shape
    x = np.arange(n1) / n1
    y = np.arange(n2) / n2
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def make_integer_shifts(nmax: int) -> Array:
    vals = np.arange(-nmax, nmax + 1)
    sx, sy = np.meshgrid(vals, vals, indexing="ij")
    return np.stack([sx.ravel(), sy.ravel()], axis=1).astype(float)


def glide_pair_density(frac_pts: Array, lattice: Array, center: Array, sigma: float, sym: GlideSymmetry2D) -> Array:
    shifts = make_integer_shifts(1)
    c0 = np.asarray(center, dtype=float)
    c1 = sym.apply(1, c0)
    rho = np.zeros(len(frac_pts), dtype=float)
    for c in (c0, c1):
        base = frac_pts - c
        cand = base[:, None, :] + shifts[None, :, :]
        d2 = metric_sq_from_frac(cand, lattice)
        best = cand[np.arange(len(frac_pts)), np.argmin(d2, axis=1)]
        r2 = metric_sq_from_frac(best, lattice)
        rho += np.exp(-0.5 * r2 / sigma**2)
    return rho


def images_of_reps(reps: Array, sym: GlideSymmetry2D) -> Array:
    return np.stack([[sym.apply(s, r) for s in range(sym.nsym)] for r in reps], axis=0)


def best_periodic_displacements(points: Array, centers: Array, lattice: Array, shifts: Array):
    base = points[:, None, None, :] - centers[None, :, None, :] + shifts[None, None, :, :]
    d2_all = metric_sq_from_frac(base, lattice)
    shift_idx = np.argmin(d2_all, axis=2)
    ii = np.arange(len(points))[:, None]
    jj = np.arange(len(centers))[None, :]
    disp = base[ii, jj, shift_idx]
    d2 = d2_all[ii, jj, shift_idx]
    return disp, d2, shift_idx


def assign_points(grid: Array, reps: Array, sym: GlideSymmetry2D, lattice: Array, shifts: Array) -> AssignResult:
    n_grid = len(grid)
    n_reps = len(reps)
    imgs = images_of_reps(reps, sym)
    d2 = np.empty((n_grid, n_reps, sym.nsym), dtype=float)
    shift_idx = np.empty((n_grid, n_reps, sym.nsym), dtype=int)
    disp = np.empty((n_grid, n_reps, sym.nsym, 2), dtype=float)
    for mu in range(n_reps):
        disp_mu, d2_mu, idx_mu = best_periodic_displacements(grid, imgs[mu], lattice, shifts)
        d2[:, mu, :] = d2_mu
        shift_idx[:, mu, :] = idx_mu
        disp[:, mu, :, :] = disp_mu
    d2_orbit = d2.min(axis=2)
    assign_mu = np.argmin(d2_orbit, axis=1)
    dmin = d2_orbit[np.arange(n_grid), assign_mu]
    return AssignResult(assign_mu=assign_mu, dmin=dmin, d2=d2, shift_idx=shift_idx, disp=disp, imgs=imgs)


def objective(grid: Array, weights: Array, reps: Array, sym: GlideSymmetry2D, lattice: Array, shifts: Array) -> float:
    out = assign_points(grid, reps, sym, lattice, shifts)
    return float(np.sum(weights * out.dmin))


def cluster_objective(points: Array, weights: Array, rep: Array, sym: GlideSymmetry2D, lattice: Array, shifts: Array) -> float:
    imgs = np.array([sym.apply(s, rep) for s in range(sym.nsym)], dtype=float)
    _, d2, _ = best_periodic_displacements(points, imgs, lattice, shifts)
    return float(np.sum(weights * d2.min(axis=1)))


def torus_step_for_fixed_cluster(current_rep: Array, points: Array, weights: Array,
                                 sym: GlideSymmetry2D, lattice: Array, shifts: Array,
                                 tie_tol: float) -> Array:
    c = np.asarray(current_rep, dtype=float)
    imgs = np.array([sym.apply(s, c) for s in range(sym.nsym)], dtype=float)
    disp, d2, _ = best_periodic_displacements(points, imgs, lattice, shifts)
    dmin = d2.min(axis=1, keepdims=True)
    tie_mask = np.isclose(d2, dmin, atol=tie_tol, rtol=1e-10)
    alpha = tie_mask / tie_mask.sum(axis=1, keepdims=True)

    lifted_pts = []
    lifted_w = []
    for s in range(sym.nsym):
        # For row-vector convention and symmetric reflection matrix, inverse linear action is @ W[s]
        delta_back = disp[:, s, :] @ sym.W[s]
        ys = c + delta_back
        lifted_pts.append(ys)
        lifted_w.append(weights * alpha[:, s])

    Y = np.concatenate(lifted_pts, axis=0)
    W = np.concatenate(lifted_w, axis=0)
    keep = W > 0
    Y = Y[keep]
    W = W[keep]
    cand = np.sum(W[:, None] * Y, axis=0) / np.sum(W)
    return sym.canonicalize(cand)


def optimize_rep_for_fixed_cluster(current_rep: Array, points: Array, weights: Array,
                                   sym: GlideSymmetry2D, lattice: Array, shifts: Array,
                                   tie_tol: float, tol: float, max_inner: int):
    rep = np.asarray(current_rep, dtype=float).copy()
    energy_hist = [cluster_objective(points, weights, rep, sym, lattice, shifts)]
    for _ in range(max_inner):
        cand = torus_step_for_fixed_cluster(rep, points, weights, sym, lattice, shifts, tie_tol)
        e_cand = cluster_objective(points, weights, cand, sym, lattice, shifts)
        energy_hist.append(e_cand)
        delta = np.mod(cand - rep + 0.5, 1.0) - 0.5
        rep = cand
        if np.linalg.norm(delta) <= tol:
            break
    return rep, np.array(energy_hist)


def kmeanspp_init(grid: Array, weights: Array, n_reps: int, sym: GlideSymmetry2D,
                  lattice: Array, shifts: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    reps = []
    p = weights / weights.sum()
    reps.append(sym.canonicalize(grid[rng.choice(len(grid), p=p)]))
    while len(reps) < n_reps:
        d2 = np.full(len(grid), np.inf)
        for rep in reps:
            imgs = np.array([sym.apply(s, rep) for s in range(sym.nsym)], dtype=float)
            _, d2_mu, _ = best_periodic_displacements(grid, imgs, lattice, shifts)
            d2 = np.minimum(d2, d2_mu.min(axis=1))
        p = weights * d2
        if p.sum() <= 0:
            break
        for _ in range(1000):
            cand = sym.canonicalize(grid[rng.choice(len(grid), p=p / p.sum())])
            if not any(np.allclose(cand, r) for r in reps):
                reps.append(cand)
                break
        else:
            raise RuntimeError("kmeans++ initialization failed")
    return np.array(reps)


def update_reps(grid: Array, weights: Array, reps: Array, sym: GlideSymmetry2D,
                lattice: Array, shifts: Array, tie_tol: float,
                centroid_tol: float, centroid_max_inner: int):
    out = assign_points(grid, reps, sym, lattice, shifts)
    new_reps = reps.copy()
    cluster_histories = []
    masses = np.zeros(len(reps), dtype=float)
    for mu in range(len(reps)):
        mask = out.assign_mu == mu
        masses[mu] = weights[mask].sum()
        if not np.any(mask):
            cluster_histories.append(np.array([], dtype=float))
            continue
        pts = grid[mask]
        w = weights[mask]
        rep_mu, hist_mu = optimize_rep_for_fixed_cluster(
            reps[mu], pts, w, sym, lattice, shifts,
            tie_tol=tie_tol, tol=centroid_tol, max_inner=centroid_max_inner)
        new_reps[mu] = rep_mu
        cluster_histories.append(hist_mu)
    return new_reps, out.assign_mu, cluster_histories, masses


def run_demo(cfg: DemoConfig, center: Array) -> dict[str, object]:
    sym = GlideSymmetry2D()
    shifts = make_integer_shifts(cfg.image_search_nmax)
    grid = make_grid(cfg.grid_shape)
    rho = glide_pair_density(grid, cfg.lattice, center, cfg.sigma, sym)
    reps0 = kmeanspp_init(grid, rho, cfg.n_reps, sym, cfg.lattice, shifts, cfg.seed)

    reps_hist = [reps0.copy()]
    energy_hist = [objective(grid, rho, reps0, sym, cfg.lattice, shifts)]
    inner_histories = []
    assign = None
    masses = None
    reps = reps0.copy()
    for _ in range(cfg.n_iter):
        reps, assign, cluster_histories, masses = update_reps(
            grid, rho, reps, sym, cfg.lattice, shifts,
            tie_tol=cfg.tie_tol, centroid_tol=cfg.centroid_tol,
            centroid_max_inner=cfg.centroid_max_inner)
        reps_hist.append(reps.copy())
        energy_hist.append(objective(grid, rho, reps, sym, cfg.lattice, shifts))
        inner_histories.append(cluster_histories)

    return {
        "config": cfg,
        "sym": sym,
        "center": np.array(center, dtype=float),
        "glide_center": sym.apply(1, center),
        "shifts": shifts,
        "grid": grid,
        "rho": rho,
        "reps0": reps0,
        "reps": reps,
        "reps_hist": reps_hist,
        "energy_hist": np.array(energy_hist),
        "inner_histories": inner_histories,
        "assign": assign,
        "masses": masses,
    }


def run_best_of_seeds(cfg: DemoConfig, center: Array) -> dict[str, object]:
    if cfg.n_init <= 1:
        return run_demo(cfg, center)
    best_result = None
    best_energy = np.inf
    for i in range(cfg.n_init):
        trial_cfg = DemoConfig(
            grid_shape=cfg.grid_shape,
            n_reps=cfg.n_reps,
            n_iter=cfg.n_iter,
            seed=cfg.seed + i,
            n_init=1,
            sigma=cfg.sigma,
            tie_tol=cfg.tie_tol,
            centroid_tol=cfg.centroid_tol,
            centroid_max_inner=cfg.centroid_max_inner,
            image_search_nmax=cfg.image_search_nmax,
            lattice=np.array(cfg.lattice, copy=True),
            center=np.array(cfg.center, copy=True),
            edge_center=np.array(cfg.edge_center, copy=True),
        )
        result = run_demo(trial_cfg, center)
        e_final = float(result["energy_hist"][-1])
        if e_final < best_energy:
            best_energy = e_final
            best_result = result
    assert best_result is not None
    best_result["n_init_used"] = cfg.n_init
    return best_result


def run_basic_tests(result: dict[str, object]) -> None:
    cfg = result["config"]
    sym = result["sym"]
    shifts = result["shifts"]
    grid = result["grid"]
    rho = result["rho"]
    reps_hist = result["reps_hist"]
    energy_hist = result["energy_hist"]

    # canonicalization idempotent on generic and special points
    test_pts = np.array([[0.23, 0.31], [0.73, 0.69], [0.0, 0.0], [0.5, 0.0], [0.5, 0.25]])
    for p in test_pts:
        c1 = sym.canonicalize(p)
        c2 = sym.canonicalize(c1)
        assert np.allclose(c1, c2), f"Canonicalization not idempotent for {p}"

    e0 = objective(grid, rho, reps_hist[-1], sym, cfg.lattice, shifts)
    e1 = objective(grid, rho, np.array([sym.apply(1, r) for r in reps_hist[-1]]), sym, cfg.lattice, shifts)
    assert abs(e0 - e1) < 1e-12, "Objective should be invariant under glide unfolding"

    diffs = np.diff(energy_hist)
    assert np.all(diffs <= 1e-11), f"Energy is not monotone: max increase = {diffs.max()}"


def summarize(result: dict[str, object], label: str) -> str:
    cfg = result["config"]
    energy_hist = result["energy_hist"]
    reps = result["reps"]
    masses = result["masses"]
    center = result["center"]
    gcenter = result["glide_center"]
    n_init_used = int(result.get("n_init_used", 1))
    lines = [
        f"{label}: glide symmetry-adapted exact weighted Lloyd demo",
        f"grid                    : {cfg.grid_shape[0]} x {cfg.grid_shape[1]}",
        f"representatives stored  : {cfg.n_reps}",
        f"generic unfolded count  : {2 * cfg.n_reps}",
        f"image search nmax       : {cfg.image_search_nmax}",
        f"n_init                  : {n_init_used}",
        f"density centers         : {center} and {gcenter}",
        f"initial energy          : {energy_hist[0]:.12f}",
        f"final energy            : {energy_hist[-1]:.12f}",
        f"energy drop             : {energy_hist[0] - energy_hist[-1]:.12f}",
        f"orbit masses min/max    : {masses.min():.6f} / {masses.max():.6f}",
        "final representatives (fractional):",
        np.array2string(reps, precision=6, suppress_small=True),
    ]
    return "\n".join(lines)


def plot_results(results: list[dict[str, object]], labels: list[str], outpath: str) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 5 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = np.array([axes])
    for row, (result, label) in enumerate(zip(results, labels)):
        cfg = result["config"]
        sym = result["sym"]
        grid = result["grid"]
        rho = result["rho"]
        reps0 = result["reps0"]
        reps = result["reps"]
        E = result["energy_hist"]
        masses = result["masses"]
        center = result["center"]
        gcenter = result["glide_center"]

        cell = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float)
        cell_cart = frac_to_cart(cell, cfg.lattice)
        grid_cart = frac_to_cart(grid, cfg.lattice)
        imgs0 = frac_to_cart(images_of_reps(reps0, sym).reshape(-1, 2), cfg.lattice)
        imgs = frac_to_cart(images_of_reps(reps, sym).reshape(-1, 2), cfg.lattice)
        cc = frac_to_cart(np.array([center, gcenter]), cfg.lattice)

        ax = axes[row, 0]
        sc = ax.scatter(grid_cart[:, 0], grid_cart[:, 1], c=rho, s=24, linewidths=0, cmap='viridis')
        ax.plot(cell_cart[:, 0], cell_cart[:, 1], color='k', linewidth=1.2)
        ax.scatter(imgs0[:, 0], imgs0[:, 1], marker='x', s=75, linewidths=2.0, color='red')
        ax.scatter(cc[:, 0], cc[:, 1], marker='+', s=140, linewidths=2.2, color='white')
        ax.set_title(f"{label}: initial")
        ax.set_aspect('equal')
        fig.colorbar(sc, ax=ax, shrink=0.85, label='weight')

        ax = axes[row, 1]
        sc = ax.scatter(grid_cart[:, 0], grid_cart[:, 1], c=rho, s=24, linewidths=0, cmap='viridis')
        ax.plot(cell_cart[:, 0], cell_cart[:, 1], color='k', linewidth=1.2)
        ax.scatter(imgs[:, 0], imgs[:, 1], marker='x', s=95, linewidths=2.2, color='red', label='glide images')
        ax.scatter(imgs[:, 0], imgs[:, 1], marker='o', s=55, facecolors='none', edgecolors='black', linewidths=1.2)
        ax.scatter(cc[:, 0], cc[:, 1], marker='+', s=140, linewidths=2.2, color='white', label='density centers')
        ax.set_title(f"{label}: final\nE: {E[0]:.4f} → {E[-1]:.4f}, masses {masses.min():.3f} / {masses.max():.3f}")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        fig.colorbar(sc, ax=ax, shrink=0.85, label='weight')
    fig.suptitle('2D glide symmetry-adapted exact weighted Lloyd demos', fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = DemoConfig()
    center_result = run_best_of_seeds(cfg, cfg.center)
    edge_result = run_best_of_seeds(cfg, cfg.edge_center)
    run_basic_tests(center_result)
    run_basic_tests(edge_result)
    print(summarize(center_result, 'generic center'))
    print()
    print(summarize(edge_result, 'edge center'))
    plot_results([center_result, edge_result], ['generic center', 'edge center'], '/mnt/data/glide_symmetry_lloyd_exact.png')
    print('\nWrote /mnt/data/glide_symmetry_lloyd_exact.png')


if __name__ == '__main__':
    main()
