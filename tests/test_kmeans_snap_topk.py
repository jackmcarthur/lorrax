"""Independent host oracle for literal duplicate repair and capacity."""


def main():
    import itertools
    import numpy as np
    from centroid.kmeans_isdf import ensure_unique_centroids
    grid = (3, 4, 5)
    total = int(np.prod(grid))

    def check(chosen, weight, grid=grid):
        total = int(np.prod(grid))
        coordinates = np.asarray(list(itertools.product(*map(range, grid))))
        points = coordinates[chosen] / grid
        result = ensure_unique_centroids(points, grid, rho=weight)
        occupied = set(chosen)
        original = sorted(occupied)
        if len(original) == len(chosen):
            expected = coordinates[chosen]
        else:
            free = [i for i in range(total) if i not in occupied]
            if weight is not None:
                flat = weight.reshape(-1)
                free.sort(key=lambda i: (np.isnan(flat[i]),
                    -flat[i] if not np.isnan(flat[i]) else 0., i))
            extra = free[:len(chosen)-len(original)]
            expected = coordinates[original+extra]
        np.testing.assert_array_equal(np.rint(result*grid).astype(int), expected)
        assert len(np.unique(np.ravel_multi_index(
            np.rint(result*grid).astype(int).T, grid))) == len(result)

    weights = [None, np.zeros(grid), np.ones(grid),
        (np.arange(total) % 4).reshape(grid).astype(float),
        np.sin(np.arange(total)*.173).reshape(grid),
        np.where(np.arange(total) % 7, np.nan, np.inf).reshape(grid),
        np.full(grid, np.nan),
        np.resize(np.array([-np.inf, np.inf, -0., 0.]), total).reshape(grid)]
    cases = [list(range(8)), [8]*12, [0, 1, 2, 3]*4,
        list(range(total))+[0]*7, [0]*total,
        list(range(total-1))+[0]*5]
    for weight in weights:
        for chosen in cases:
            check(chosen, weight)
    # Occupied points must never supply replacements, even when they hold
    # the largest weights or every free point is tied at the cutoff.
    weight = np.zeros(grid)
    weight.flat[[0, 3, 59]] = 100.
    check([0, 3, 59]*8, weight)
    # The first scan block contains only occupied points; tied replacements
    # must come from the next block without retaining a full tie-index list.
    large_grid = (33, 33, 64)
    check(list(range(1 << 16))+[0]*4, np.zeros(large_grid), large_grid)
    # Strictly better points after the block boundary precede low-index ties.
    weight = np.zeros(large_grid)
    weight.flat[-8:] = 1.
    check([0, 3, len(weight.reshape(-1))-1]*8, weight, large_grid)
    print('KMEANS_SNAP_TOPK PASS: independent weights/ties/capacity oracle', flush=True)


if __name__ == '__main__':
    main()
