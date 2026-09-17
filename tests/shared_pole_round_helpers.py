"""Test helper: run the production round selection on planted samples and hand back one slot on the face.

``round_states(mesh, sample, recipe, ...)`` places ``sample(slot, sample_id) -> (Wc, dWc_ds)`` host
matrices for every fitted sample of every mesh rank's slot in batch layout, runs
``gw.shared_pole_directions.select_round_states`` over the round, and returns the chosen slot's
states as [1, n, r] face panels (a conjugate state's direction stays the same object as its O
panel), its counts [1, A] and role records; ``batch=True`` returns the whole round as selected
(batch layout, counts [P, A]). Ordered rounds exchange with ``partners`` (default: every slot is
its own partner, identity realization), which is exact for a q = -q plant.
"""
import numpy as np


def round_states(mesh, sample, recipe, *, n, eig, svd, extent, ordered=False, slot=0, real=None,
                 partners=None, tables=None, batch=False):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_directions import select_round_states
    from gw.shared_pole_local import batch_to_face, face_rows

    ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
    fit = [int(i) for i in recipe["fit_ids"]]
    lo, hi = min(fit), max(fit) + 1
    layout = NamedSharding(mesh, P(("x", "y")))
    put = lambda a: jax.make_array_from_callback(a.shape, layout, lambda idx: a[idx])
    w = np.zeros((ranks, hi - lo, n, n), np.complex128)
    d = np.zeros_like(w)
    for r in range(ranks):
        for i in range(lo, hi):
            w[r, i - lo], d[r, i - lo] = sample(r, i)
    exchange = None
    if ordered:
        slots = np.arange(ranks) if partners is None else np.asarray(partners)
        if tables is None:
            alpha = np.broadcast_to(np.arange(n, dtype=np.int32), (ranks, n)).copy()
            tables = (put(alpha), put(alpha.copy()), put(np.ones((ranks, n), np.complex128)))
        exchange = (slots, *tables)
    states, counts, roles = select_round_states(
        dict(Wc=put(w), dWc_ds=put(d)), recipe, sample_lo=lo, real=ranks if real is None else real,
        mesh_xy=mesh, eigh_plan=eig, svd_plan=svd, column_extent=extent, logical_n=n, ordered=ordered,
        exchange=exchange)
    if batch:
        return states, counts, roles
    to_face, take = batch_to_face(mesh), face_rows(mesh, (slot,))
    moved = {}

    def face(a):
        if id(a) not in moved:
            moved[id(a)] = (a, take(to_face(a)))
        return moved[id(a)][1]
    return ([(st[0], *(face(a) for a in st[1:])) for st in states], counts[slot:slot + 1], [roles[slot]])
