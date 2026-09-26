"""Test helper: run the production selection on planted samples and hand back one slot on the face.

``round_states(mesh, sample, recipe, ...)`` places ``sample(slot, sample_id) -> (Wc, dWc_ds)`` host
matrices for every fitted sample of every mesh rank's slot in batch layout. A fitted line sample off
the imaginary axis goes the producer's way: ``line_sample_states`` selects its directions from that
sample alone, ordered rounds add the mirror states from ``partner(slot, sample_id) -> (W(-conj z),
dW/ds(-conj z))``, and the states round-trip through their stored panel form
(``line_panel_states``). The dense samples then go through
``gw.shared_pole_directions.select_round_states`` over the round, and the chosen slot's states come
back as [1, n, r] face panels (a conjugate state's direction stays the same object as its O panel),
with its counts [1, A] and role records; ``batch=True`` returns the whole round as selected (batch
layout, counts [P, A]).
"""
import numpy as np


def round_states(mesh, sample, recipe, *, n, eig, svd, extent, ordered=False, slot=0, real=None,
                 partner=None, batch=False):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_directions import (_round_kernels, _sample_point, line_panel_states,
                                           line_sample_mirrors, line_sample_states, select_round_states)
    from gw.shared_pole_local import batch_to_face, face_rows

    ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
    real = ranks if real is None else real
    fit = [int(i) for i in recipe["fit_ids"]]
    line = [i for i in fit if _sample_point(recipe, i).real != 0]
    dense = [i for i in fit if i not in line]
    layout = NamedSharding(mesh, P(("x", "y")))
    put = lambda a: jax.make_array_from_callback(a.shape, layout, lambda idx: a[idx])

    def stack(ids, source):
        w = np.zeros((ranks, len(ids), n, n), np.complex128)
        d = np.zeros_like(w)
        for r in range(ranks):
            for k, i in enumerate(ids):
                w[r, k], d[r, k] = source(r, i)
        return put(w), put(d)

    kernels = _round_kernels(mesh, "batch")
    line_states = {}
    for sid in line:
        W, dW = stack([sid], sample)
        selected = line_sample_states(W, dW, recipe, sid=sid, ordered=ordered, real=real, mesh_xy=mesh,
                                      eigh_plan=eig, svd_plan=svd, column_extent=extent, logical_n=n)
        mirrors = (line_sample_mirrors(selected, stack([sid], partner), recipe, sid=sid, mesh_xy=mesh)
                   if ordered else [])
        states = selected["states"] + mirrors
        panels = kernels.stack(states[0][1], *[a for st in states for a in st[2:]])
        line_states[sid] = line_panel_states(panels, selected["counts"], recipe, sid=sid,
                                             ordered=ordered, mesh_xy=mesh)
    W, dW = stack(dense, sample)
    states, counts, roles = select_round_states(
        dict(Wc=W, dWc_ds=dW), recipe, sample_ids=dense, real=real, mesh_xy=mesh, eigh_plan=eig,
        svd_plan=svd, column_extent=extent, logical_n=n, ordered=ordered, line_states=line_states)
    if batch:
        return states, counts, roles
    to_face, take = batch_to_face(mesh), face_rows(mesh, (slot,))
    moved = {}

    def face(a):
        if id(a) not in moved:
            moved[id(a)] = (a, take(to_face(a)))
        return moved[id(a)][1]
    return ([(st[0], *(face(a) for a in st[1:])) for st in states], counts[slot:slot + 1], [roles[slot]])
