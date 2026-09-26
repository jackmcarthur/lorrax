"""Test helper: write a planted bank's line samples through the producer's selection.

``plant_line_samples(path, value, ...)`` takes ``value(z) -> (W, dW/ds)``, the planted operator and its
s-derivative at complex z as the bank's dense face stacks [nq, d, d] (packed charge operator, or the
native photon layout), and writes every fitted line sample off the imaginary axis as the production
producer does: ``LineSelection.select`` on W(z), then ``mirror`` on the minus-q partner W(-conj z) on
the ordered route, then one line write of the panels. The dense samples are the caller's writes.
"""
import numpy as np


def plant_line_samples(path, value, *, meta, recipe, identity, mesh, nq, ordered, bank=None,
                       execution="local"):
    from file_io.shared_pole_store import _bank_plan, line_panel_span, write_shared_pole_bank
    from gw.shared_pole_directions import _sample_point, charge_line_selection

    p0, p1 = line_panel_span(_bank_plan(recipe))
    if bank is None:
        selection = charge_line_selection(meta, mesh_xy=mesh, ordered=ordered, execution=execution, nq=nq)
    else:
        from gw.shared_pole_sectors import sector_line_selection
        selection = sector_line_selection(bank, meta, mesh_xy=mesh, execution=execution, nq=nq)
    header = None
    for sid in range(p0, p1):
        z = _sample_point(recipe, sid)
        lines = selection.select(sid, *value(z))
        if ordered:
            selection.mirror(sid, lines, *value(-np.conj(z)))
        header = write_shared_pole_bank(path, q_span=(0, nq), line=selection.panels(sid, lines),
                                        meta=meta, expected_identity=identity, mesh_xy=mesh)
    return header
