"""Real SymMaps domains through actual source preparation; no GW/FFT solve."""
import copy
from types import SimpleNamespace as NS
import warnings

import runtime  # noqa: F401
import jax
import numpy as np
import pytest

from symmetry_maps import SymMaps, QirrOperator
from tdagw.snapshot_runtime import prepare_native_source


def source():
    grid, qgrid = (8, 8, 1), (4, 4, 1)
    coords = np.indices(grid).reshape(3, -1).T
    frac = coords/np.asarray(grid)
    facts = NS(trs_holds=False, ntran=1, kgrid=np.array(grid), nkpts=64,
        kpoints=frac, sym_matrices=np.eye(3, dtype=int)[None], shift=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fine = SymMaps(facts)
    coarse = fine.qgrid_view(qgrid, kgrid=grid)
    plan = NS(k_parent_frac=frac.copy(), irr_idx=np.arange(64), parent_full_rows=np.arange(64), sym=fine)
    # Only device placement, no compilation or full-matrix native provider.
    psi = jax.device_put(np.ones((64, 2, 2, 1), np.complex128))
    carrier = NS(plan=plan, projection_faces=lambda: (psi,))
    wfns = NS(green_parent=carrier, slices=NS(b0=0, b4_logical=2, b4_chi=2, b4_sigma=2),
        enk=np.broadcast_to([-.4, .3], (64, 2)), occ=np.broadcast_to([1., 0.], (64, 2)))
    meta = NS(kgrid=grid, qgrid=qgrid, q_sym=coarse, nspin=1, nspinor_wfnfile=2, n_rmu=1)
    V = QirrOperator.whole_zone(jax.device_put(np.ones((64, 1, 1), np.complex128)))
    value = dict(wfns=wfns, meta=meta, V_q=V,
        screening_inputs=dict(kwargs=dict(sym=fine)))
    return value, fine, coarse


def test_actual_prepare_prefers_same_q_view_as_general_builder_not_fine_kwargs():
    value, fine, coarse = source()
    assert len(fine.q_irr_full_idx) == 64 and len(coarse.q_irr_full_idx) == 16
    prepared = prepare_native_source(value, mesh=None)
    assert prepared["nk"] == 64 and prepared["nq"] == 16
    assert prepared["k_frac"].shape == (64, 3)
    assert prepared["flat_grid"]["k_order"] == "canonical-C"
    assert prepared["flat_grid"]["response_grid"] == [4, 4, 1]
    assert value["screening_inputs"]["kwargs"]["sym"] is fine


def test_actual_prepare_rejects_malformed_preferred_coarse_view_not_guard_waiver():
    value, _, coarse = source()
    wrong = copy.copy(coarse)
    wrong.q_irr_kgrid_int = np.roll(coarse.q_irr_kgrid_int, 1, axis=0)
    value["meta"].q_sym = wrong
    with pytest.raises(ValueError, match="canonical full coarse q"):
        prepare_native_source(value, mesh=None)


def test_actual_prepare_independently_rejects_bad_electronic_order_with_good_coarse_view():
    value, _, _ = source()
    plan = value["wfns"].green_parent.plan
    plan.k_parent_frac = np.roll(plan.k_parent_frac, 1, axis=0)
    with pytest.raises(ValueError, match="parent k coordinates are not in canonical"):
        prepare_native_source(value, mesh=None)
