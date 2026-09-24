"""The density scan's three routes against an independent host density.

``gw.qsgw_density.rho_from_wfns`` plans one of three per-rank routes
(``g_split``, ``band_2d``, ``local``), may tile k, and transforms only the
occupied rotated bands.  Every arm here is compared with a NumPy density that
shares no code with the scan: host rotation, host box scatter, ``numpy.fft``
and the dense Dirac alpha matrices built from the gamma table.  The fixture's
G extent (37) is not divisible by the mesh, its occupations are fractional,
signed and have an FD tail, and its sphere is large enough against the
box (12 k, 37 G, 4x4x4) that the planner tiles k.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding                   # noqa: E402
from jax.sharding import PartitionSpec as P                    # noqa: E402

from common.wfn_layout import band_sphere_spec                 # noqa: E402
from gw import qsgw_density                                    # noqa: E402
from gw.qsgw_density import (                                  # noqa: E402
    DENSITY_TAIL_ELECTRON_TOL, band_rotation_spec,
    density_active_band_count, plan_density_scan, rho_from_wfns)

RTOL = 1.0e-12
NK, NB, NG = 12, 16, 37
GRID = (4, 4, 4)
VOLUME = 23.0
#: Identity star table: the random weights are non-uniform, and the scan
#: refuses a reduced k-set without one.
IDENT = np.arange(int(np.prod(GRID)), dtype=np.int32)[None]


def _mesh(count: int = 4) -> Mesh:
    devices = jax.devices()
    side = int(np.sqrt(count)) if len(devices) >= count else 1
    return Mesh(np.asarray(devices[:side * side]).reshape(side, side),
                ("x", "y"))


def _put(value, mesh, spec):
    return jax.make_array_from_callback(
        value.shape, NamedSharding(mesh, spec), lambda index: value[index])


def _haar(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))[None]


def _fixture(ns):
    rng = np.random.default_rng(20260923 + ns)
    psi = (rng.standard_normal((NK, NB, ns, NG))
           + 1j * rng.standard_normal((NK, NB, ns, NG)))
    ngrid = int(np.prod(GRID))
    bidx = np.zeros((NK, NG), dtype=np.int32)      # per-k sphere index
    cells = []
    for ik in range(NK):
        c = rng.choice(ngrid, size=NG, replace=False)
        xyz = np.column_stack(np.unravel_index(c, GRID))
        bidx[ik] = c
        cells.append(xyz)
    U = np.stack([_haar(rng, NB) for _ in range(NK)])
    # FD-like: three full-ish bands, a fractional fourth, an FD tail that
    # the cut drops (<1e-15 per state), exact zeros above, and one signed
    # MP1-like weight.  Rotated band order is the occupation order.
    occ = np.zeros((NK, NB))
    for ik in range(NK):
        e = np.sort(rng.uniform(-1.0, 1.0, NB)) + np.arange(NB) * 0.5
        occ[ik] = 1.0 / (1.0 + np.exp((e - e[3]) / 0.01))
    occ[:, 7:] = 0.0
    occ[2, 1] = -0.03
    kw = rng.uniform(0.5, 1.5, NK)
    return psi.astype(np.complex128), bidx, cells, U, occ, kw / kw.sum()


def _alpha_matrices():
    from common.gamma_matrices import gamma_perm_phase
    mats = []
    for mu in (1, 2, 3):
        perm, phase = (np.asarray(v) for v in gamma_perm_phase(mu))
        a = np.zeros((4, 4), dtype=np.complex128)
        a[np.arange(4), perm] = phase
        mats.append(a)
    return mats


def _host_density(psi, U, occ, kw, bidx_cells, *, current, charge_ns):
    """NumPy only: rotate, scatter, ifftn, contract.  Shares no scan code."""
    rot = psi if U is None else np.einsum("kmn,kmsg->knsg", U, psi)
    ns = psi.shape[2]
    ngrid = int(np.prod(GRID))
    scale = np.sqrt(ngrid / VOLUME)
    alphas = _alpha_matrices() if current else []
    out = np.zeros(((4,) if current else ()) + GRID)
    for ik, xyz in enumerate(bidx_cells):
        box = np.zeros((NB, ns, *GRID), dtype=np.complex128)
        box[:, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = rot[ik]
        r = np.fft.ifftn(box, axes=(-3, -2, -1), norm="ortho") * scale
        f = kw[ik] * occ[ik][:, None, None, None]
        rho = np.sum(f * np.sum(np.abs(r[:, :charge_ns]) ** 2, axis=1),
                     axis=0)
        if not current:
            out += rho
            continue
        out[0] += rho
        for i, a in enumerate(alphas):
            j = np.einsum("ns...,st,nt...->n...", np.conj(r), a, r).real
            out[1 + i] += np.sum(f * j, axis=0)
    return out


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - b))
                 / max(float(np.max(np.abs(b))), 1e-300))


@pytest.mark.mesh(4)
@pytest.mark.parametrize("current", [False, True],
                         ids=["charge", "four_current"])
def test_every_route_matches_the_independent_host_density(current,
                                                          monkeypatch):
    """g_split (tiled k), band_2d and local agree with NumPy at 1e-12.

    The route predicate is asserted from the plan the call used, so a
    fixture that silently fell back to another route cannot pass as this
    one (TASTE 30).
    """
    mesh = _mesh()
    ns = 4 if current else 2
    psi, bidx, cells, U, occ, kw = _fixture(ns)
    charge_ns = 2
    psi_j = _put(psi, mesh, band_sphere_spec())
    U_j = _put(U, mesh, band_rotation_spec())
    # A reduced (non-uniform) k-set needs a star table, and the current
    # additionally a SymMaps: the charge arm carries the weights, the
    # current arm the uniform full-zone case.
    if current:
        kw = np.full(NK, 1.0 / NK)
    kw_args = dict(mesh=mesh, box_index=bidx, fft_grid=GRID,
                   cell_volume=VOLUME, spin_degeneracy=1.0,
                   sym_perm=None if current else IDENT,
                   include_dirac_current=current, charge_nspinor=charge_ns)
    ref = _host_density(psi, U, occ, kw, cells, current=current,
                        charge_ns=charge_ns)
    ref_none = _host_density(psi, None, occ, kw, cells, current=current,
                             charge_ns=charge_ns)

    p = int(mesh.devices.size)
    n_act = density_active_band_count(occ, kw, 1.0)
    assert n_act == 4, n_act          # the FD tail above band 4 is dropped
    plans = {}
    real_plan = qsgw_density.plan_density_scan

    def recording_plan(**kw_plan):
        plan = real_plan(**kw_plan)
        plans[plan.route] = plan
        return plan

    monkeypatch.setattr(qsgw_density, "plan_density_scan", recording_plan)
    got_g = rho_from_wfns(psi_j, occ, kw, U=U_j, **kw_args)
    got_b = rho_from_wfns(psi_j, occ, kw, U=U_j,
                          memory_budget_bytes=1.0, **kw_args)
    got_l = rho_from_wfns(psi_j, occ, kw, **kw_args)

    g = plans["g_split"]
    if p > 1:
        assert g.n_rot == max(p, 4) and g.n_rot < NB
        assert 1 < g.k_tile < NK and NK % g.k_tile == 0
        assert g.g_carrier % p == 0 and g.g_carrier >= NG
    assert plans["band_2d"].n_rot == NB and plans["band_2d"].k_tile == 1
    assert plans["local"].route == "local"
    tail = 1.0e-15 * NB                 # dropped FD mass, far below 1e-12
    assert _rel(got_g, ref) <= RTOL + tail
    assert _rel(got_b, ref) <= RTOL
    assert _rel(got_l, ref_none) <= RTOL


@pytest.mark.mesh(4)
def test_a_transposed_rotation_is_caught():
    """Red twin: U^T is unitary and mixes the same bands, yet must fail.

    Occupied-block invariance, the electron count and every norm survive a
    transposed U, so the parity above is only evidence if the SAME
    comparison rejects it.
    """
    mesh = _mesh()
    psi, bidx, cells, U, occ, _ = _fixture(4)
    kw = np.full(NK, 1.0 / NK)
    kw_args = dict(mesh=mesh, box_index=bidx, fft_grid=GRID,
                   cell_volume=VOLUME, spin_degeneracy=1.0,
                   include_dirac_current=True, charge_nspinor=2)
    ref = _host_density(psi, U, occ, kw, cells, current=True, charge_ns=2)
    wrong = rho_from_wfns(
        _put(psi, mesh, band_sphere_spec()), occ, kw,
        U=_put(np.ascontiguousarray(np.swapaxes(U, 1, 2)), mesh,
               band_rotation_spec()), **kw_args)
    assert _rel(wrong, ref) > 1.0e-3


def test_occupied_cut_budget_and_plan_buckets():
    """The cut is the window budget; padded counts are the only shape."""
    kw = np.full(4, 0.25)
    step = np.zeros((4, 12)); step[:, :5] = 1.0
    assert density_active_band_count(step, kw, 2.0) == 5
    fd = step.copy(); fd[:, 5] = 0.4 * DENSITY_TAIL_ELECTRON_TOL
    assert density_active_band_count(fd, kw, 1.0) == 5     # 0.4 tol dropped
    fd[:, 5] = 2.0 * DENSITY_TAIL_ELECTRON_TOL
    assert density_active_band_count(fd, kw, 1.0) == 6     # kept
    mp1 = step.copy(); mp1[1, 7] = -3.0e-3                  # signed weight
    assert density_active_band_count(mp1, kw, 1.0) == 8

    mesh = _mesh()
    p = int(mesh.devices.size)
    common = dict(mesh=mesh, n_k=6, nb_carrier=4 * p, ngkmax=37, ns=2,
                  n_grid=64, have_U=True, budget_bytes=72e9)
    a = plan_density_scan(n_active=1, **common)
    b = plan_density_scan(n_active=p, **common)
    assert a == b and a.route == "g_split" and a.n_rot == p
    assert plan_density_scan(n_active=p + 1, **common).n_rot == 2 * p
    big = plan_density_scan(n_active=1, **{**common, "budget_bytes": 1.0})
    assert big.route == "band_2d" and big.n_rot == 4 * p
    assert plan_density_scan(n_active=1, **{**common, "have_U": False}
                             ).route == "local"
    # A single k-point (a molecule) never tiles; a huge box never tiles.
    assert plan_density_scan(n_active=1, **{**common, "n_k": 1}).k_tile == 1
    assert plan_density_scan(n_active=1, **{**common, "n_grid": 10**7}
                             ).k_tile == 1


@pytest.mark.mesh(4)
def test_the_scan_body_moves_no_field_and_no_k_stack():
    """HLO: inside the while body no all-reduce, no k-stack payload.

    The incumbent's body held an all-reduce of the band-summed field and an
    all-gather of the WHOLE (n_k, nb/P, s, G) stack (the scan operand, not
    its slice) every trip.  After the change the field is reduced once,
    outside the loop, and every per-trip collective payload is at most one
    k tile of the sphere.
    """
    mesh = _mesh()
    if int(mesh.devices.size) < 4:
        pytest.skip("collective census needs a 2x2 mesh (TASTE 32)")
    psi, bidx, _, U, occ, _ = _fixture(4)
    kw = np.full(NK, 1.0 / NK)
    args = dict(mesh=mesh, box_index=bidx, fft_grid=GRID, cell_volume=VOLUME,
                spin_degeneracy=1.0, include_dirac_current=True,
                charge_nspinor=2)
    psi_j = _put(psi, mesh, band_sphere_spec())
    U_j = _put(U, mesh, band_rotation_spec())
    rho_from_wfns(psi_j, occ, kw, U=U_j, **args)
    from common.wfn_transforms import _KERNEL_CACHE
    # Key: (name, psi.shape, grid, volume, f_spin, have_U, current,
    # charge_ns, spin_matrix, sym_perm shape, plan, mesh, psi spec).
    hits = [(k[10], v) for k, v in _KERNEL_CACHE.items()
            if k[0] == "rho_density_scan" and k[1] == psi.shape
            and k[5] and k[6] and k[10][0] == "g_split"]
    assert len(hits) == 1, "exactly one g_split four-current executable"
    (plan, fn), = hits
    k_tile = int(plan[2])
    assert 1 < k_tile < NK
    fns = [fn]
    text = fns[-1].lower(
        psi_j, U_j, jnp.asarray(occ), jnp.asarray(kw),
        jnp.asarray(bidx), jnp.zeros((1, 1), jnp.int32)).compile().as_text()
    colls = re.findall(
        r"= \(?c128\[([\d,]+)\][^\n]*? (all-gather|all-to-all|"
        r"reduce-scatter|collective-permute)(?:-start)?\(", text)
    assert colls, "no sphere collective found: the census is blind"
    for dims, op in colls:
        # XLA squeezes a unit k axis, so the predicate is "no collective
        # carries the whole k extent", not "the leading axis is the tile".
        assert NK not in [int(d) for d in dims.split(",")], (op, dims)
    field_reductions = re.findall(r"= f64\[[\d,]*\][^\n]*? all-reduce", text)
    assert field_reductions, "the one field psum is missing"
    body = re.search(r"body=%?([\w.\-]+)", text)
    if body is not None:                      # loop not unrolled by XLA
        comps = dict(re.findall(
            r"\n(?:ENTRY )?%?([\w.\-]+) [^\n]*\{\n(.*?)\n\}", text,
            flags=re.S))
        assert "all-reduce" not in comps[body.group(1)], (
            "the field is reduced inside the scan")
