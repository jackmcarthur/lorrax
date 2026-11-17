"""gw.stage_inventory against measured GN-PPM and tangential stage peaks (CPU, no device; seconds).

``tests/data/stage_peaks_measured.json`` holds the allocator high-water of each
``LORRAX_MEMORY_TRACE`` section, max over ranks, of five one-node P = 4 runs
(MoS₂/hBN 148 atoms at N_μ 8160 and 16320; Al₂O₃ 4×4×4 at N_μ 6000, GN-PPM, and tangential
on both Σ factor routes), and the report rows of the nine-node MoS₂/hBN N_μ 60012 run at
P = 36 (linalg distributed).  Each inventory stage is the trace section named in
:data:`SECTION`; every stored peak must lie within :data:`BAND` of the inventory's price
at the run's route, or within :data:`SLACK` of it (stages under a GB); a stage in
:data:`TIGHT` within its own band.
"""
import json
import pathlib

import pytest

#: Inventory stage -> the LORRAX_MEMORY_TRACE section whose measured peak it prices.
SECTION = {
    "ISDF load": "gw_jax.load_centroid_wfns",
    "Coulomb all-q validation": "gw_init._canonical_coulomb_stats (in gw_jax.isdf)",
    "ISDF setup + I/O": "report row ISDF setup + I/O (gw_jax.isdf outside zeta, V(q), restart)",
    "zeta C = (psi psi)(psi psi)^H": "zeta_fit.CCT",
    "zeta factor B": "zeta_fit.cholesky",
    "V(q)": "gw_jax.V_q_compute",
    "chi0": "chi.exec",
    "W slice to IBZ": "W.slice_to_ibz",
    "W": "W.exec",
    "Sigma exchange": "sigma.exchange",
    "GN-PPM fit": "gn_ppm.fit",
    "Sigma tau sweep": "sigma.tau_sweep",
    "Sigma tau sweep (tangential)": "sigma.tau_sweep",
    "Sigma other": "sigma.exec",
    "Sigma Hartree": "sigma.hartree",
}
#: measured / priced, per stage; or |measured - priced| at most SLACK GB.
BAND, SLACK = (0.75, 1.3), 0.5
#: Stages pinned closer: tangential Σ τ on the axis (replicated pole columns) and face routes.
TIGHT = {"Sigma tau sweep (tangential)": (0.95, 1.05)}
DATA = pathlib.Path(__file__).parent / "data" / "stage_peaks_measured.json"


def _runs():
    return json.loads(DATA.read_text())["runs"]


@pytest.mark.parametrize("run", _runs(), ids=lambda r: r["name"])
def test_stage_inventory_prices_measured_peaks(run):
    from gw.stage_inventory import Deck, stage_prices
    deck = dict(run["deck"], fft=tuple(run["deck"]["fft"]), kgrid=tuple(run["deck"]["kgrid"]))
    prices = stage_prices(Deck(**deck), run["P"])
    off = []
    for stage, measured in run["measured"].items():
        routes = dict(prices[stage])
        priced = routes.get(run["linalg"], next(iter(routes.values()))).bytes() / 1e9
        ratio = measured / priced
        lo, hi = TIGHT.get(stage, BAND)
        if not (lo <= ratio <= hi or (stage not in TIGHT and abs(measured - priced) <= SLACK)):
            off.append(f"{stage} ({SECTION[stage]}): measured {measured:.2f} GB, "
                       f"priced {priced:.2f} GB, ratio {ratio:.2f}")
    assert not off, f"{run['name']}: " + "; ".join(off)


def _deck(name):
    from gw.stage_inventory import Deck
    run = next(r for r in _runs() if r["name"].startswith(name))
    return Deck(**dict(run["deck"], fft=tuple(run["deck"]["fft"]),
                       kgrid=tuple(run["deck"]["kgrid"]))), run["P"]


def test_validation_width_is_the_runs_own_choice():
    """The all-q validation admits the widest of min(64, nk), /2, ... that fits (the widths the
    runs took: Al2O3 4^3 64 q, MoS2/hBN 60k all 4); with no room, its floor."""
    from dataclasses import replace
    from gw.stage_inventory import stage_prices, validation_width
    al, P = _deck("Al2O3 10 atoms 4x4x4, 154 bands, N_mu 6000")
    assert validation_width(al, P) == 64
    slab, P = _deck("MoS2/hBN 148 atoms, 9010 bands")
    assert validation_width(slab, P) == 4
    assert validation_width(slab, 4) == 1
    assert validation_width(replace(slab, validation_width=2), P) == 2
    # The report row is the larger of the load and the validation.
    prices = stage_prices(slab, P)
    row = dict(prices["ISDF setup + I/O"])["only"].bytes()
    assert row == max(dict(prices[s])["only"].bytes()
                      for s in ("ISDF load", "Coulomb all-q validation"))


def test_chi0_axis_banks_grow_with_the_mesh_side():
    """The axis layout's band-complete banks are (px + py)/2 F per rank: 6 F on 6x6, where the
    old flat 3 F priced 60k's chi0 at 21.5 GB against 27.2 measured; face where they do not fit."""
    from gw.stage_inventory import BUDGET, _units, chi0_route
    slab, P = _deck("MoS2/hBN 148 atoms, 9010 bands")
    route, u = chi0_route(slab, P), _units(slab, P)
    assert route["layout"] == "axis"
    banks = route["xla"] - slab.nk * u["M"]
    assert 6 * u["F"] < banks <= 6 * u["F"] + (1 << 30)
    # 3x3: residents + 3 F of banks + the accumulator exceed the budget, so the run takes face.
    u9 = _units(slab, 9)
    assert u9["F"] * 3 + 2 * u9["QM"] + u9["F"] + slab.nk * u9["M"] > BUDGET["local"]
    face = chi0_route(slab, 9)
    assert face["layout"] == "face"
    assert u9["F"] + slab.nk * u9["M"] < face["xla"] <= u9["F"] + slab.nk * u9["M"] + (1 << 30)
