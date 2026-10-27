"""gw.stage_inventory against measured GN-PPM stage peaks (CPU, no device; seconds).

``tests/data/stage_peaks_measured.json`` holds the allocator high-water of each
``LORRAX_MEMORY_TRACE`` section, max over ranks, of three one-node P = 4 runs
(MoS₂/hBN 148 atoms at N_μ 8160 and 16320, Al₂O₃ 4×4×4 at N_μ 6000).  Each
inventory stage is the trace section named in :data:`SECTION`; every stored
peak must lie within :data:`BAND` of the inventory's price at the run's route, or within
:data:`SLACK` of it (stages under a GB).
"""
import json
import pathlib

import pytest

#: Inventory stage -> the LORRAX_MEMORY_TRACE section whose measured peak it prices.
SECTION = {
    "ISDF setup + I/O": "gw_jax.load_centroid_wfns",
    "Coulomb all-q validation": "gw_jax.isdf (report row ISDF setup + I/O)",
    "zeta C = (psi psi)(psi psi)^H": "zeta_fit.CCT",
    "zeta factor B": "zeta_fit.cholesky",
    "V(q)": "gw_jax.V_q_compute",
    "chi0": "chi.exec",
    "W slice to IBZ": "W.slice_to_ibz",
    "W": "W.exec",
    "Sigma exchange": "sigma.exchange",
    "GN-PPM fit": "gn_ppm.fit",
    "Sigma tau sweep": "sigma.tau_sweep",
    "Sigma other": "sigma.exec",
    "Sigma Hartree": "sigma.hartree",
}
#: measured / priced, per stage; or |measured - priced| at most SLACK GB.
BAND, SLACK = (0.75, 1.3), 0.5
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
        if not (BAND[0] <= ratio <= BAND[1] or abs(measured - priced) <= SLACK):
            off.append(f"{stage} ({SECTION[stage]}): measured {measured:.2f} GB, "
                       f"priced {priced:.2f} GB, ratio {ratio:.2f}")
    assert not off, f"{run['name']}: " + "; ".join(off)
