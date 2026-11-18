"""Rebuild the v2 test sidecars of this directory from their radial inputs.

Run from the repository root (sources on PYTHONPATH as for the tests)::

    python tests/data/onsite_breit/make_v2_fixtures.py

``lorrax.onsite_breit_sidecar/v2`` stores the radial numbers ``N_L``, ``K`` of the
four-spinor tensor (docs/dev/onsite_breit.md sections 6, 7); every block below goes
through ``psp.onsite_breit_sidecar.tensor_block`` (and its ``verify_breit``).

- ``Cr``, ``Cl``: ``make_sidecar`` on ``CrCl3_{Cr,Cl}.upf`` with ``{Cr_3d,Cl_3p}_ae.dat``,
  exactly what ``python -m psp.onsite_breit_sidecar --upf ... --ae ...`` writes.
- ``Fe``: no Fe UPF is shipped.  AE from ``Fe_3d_ae.dat``; PS from the projector chi of
  ``v1/Fe.onsite_breit.json`` (the plain j-average, which equals the aligned one for that
  file up to normalisation); the projector block is copied from the v1 file.
- ``Cr3plus_uhf3dup``, ``Fe2plus_uhf3dup``: operator-algebra fixtures (projector type
  ``none``, ``ps`` null).  ``u = r R``, ``R = sum_i c_i g_i(r)``, ``g_i = r^2 exp(-a_i r^2)``
  normalised, from the UHF 3d-up contraction kept in the v1 file's
  ``provenance.sibling.basis``, on ``r = [0, geomspace(1e-6, 40, 4000)]``.

The v1 files (``v1/``, spin-other-orbit tensor) stay as the input of the independent
checks: the v1 refusal, and the spin-other-orbit tensor against the post-processing code
``onsite_ops`` (``v1/reference_values.json``).
"""
import copy
import json
import os
import sys
from math import gamma

import numpy as np

from psp import onsite_breit as ob
from psp import onsite_breit_sidecar as gen

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-11-18"
SELF = "tests/data/onsite_breit/make_v2_fixtures.py"


def _p(*a):
    return os.path.join(HERE, *a)


def _v1(name):
    with open(_p("v1", f"{name}.onsite_breit.json")) as f:
        return json.load(f)


def species(el, upf, label, table):
    side = gen.make_sidecar(_p(upf), label, ae=gen.read_radial_table(_p(table)),
                            ae_reference=_v1(el)["shells"][0]["ae"]["reference"],
                            ae_source=f"{table} sha256 {gen._sha256(_p(table))}", date=DATE)
    return side


def iron():
    v1 = _v1("Fe")
    sh1 = v1["shells"][0]
    pj = sh1["projector"]
    ae, ae_info = gen.tensor_block(*gen.read_radial_table(_p("Fe_3d_ae.dat")), 2, sh1["ae"]["reference"])
    ps, ps_info = gen.tensor_block(np.asarray(pj["r"]), np.asarray(pj["chi"]), 2,
                                   "projector.chi of v1/Fe.onsite_breit.json (plain j-average of the Fe UPF "
                                   "PP_CHI pair; no Fe UPF in the test data), normalised")
    shell = dict(label=sh1["label"], l=2, projector=copy.deepcopy(pj), ae=ae, ps=ps)
    prov = dict(generator=f"{SELF} (psp.onsite_breit_sidecar.tensor_block)", date=DATE,
                source_upf=v1["provenance"]["source_upf"],
                ae_source=f"Fe_3d_ae.dat sha256 {gen._sha256(_p('Fe_3d_ae.dat'))}",
                quadrature="cubic spline in the grid index (psp.onsite_breit_sidecar.radial_integral)",
                checks={"ae": ae_info, "ps": ps_info})
    return dict(format=ob.SIDECAR_FORMAT, element="Fe", conventions=gen.CONVENTIONS, shells=[shell],
                provenance=prov)


def uhf_fixture(name):
    v1 = _v1(name)
    bas = v1["provenance"]["sibling"]["basis"]
    l = int(bas[0])
    a = np.array([p[0] for p in bas[1:]])
    c = np.array([p[1] for p in bas[1:]])
    c = c / np.sqrt(np.array([gamma(l + 1.5) / (2.0 * (2.0 * x) ** (l + 1.5)) for x in a]))
    r = np.concatenate([[0.0], np.geomspace(1e-6, 40.0, 4000)])
    u = r * r ** l * (c[None, :] * np.exp(-a[None, :] * r[:, None] ** 2)).sum(1)
    sh1 = v1["shells"][0]
    ae, info = gen.tensor_block(r, u, l, sh1["ae"]["reference"])
    shell = dict(label=sh1["label"], l=l, projector={"type": "none"}, ae=ae, ps=None)
    prov = dict(generator=f"{SELF} (psp.onsite_breit_sidecar.tensor_block)", date=DATE, source_upf=None,
                ae_source=v1["provenance"]["ae_source"] + " (v1 provenance.sibling.basis, normalised primitives)",
                sibling=v1["provenance"]["sibling"], checks={"ae": info},
                purpose="operator-algebra unit tests only (atomic limit); no projector")
    return dict(format=ob.SIDECAR_FORMAT, element=v1["element"], conventions=gen.CONVENTIONS, shells=[shell],
                provenance=prov)


def main():
    out = {"Cr": species("Cr", "CrCl3_Cr.upf", "3D", "Cr_3d_ae.dat"),
           "Cl": species("Cl", "CrCl3_Cl.upf", "3P", "Cl_3p_ae.dat"),
           "Fe": iron(),
           "Cr3plus_uhf3dup": uhf_fixture("Cr3plus_uhf3dup"),
           "Fe2plus_uhf3dup": uhf_fixture("Fe2plus_uhf3dup")}
    for name, side in out.items():
        path = _p(f"{name}.onsite_breit.json")
        sha = gen.write_sidecar(side, path)
        for kind in ("ae", "ps"):
            b = side["shells"][0][kind]
            if b is not None:
                print(f"{name} {kind}: N = {b['N']}, K = {b['K']!r}")
        print(f"wrote {path} (sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
