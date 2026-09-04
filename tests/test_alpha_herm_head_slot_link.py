"""The α-Hermiticity invariant, held against the Coulomb head-slot rule.

What this file is for
---------------------
``solvers/lanczos.py`` gates ``α_j = ⟨q_j, H q_j⟩``, which is REAL for any
Hermitian ``H`` and any ``q_j`` whatsoever, at
``ALPHA_HERM_RTOL = 1e-9``.  On every ``main``-based tree that gate has
failed on every BSE deck ever run — twenty-plus configurations, 3.5e-07 to
4.7e-06, zero passes — and the cause is not in the solver.  It is the
mini-BZ Coulomb head, which used to be injected at the slot labelled
Miller (0,0,0), a label that is not equivariant under ``q → −q``.  ``V_q``
carried the break, ``W`` inherited it, and the matvec transported it
faithfully into ``α``.

MEASURED, si_bse_debug, P=4, 400 Lanczos iterations, same 1e-9 tolerance
on both arms (``/pscratch/sd/j/jackm/vcoul_head_0808/_reports/``, and
reproduced independently at ``/pscratch/sd/j/jackm/vcoulhead_land_0808/``):

    label rule (main)                 rel = 1.155e-06   SANITY FAILURE
    argmin + tied-mean (this tree)    rel = 3.165e-14   OK

``services/vcoul/tests/test_vcoul_head_slot_reciprocity.py`` already holds
the rule at ``v`` level, where the statement is exact and arm-independent.
This file holds the OTHER end of the same wire: it shows that the quantity
``solvers.lanczos`` actually reports is the one that ``v``-level
reciprocity controls, so that nobody can relax ``ALPHA_HERM_RTOL`` to
silence a Coulomb defect, and nobody can reinstate the label rule without
a Lanczos-side cell going red.  Nothing else in the tree connects the two
packages, and the disconnect is precisely why a correct, deterministic
gate was read as noise for ten days.

Evidence: ``~/lorrax_bse_perf_2026-08-08/HERMITICITY_INVESTIGATION.md``.

The model
---------
Not the production matvec — a minimal faithful stand-in for it, small
enough to run in a second on a login node.  The exciton Hamiltonian is
Hermitian iff ``W_q = conj(W_{−q})``, and ``W`` is bilinear in ζ over
``v(q+G)``, so the *only* way the head can reach ``α`` is through
``v(+q, i) ≠ v(−q, pair(i))``.  Build exactly that dependence and nothing
else::

    H[a, b] = conj(ζ[a]) · v[q(a,b), s(a,b)] · ζ[b]

with ``q(a, b) = wrap(k_a − k_b)`` on the fixture's own 4×4×4 grid,
``s(a, b)`` the argmin slot of that q, and ζ a fixed-seed complex vector.
``H[a, b] = conj(H[b, a])`` then holds **iff** ``v`` is reciprocal, so
``Im ⟨x, H x⟩`` is a pure readout of the head-slot rule.  The residual is
handed to the real gate — ``lanczos._report_alpha_herm`` — so the verdict
comes from the shipped code, not from a re-implementation of it.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Bootstrap first, door second -- the same order rule the shim-identity
# cells use: nothing guarantees another test module has put
# services/*/src on sys.path yet.
import vcoul as V                                    # noqa: E402
from solvers import lanczos                          # noqa: E402


def _load_service_gate():
    """The v-level gate's geometry helpers, by path.

    Imported rather than copied: the per-q G-lists, the BGW wrap and the
    ``K ↦ −K`` pairing must be the SAME construction the v-level gate
    accepts on, or this cell would be measuring a different geometry and
    could pass while that one fails.  A unique module name keeps pytest's
    collector from seeing two modules with the same basename.
    """
    path = os.path.join(_ROOT, "services", "vcoul", "tests",
                        "test_vcoul_head_slot_reciprocity.py")
    spec = importlib.util.spec_from_file_location(
        "_vcoul_head_slot_gate_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_service_gate()

_NMC = 512          # the invariant is a SYMMETRY: exact at any sample count
_NK = 24            # size of the toy Krylov/exciton basis


@pytest.fixture(scope="module")
def si():
    q_int, q_frac, gvec, ngk = R._per_q_glists(
        R.SI_BVEC, R.SI_KGRID, (12, 12, 12), R._CUTOFF_RY)
    neg_q, pair, worst, K = R._pairing(
        q_int, q_frac, gvec, ngk, R.SI_BVEC, R.SI_KGRID)
    return dict(q_int=q_int, q_frac=q_frac, gvec=gvec, ngk=ngk,
                neg_q=neg_q, pair=pair, worst=worst, K=K)


def _argmin_slot(si, qi):
    """The slot the fixed rule injects into (first of the tied set)."""
    d = np.sum(si["K"][qi, :int(si["ngk"][qi])] ** 2, axis=1)
    return int(np.argmin(d))


def _alpha_stats(v, si):
    """``(dev, scale)`` — the two scalars ``lanczos`` ships out of the trace.

    ``dev = max_j |Im α_j|``, ``scale = max_j |α_j|``, over a fixed set of
    probe vectors standing in for the Krylov basis.  The probes are seeded,
    so this is bit-stable — which matters, because determinism is what
    separates an operator defect from the gloo reduce-scatter corruption
    the gate's message used to blame.
    """
    kg = np.asarray(R.SI_KGRID, dtype=np.int64)
    where = {tuple(int(c) for c in si["q_int"][i]): i
             for i in range(len(si["q_int"]))}
    rng = np.random.default_rng(20260808)
    ks = si["q_int"][:_NK]                      # the toy "k-points"
    zeta = (rng.standard_normal(_NK) + 1j * rng.standard_normal(_NK))

    H = np.zeros((_NK, _NK), dtype=np.complex128)
    for a in range(_NK):
        for b in range(_NK):
            qi = where[tuple((ks[a] - ks[b]) % kg)]
            H[a, b] = np.conj(zeta[a]) * float(v[qi, _argmin_slot(si, qi)]) \
                * zeta[b]

    dev = 0.0
    scale = 0.0
    for j in range(_NK):
        x = (rng.standard_normal(_NK) + 1j * rng.standard_normal(_NK))
        x /= np.linalg.norm(x)
        alpha = complex(np.vdot(x, H @ x))
        dev = max(dev, abs(alpha.imag))
        scale = max(scale, abs(alpha))
    return dev, scale


def _v_fixed(si):
    head_fn = V.build_v_head_miniBZ_fn_3d(
        R.SI_KGRID, R.SI_BVEC, R.SI_CELVOL, nmc=_NMC)
    return np.asarray(V.v_qG_table(
        V.get_kernel(3), si["q_frac"], si["gvec"],
        geometry=V.CoulombGeometry(bvec=R.SI_BVEC,
                                   cell_volume=R.SI_CELVOL),
        v_head_fn=head_fn), dtype=np.float64)


def _v_label_rule(si):
    head_fn = V.build_v_head_miniBZ_fn_3d(
        R.SI_KGRID, R.SI_BVEC, R.SI_CELVOL, nmc=_NMC)
    return R._label_rule_v(si, head_fn)


# ---------------------------------------------------------------------------
# The acceptance, and the twin that must fail it.
# ---------------------------------------------------------------------------

def test_alpha_herm_invariant_holds_under_the_argmin_tied_mean_rule(
        si, monkeypatch, capsys):
    """THE ACCEPTANCE — and it is the SHIPPED gate that returns the verdict.

    With the head injected at ``argmin |q+G|²`` over the tied set, ``v`` is
    exactly reciprocal, so ``H`` is Hermitian to the last bit and
    ``Im α`` is exactly zero.  On the real deck the same rule scores
    3.165e-14 against the same 1e-9 — round-off, and 2.5 orders BELOW the
    round-off budget derived for that shape.
    """
    monkeypatch.setenv("LORRAX_SANITY", "1")
    dev, scale = _alpha_stats(_v_fixed(si), si)
    assert scale > 0.0
    assert dev / scale <= lanczos.ALPHA_HERM_RTOL, (
        f"rel = {dev / scale:.3e} > {lanczos.ALPHA_HERM_RTOL:.0e}")
    assert lanczos._report_alpha_herm("head_slot_probe", "vec",
                                      dev, scale, 0) is True
    assert "SANITY FAILURE" not in capsys.readouterr().out


def test_RED_TWIN_the_miller_zero_label_rule_fails_the_alpha_gate(
        si, monkeypatch, capsys):
    """The case where the invariant comes out FALSE.

    The retired Miller-(0,0,0) rule, reinstated locally by the v-level
    gate's own twin helper, must drive ``Im α`` above 1e-9 by orders.  If
    this ever passes, either the head fix has been reverted without anyone
    noticing or this cell has stopped touching the head at all — and the
    acceptance above becomes unfalsifiable.
    """
    monkeypatch.setenv("LORRAX_SANITY", "1")
    dev, scale = _alpha_stats(_v_label_rule(si), si)
    rel = dev / scale
    assert rel > 1e3 * lanczos.ALPHA_HERM_RTOL, (
        f"the label rule scored rel = {rel:.3e}; it is supposed to FAIL "
        f"the 1e-9 gate by orders (1.155e-06 on the real deck)")
    assert lanczos._report_alpha_herm("head_slot_probe", "vec",
                                      dev, scale, 0) is False
    out = capsys.readouterr().out
    assert "is NOT Hermitian" in out
    # and the message must send the reader to the head slot, not to a
    # collective — see tests/test_lanczos_alpha_gate_message.py
    assert "Miller (0,0,0)" in out


def test_anti_attribution_the_bare_kernel_alone_is_alpha_hermitian(si):
    """With no head at all, ``Im α`` is zero too.

    So everything the twin measures is the head-slot rule, and none of it
    can be charged to the bare Coulomb formula, to the probe vectors, or
    to the toy Hamiltonian's own construction.
    """
    v = np.asarray(V.v_qG_table(
        V.get_kernel(3), si["q_frac"], si["gvec"],
        geometry=V.CoulombGeometry(bvec=R.SI_BVEC,
                                   cell_volume=R.SI_CELVOL)),
        dtype=np.float64)
    dev, scale = _alpha_stats(v, si)
    assert dev / scale <= lanczos.ALPHA_HERM_RTOL


def test_the_tolerance_is_not_the_thing_that_moved():
    """No landing gets to relax it.

    1e-9 is derived (``solvers/lanczos.py`` carries the round-off budget:
    2.7e-11 at the largest production shape), not tuned, and the fixed arm
    clears it by four and a half orders on the very deck that was failing.
    """
    assert lanczos.ALPHA_HERM_RTOL == 1e-9
