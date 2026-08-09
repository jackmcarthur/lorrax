"""THE import-time gate: what a LORRAX driver is allowed to DO while importing.

Roughly four-fifths of a warm LORRAX driver's wall is process bring-up rather
than physics — 80.1 % on the Si record BSE deck at P=4, 78 % on htransform —
and the 2026-08-09 import audit measured what that is made of.  Most of it is
irreducible: Python starting inside the container, ``import jax``, the first
``jax.devices()``, the ``jax.distributed`` handshake, the communicator
warm-up.  Exactly one item was LORRAX's own doing and exactly one was an
accounting error, and this file is the pair of them in a form that fails.

**The item.**  ``common.gamma_matrices`` — on the import path of every driver
that touches ``isdf.core``, which includes ``bandstructure.htransform`` — ran
0.437 s of EAGER JAX WORK in its module body: eight ``jnp.array`` literals
(0.2295 s), four ``jnp.nonzero`` calls building a product nothing read
(0.2085 s), and two ``jnp.stack`` calls, one of which COMPILED AN XLA PROGRAM
at import time.  Built through numpy and handed to jax once, the identical
constants cost 0.006 s.  The same class of defect — a module-scope device-put
that had been in the tree since an extraction — was landed against on
2026-08-08 by the isolation-edge lane; this is its sibling, found by
measurement rather than by reading.

**The accounting error.**  The one-time ``import jax`` was charged to a
startup row named ``env_and_distributed``, whose own report sentence says
"jax.distributed and backend init".  At P=4 that put 2.165 s of Python import
inside a 6.1 s row about a network handshake, which is the difference between
an owner looking at the right thing and the wrong thing.  ``jax_import`` is
now carved OUT of that row — carved, not added, so the rows still sum.

Every cell below is falsifiable: the scanner cells are run against a
deliberately bad synthetic module in the same test, so a scanner that stopped
detecting anything would fail rather than pass quietly.
"""

from __future__ import annotations

import ast
import os
import textwrap

import numpy as np
import pytest

import runtime


_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
_GAMMA = os.path.join(_SRC, "common", "gamma_matrices.py")


# ---------------------------------------------------------------------------
# The scanner, and the proof that it can see
# ---------------------------------------------------------------------------

#: Calls that are forbidden at MODULE SCOPE in the files this gate covers.
#: ``jnp.array``/``jnp.asarray`` differ on purpose: ``asarray`` of an array
#: that already has the target dtype is a plain transfer, while ``array`` of a
#: Python list goes through jax's dtype-conversion dispatch — that is the
#: 0.2295 s vs 0.0059 s the audit measured, not a style preference.
_FORBIDDEN_AT_MODULE_SCOPE = ("jnp.array", "jnp.stack", "jnp.nonzero",
                              "jnp.concatenate", "jnp.zeros", "jnp.ones",
                              "jnp.eye", "jnp.linspace", "jnp.arange")


def _dotted(node: ast.AST) -> str | None:
    """``jnp.array`` from the AST of a call's ``func``, or None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def eager_jax_calls_at_module_scope(source: str) -> list[tuple[int, str]]:
    """(lineno, name) for every forbidden eager-jax call in a module BODY.

    Function and class bodies are skipped deliberately: the cost this gate is
    about is paid at import, and a ``jnp.array`` inside a function is paid
    when the function runs, which is the physics' own budget.
    """
    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                name = _dotted(sub.func)
                if name in _FORBIDDEN_AT_MODULE_SCOPE:
                    hits.append((sub.lineno, name))
    return hits


def test_the_scanner_sees_a_module_that_is_actually_bad():
    """RED TWIN for the scanner itself.

    A gate whose detector silently stopped working reads exactly like a gate
    that is passing.  This is the synthetic bad module: if the cells below go
    green because ``eager_jax_calls_at_module_scope`` returns ``[]`` for
    everything, this cell goes red first.
    """
    bad = textwrap.dedent("""
        import jax.numpy as jnp
        import numpy as np
        sigma = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        stacked = jnp.stack([sigma, sigma])
        def fine():
            return jnp.array([1, 2, 3])      # inside a function: allowed
    """)
    hits = eager_jax_calls_at_module_scope(bad)
    names = sorted(n for _, n in hits)
    assert names == ["jnp.array", "jnp.stack"], hits
    assert "fine" not in bad[:0]  # the function-body call was NOT reported


def test_the_scanner_passes_a_module_that_is_actually_good():
    """The other half of the twin: the sanctioned spelling must not trip it."""
    good = textwrap.dedent("""
        import jax.numpy as jnp
        import numpy as np
        sigma = jnp.asarray(np.array([[0, 1], [1, 0]], dtype=np.complex128))
        stacked = jnp.asarray(np.stack([np.eye(2), np.eye(2)]))
    """)
    assert eager_jax_calls_at_module_scope(good) == []


# ---------------------------------------------------------------------------
# The item: common.gamma_matrices
# ---------------------------------------------------------------------------

def test_gamma_matrices_does_no_eager_jax_work_at_import():
    """The 0.437 s -> 0.006 s conversion, pinned.

    Reverting the module to ``jnp.array([[...]], dtype=...)`` /
    ``jnp.stack([...])`` puts the third of a second back on every htransform
    and GW bring-up, and turns this cell red on the exact lines.
    """
    hits = eager_jax_calls_at_module_scope(open(_GAMMA, encoding="utf-8").read())
    assert hits == [], (
        f"common/gamma_matrices.py builds device arrays at module scope again: "
        f"{hits}.  Build with numpy and hand to jax once — the module docstring "
        f"carries the measurement.")


def test_the_dead_sparse_product_stays_dead():
    """``gammas_sparse`` / ``_to_sparse`` were defined and never read.

    They cost 0.2085 s of every import AND they were the reason
    ``gw.isdf_fitting`` carried a "force-eager-import" workaround: their
    ``jnp.nonzero`` has a data-dependent output shape, so a first evaluation
    inside a jit trace raises ConcretizationTypeError.  Deleting the dead
    product deleted the hazard; this cell is what keeps someone from
    reintroducing both by "restoring" a helper that looks useful.
    """
    from common import gamma_matrices as gm
    assert not hasattr(gm, "gammas_sparse"), (
        "gammas_sparse is back.  It has no readers anywhere in the tree; it "
        "costs 0.21 s per import and it reintroduces the trace hazard that "
        "gw/isdf_fitting.py used to work around.")
    assert not hasattr(gm, "_to_sparse"), "the _to_sparse helper is back"


def test_gamma_constants_are_bit_identical_to_the_literals_they_replaced():
    """BEHAVIOUR UNCHANGED — the half of a side-effect move that matters.

    The conversion is only allowed to change WHEN and HOW the arrays are
    built, never WHAT they are.  The references here are written out again by
    hand rather than imported, so a typo in the module cannot agree with
    itself.
    """
    import jax.numpy as jnp
    from common import gamma_matrices as gm

    ref = {
        "sigma_x": [[0, 1], [1, 0]],
        "sigma_y": [[0, -1j], [1j, 0]],
        "sigma_z": [[1, 0], [0, -1]],
        "gamma0": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "gamma1": [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        "gamma2": [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]],
        "gamma3": [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]],
        "gamma5": [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    }
    for name, rows in ref.items():
        got = getattr(gm, name)
        assert got.dtype == jnp.complex128, (name, got.dtype)
        np.testing.assert_array_equal(np.asarray(got),
                                      np.array(rows, dtype=np.complex128))

    # The derived pair, and its defining property: gamma[a, perm[a]] == phase[a]
    # and every other entry of the row is zero.
    perm = np.asarray(gm.gammas_perm)
    phase = np.asarray(gm.gammas_phase)
    assert perm.shape == (4, 4) and phase.shape == (4, 4)
    assert perm.dtype == np.int32, perm.dtype
    for mu, name in enumerate(("gamma0", "gamma1", "gamma2", "gamma3")):
        m = np.array(ref[name], dtype=np.complex128)
        rebuilt = np.zeros_like(m)
        for a in range(4):
            rebuilt[a, perm[mu, a]] = phase[mu, a]
        np.testing.assert_array_equal(rebuilt, m)


def test_the_module_body_is_safe_to_evaluate_inside_a_jit_trace():
    """The property the deleted ``isdf_fitting`` workaround existed to buy.

    ``vertex_mu_L != 0`` used to force an eager import so the module body
    could not first run inside the per-chunk kernel's trace.  With the
    ``jnp.nonzero`` product gone, the body is trace-clean and the workaround
    is unnecessary — which is a claim, so here it is executed.
    """
    import jax
    import jax.numpy as jnp

    src = open(_GAMMA, encoding="utf-8").read()

    def _run(x):
        ns: dict = {"__name__": "common.gamma_matrices_trace_probe"}
        exec(compile(src, _GAMMA, "exec"), ns)          # noqa: S102 — that IS the test
        return x * ns["gammas_phase"][0, 0]

    out = jax.jit(_run)(jnp.asarray(2.0 + 0j, dtype=jnp.complex128))
    np.testing.assert_allclose(np.asarray(out), 2.0 + 0j)


def test_a_nonzero_in_the_body_would_still_be_caught_by_a_trace(monkeypatch):
    """RED TWIN for the cell above: the trace really does reject the old body.

    Without this, ``test_the_module_body_is_safe_to_evaluate_inside_a_jit_trace``
    could be passing because tracing accepts everything.
    """
    import jax
    import jax.numpy as jnp

    old_body = textwrap.dedent("""
        import jax.numpy as jnp
        g = jnp.asarray([[1, 0], [0, 1]])
        r, c = jnp.nonzero(g)          # data-dependent shape
        out = (r, c, g[r, c])
    """)

    def _run(x):
        ns: dict = {}
        exec(compile(old_body, "<old_gamma_body>", "exec"), ns)  # noqa: S102
        return x

    with pytest.raises(Exception) as exc:
        jax.jit(_run)(jnp.asarray(1.0))
    assert "Concretization" in type(exc.value).__name__ or \
           "nonzero" in str(exc.value) or "concrete" in str(exc.value).lower(), \
        f"expected a concreteness refusal, got {type(exc.value).__name__}: {exc.value}"


# ---------------------------------------------------------------------------
# The accounting error: jax_import is carved out of env_and_distributed
# ---------------------------------------------------------------------------

def _elapsed(**over):
    base = {"jax_import": 2.165, "env_and_distributed": 3.935,
            "mesh_and_warmup": 2.220, "compile_cache": 0.402,
            "measurement": 0.030, "total": 8.752}
    base.update(over)
    return base


def test_the_startup_phases_still_sum_to_their_total():
    """CARVED, NOT ADDED.

    Three driver epilogues (``htransform``, ``gw_jax``, ``bse_jax``) re-record
    every key of ``elapsed`` except ``total`` and then charge the REMAINDER of
    the process's pre-main wall to imports.  A phase added beside the others
    instead of carved out of one makes that remainder negative and the whole
    table stops summing to the wall — the exact property the tables were
    given a ``(untimed)`` closer to expose.
    """
    el = _elapsed()
    parts = sum(v for k, v in el.items() if k != "total")
    assert parts == pytest.approx(el["total"], abs=1e-9), el


def test_a_phase_added_beside_the_others_is_caught():
    """RED TWIN: the sum check has to be able to fail."""
    bad = _elapsed(env_and_distributed=6.100)     # the un-carved value
    parts = sum(v for k, v in bad.items() if k != "total")
    assert parts != pytest.approx(bad["total"], abs=1e-9)


def test_the_report_states_the_import_separately_from_the_handshake():
    """An owner reading the block must not have to know that a row named
    ``env_and_distributed`` silently contains the Python import storm."""
    from tests.test_runtime_startup_report import _facts  # noqa: PLC0415
    f = _facts()
    f["elapsed"] = _elapsed()
    text = "\n".join(runtime.format_startup_report(f))
    assert "2.2 s to import jax itself" in text, text
    assert "3.9 s for the environment, jax.distributed and backend init" in text


def test_the_report_survives_an_elapsed_with_no_jax_import():
    """Probes and older logs hand-build ``elapsed``; the key is optional.

    This is the compatibility half — the sentence must degrade to the pre-split
    wording rather than raise KeyError.
    """
    from tests.test_runtime_startup_report import _facts  # noqa: PLC0415
    f = _facts()
    el = _elapsed()
    del el["jax_import"]
    el["env_and_distributed"] = 6.100
    f["elapsed"] = el
    text = "\n".join(runtime.format_startup_report(f))
    assert "Bringing this stack up took" in text
    assert "to import jax itself" not in text


def test_the_import_counter_charges_only_the_first_import():
    """``_import_jax`` must be idempotent in COST as well as in effect.

    Every bring-up piece that needs jax calls it; if each call re-charged, the
    carved-out phase would exceed the row it was carved from and
    ``initialize_communicator_stack``'s ``min()`` clamp would be doing the
    work this helper is supposed to do honestly.
    """
    before = runtime._JAX_IMPORT_SECONDS[0]
    runtime._import_jax()
    runtime._import_jax()
    assert runtime._JAX_IMPORT_SECONDS[0] == before, (
        "a second _import_jax() charged again; jax was already in sys.modules")


# ---------------------------------------------------------------------------
# The BSE driver's bring-up row
# ---------------------------------------------------------------------------

def test_bse_jax_decomposes_its_bring_up_row():
    """``bse.imports_and_runtime`` was 80.1 % of a warm P=4 wall in ONE row.

    Static, because the alternative is running the driver: the epilogue must
    record the per-phase rows and an ``imports`` remainder, the same shape
    ``htransform`` has had since 2026-08-08.
    """
    src = open(os.path.join(_SRC, "bse", "bse_jax.py"), encoding="utf-8").read()
    assert 'timing.record("bse.imports_and_runtime"' not in src, (
        "the opaque single row is back")
    assert 'f"bse.runtime_stack.{_phase}"' in src, "the phase rows are missing"
    assert 'timing.record("bse.imports"' in src, "the imports remainder is missing"


def test_the_bring_up_rows_sum_to_the_pre_main_wall():
    """The arithmetic the epilogue does, done here against known numbers.

    Rows carved out of ``_pre_main`` must still add up to it, or the driver's
    ``TOTAL (wall)`` stops meaning the process wall.
    """
    pre_main = 11.190                       # the measured Si P=4 warm value
    el = _elapsed()
    rows = [v for k, v in el.items() if k != "total"]
    rows.append(max(pre_main - el["total"], 0.0))          # the imports remainder
    assert sum(rows) == pytest.approx(pre_main, abs=1e-9)
