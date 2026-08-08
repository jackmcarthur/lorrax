"""The two P>1 breakages that reach the solvers, held shut module-wide.

Both families here are INVISIBLE to every other gate in this tree, because
every in-tree gate builds a 1x1 mesh where a global array is trivially
addressable and a closed-over array is trivially legal.  That is exactly how
both survived to be found by a profiler and a census instead of by the suite:

  FAMILY 1 -- a bare ``jax.device_get`` on a possibly-sharded array.
  ``device_get`` raises ("Fetching value for `jax.Array` that spans
  non-addressable (non process local) devices is not possible") whenever the
  array's shards live on other processes.  It does NOT raise on a fully
  REPLICATED array -- jax tests ``is_fully_replicated`` first and reads the
  local shard -- which is why the Lanczos eigensolve path had always worked and
  ``--solver davidson --write-eigs`` had never been tried at P>1.
  ``common.collectives.gather_to_host`` asks the question instead of assuming,
  and its replicated / single-process arms ARE the plain ``device_get`` that
  was there, so routing a genuinely replicated array through it costs nothing.

  FAMILY 2 -- a ``jax.jit`` body that CLOSES OVER a concrete ``jax.Array``.
  Closed-over arrays are compile-time constants of the executable, and jax
  refuses to trace them at P>1: "Closing over jax.Array that spans
  non-addressable (non process local) devices is not allowed.  Please pass such
  arrays as arguments to the function."  ``bse_feast._get_feast_runner`` did
  this with the ten ``matvec_operands`` arrays, which is why
  ``bse_feast --feast-ritz`` and ``bse_pseudopoles`` could not run
  multi-process AT ALL -- the convergence census had to take its entire GMRES
  iteration-count measurement at P=1 for this reason.

Family 1 can only be a SOURCE check here, for the reason above.  Family 2 does
NOT have that limitation, and this file exploits the difference: whether a
Python function captures a ``jax.Array`` in its ``__closure__`` is decidable at
P=1, on one GPU, with no mesh at all.  So the family that used to be
undetectable except on a 4-rank job is now a unit test.

Runtime evidence for both, at P=4 on the Si record deck, is in
``~/lorrax_bse_perf_2026-08-08/FIX_solver_robustness.md``.
"""
from __future__ import annotations

import ast
import inspect
import pkgutil

import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# ===========================================================================
# FAMILY 1 -- no bare jax.device_get anywhere in src/solvers/, or in the BSE
# writer that consumes what the solvers return.
# ===========================================================================

def _bare_device_get_lines(src: str):
    """Source lines calling ``jax.device_get`` anywhere in ``src``.

    Same shape as the w-omega lane's guard in
    ``tests/test_bse_w_omega_chain_scan.py`` -- an AST walk, NOT a grep.  The
    modules here name ``jax.device_get`` in prose to explain why they do not
    call it, and a text search cannot tell a docstring from a call.  (That is
    not hypothetical: it is the exact failure the w-omega lane hit on its first
    run.)
    """
    tree = ast.parse(src)
    return sorted(node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "jax.device_get")


def _solver_modules():
    """Every importable module under ``src/solvers/``."""
    import solvers

    names = []
    for info in pkgutil.iter_modules(solvers.__path__):
        if info.ispkg:
            continue
        names.append(f"solvers.{info.name}")
    assert names, "no solver modules discovered -- the guard would be vacuous"
    return names


def test_solvers_package_has_no_bare_device_get():
    """MODULE-WIDE over src/solvers/, not a hand-kept list of files.

    A list of files is defeated by adding a file; "the solvers package fetches
    to host through gather_to_host, always" is the rule that cannot be.
    """
    import importlib

    offenders = {}
    for name in _solver_modules():
        try:
            mod = importlib.import_module(name)
        except Exception as exc:                      # pragma: no cover
            pytest.skip(f"{name} not importable in this environment: {exc}")
        try:
            src = inspect.getsource(mod)
        except OSError:                               # pragma: no cover
            continue
        hits = _bare_device_get_lines(src)
        if hits:
            offenders[name] = hits

    assert not offenders, (
        f"bare jax.device_get in the solvers package: {offenders} (source line "
        f"offsets within each module).  A solver cannot know whether its "
        f"caller's mesh left the array replicated or tiled, so this raises at "
        f"P>1 on every rank.  Use common.collectives.gather_to_host -- its "
        f"replicated and 1-GPU arms are the plain device_get you are replacing.")


def test_bse_eigenvector_writer_has_no_bare_device_get():
    """``bse_io`` is the consumer that the solver-convention mismatch bit.

    ``write_eigenvectors_stream`` fetched with a bare ``device_get`` on the
    strength of a comment asserting that ``solve_bse_sharded`` returns
    replicated arrays.  That was true of the Lanczos routes and false of the
    Davidson route, and ``--solver davidson --write-eigs`` died on every rank
    at P>1.  A writer must not depend on a solver's layout to be correct.
    """
    from bse import bse_io

    src = inspect.getsource(bse_io)
    hits = _bare_device_get_lines(src)
    assert not hits, (
        f"bare jax.device_get in bse_io at source line(s) {hits}.  This module "
        f"writes what the solvers return, and not every solver returns "
        f"replicated arrays -- route it through gather_to_host.")
    assert "gather_to_host" in src, (
        "bse_io no longer references gather_to_host at all, so it is clean "
        "only because the host fetches moved somewhere unexamined")


def test_red_twin_the_device_get_guard_actually_fires():
    """The FALSE case, both directions.

    A guard that flags nothing and a guard that flags everything are equally
    useless, so assert the bad pattern IS caught and the fixed pattern is NOT.
    """
    bad = (
        "import jax\n"
        "def f(x):\n"
        "    return jax.device_get(x)\n"
        "def g(y):\n"
        "    return jax.device_get(y)\n"
    )
    assert _bare_device_get_lines(bad) == [3, 5], (
        f"the guard did not flag the bad pattern: {_bare_device_get_lines(bad)}")

    good = (
        "from common.collectives import gather_to_host\n"
        "def f(x):\n"
        "    return gather_to_host(x)\n"
    )
    assert _bare_device_get_lines(good) == [], (
        f"the guard fires on the FIXED pattern too: "
        f"{_bare_device_get_lines(good)}")

    # And the reason it is an AST walk and not a grep: prose that NAMES the
    # call must not count as a call.
    prose = (
        "import jax\n"
        "def f(x):\n"
        '    """We deliberately avoid jax.device_get here; see the note."""\n'
        "    return x\n"
    )
    assert _bare_device_get_lines(prose) == [], (
        "the guard counted a docstring mention as a call -- it has regressed "
        "to a grep")


# ===========================================================================
# FAMILY 2 -- no jitted solver body may CLOSE OVER a jax.Array.
# ===========================================================================

def _closed_over_jax_arrays(fn):
    """Concrete ``jax.Array`` objects captured in ``fn``'s closure.

    Containers are searched one level down, because the operand set that caused
    the real breakage was captured as a single 10-tuple, not as ten cells.
    Callables and Python scalars are NOT flagged: closing over ``matvec`` (a
    function) or over ``max_iter`` (an int) is legal and is how these engines
    are meant to be keyed.  Only ARRAYS are compile-time constants that jax
    refuses to trace at P>1.
    """
    found = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            val = cell.cell_contents
        except ValueError:          # empty cell (recursive reference)
            continue
        if isinstance(val, jax.Array):
            found.append(val)
        elif isinstance(val, (tuple, list)):
            found.extend(v for v in val if isinstance(v, jax.Array))
        elif isinstance(val, dict):
            found.extend(v for v in val.values() if isinstance(v, jax.Array))
    return found


def _unwrap_jit(jitted):
    """The Python function underneath a ``jax.jit``."""
    for attr in ("__wrapped__", "_fun", "func"):
        inner = getattr(jitted, attr, None)
        if callable(inner):
            return inner
    raise AssertionError(
        f"cannot unwrap {jitted!r} to its Python function; this jax version "
        f"changed jit's introspection surface and the guard needs updating")


def test_red_twin_the_closure_detector_actually_fires():
    """The detector must catch the real shape of the bug, and only that shape.

    Written FIRST because everything below is worthless if this is vacuous.
    """
    arr = jnp.ones((3,))
    operands = (jnp.ones((2,)), jnp.zeros((2,)))

    def bad_single():
        return arr + 1

    def bad_tuple():
        return operands[0] + 1

    def fine_callable():
        return _unwrap_jit

    scalar = 7

    def fine_scalar():
        return scalar + 1

    assert len(_closed_over_jax_arrays(bad_single)) == 1, (
        "the detector missed a directly captured jax.Array")
    assert len(_closed_over_jax_arrays(bad_tuple)) == 2, (
        "the detector missed arrays captured inside a tuple -- which is "
        "exactly how the ten matvec_operands were captured")
    assert _closed_over_jax_arrays(fine_callable) == [], (
        "the detector flags a captured CALLABLE; closing over matvec is legal")
    assert _closed_over_jax_arrays(fine_scalar) == [], (
        "the detector flags a captured python scalar; that is legal too")


def test_feast_runner_core_closes_over_no_arrays():
    """RED BEFORE THE FIX, at P=1, on one GPU.

    ``_get_feast_runner``'s jitted body used to pull the ten operand arrays out
    of ``data`` inside the closure.  This cell fails on that tree WITHOUT a
    mesh, a second process or a deck -- which is the whole point: the bug's
    only previous symptom was a 4-rank job refusing to start.
    """
    from bse import bse_feast as BF

    operand_keys = ("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y",
                    "eps_c", "eps_v", "W_R", "V_q0", "M_X", "M_Y")
    data = {k: jnp.full((1,), 1.0, dtype=jnp.complex128) for k in operand_keys}

    def matvec(x, *operands):
        return operands[0] * x

    saved = dict(BF._FEAST_RUNNER_CACHE)
    BF._FEAST_RUNNER_CACHE.clear()
    try:
        runner = BF._get_feast_runner(
            matvec, data, 1, 1, 2, 1e-10, 13.6056980659,
            jnp.complex128, use_conjugate_symmetry=True)
        core = getattr(runner, "core", None)
        assert core is not None, (
            "the runner no longer exposes its jitted core; the guard cannot "
            "see what is baked into the executable")
        leaked = _closed_over_jax_arrays(_unwrap_jit(core))
        assert leaked == [], (
            f"the FEAST runner's jitted body closes over {len(leaked)} "
            f"jax.Array(s) {[a.shape for a in leaked]}.  Those become "
            f"compile-time constants, and jax REFUSES to trace them when their "
            f"shards span processes: 'Closing over jax.Array that spans "
            f"non-addressable (non process local) devices is not allowed.  "
            f"Please pass such arrays as arguments to the function.'  Pass the "
            f"operands as a runtime argument, as _get_gmres_solver does.")
    finally:
        BF._FEAST_RUNNER_CACHE.clear()
        BF._FEAST_RUNNER_CACHE.update(saved)


def test_feast_runner_core_takes_operands_as_an_argument():
    """The positive statement of the same contract.

    Without this cell, "close over nothing" could be satisfied by a body that
    stopped using the operands at all.
    """
    from bse import bse_feast as BF

    assert "data" in inspect.signature(BF._get_feast_runner).parameters, (
        "_get_feast_runner no longer takes the operand source; this cell's "
        "sibling needs rewriting")

    operand_keys = ("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y",
                    "eps_c", "eps_v", "W_R", "V_q0", "M_X", "M_Y")
    data = {k: jnp.full((1,), 1.0, dtype=jnp.complex128) for k in operand_keys}

    def matvec(x, *operands):
        return operands[0] * x

    saved = dict(BF._FEAST_RUNNER_CACHE)
    BF._FEAST_RUNNER_CACHE.clear()
    try:
        runner = BF._get_feast_runner(
            matvec, data, 1, 1, 2, 1e-10, 13.6056980659,
            jnp.complex128, use_conjugate_symmetry=True)
        params = list(inspect.signature(_unwrap_jit(runner.core)).parameters)
        assert "operands" in params, (
            f"the FEAST runner's jitted core takes {params}; the ten operand "
            f"arrays must arrive as a runtime ARGUMENT, which is what makes "
            f"the program legal at P>1 and reusable across operand sets")
        # The caller-facing wrapper keeps the 4-argument shape both drivers and
        # tests/test_bse_feast_runner_cache.py already use.
        assert len(inspect.signature(runner).parameters) == 4, (
            "the wrapper's call shape changed; both FEAST drivers and the "
            "cache gate pass exactly (X_batch, z_nodes, w_weights, diag_h)")
    finally:
        BF._FEAST_RUNNER_CACHE.clear()
        BF._FEAST_RUNNER_CACHE.update(saved)
