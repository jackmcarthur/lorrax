"""Gate: ``local_ifftn3``/``local_fftn3`` are shard_map-INTERIOR kernels.

``common.fft_helpers.local_{i,}fftn3`` are documented as the inner kernels of
``make_sharded_{i,}fftn_3d`` (fft_helpers.py): call them directly ONLY from
code already inside a ``shard_map`` (shard_map cannot nest), or on genuinely
single-device / replicated operands.  Called EAGERLY on a (μ,ν)-sharded
global array they make jax gather the full operand onto every rank — an
N_μ²-class tile per rank, forbidden by the scaling doctrine.  That is audit
finding P0-4: ``bse_io._interpolate_bse_data_to_grid`` densified the coarse W
through eager ``local_ifftn3``/``local_fftn3`` and then papered the sharding
back on with a ``device_put``.  The fix is ``bse_io.make_w_densifier`` (ONE
sharded densifier, shard_map FFTs + jitted pad with ``out_shardings``); this
file is the gate that keeps the eager form from coming back.

THE RULE (syntactic proxy, tuned to zero false positives on the current
tree): inside ``src/bse`` and ``src/gw``, every call to ``local_ifftn3`` /
``local_fftn3`` must sit in an enclosing-function chain in which SOME
function also references a shard_map-constructing name (``shard_map`` in any
spelling, or ``make_sharded_*``) — i.e. the kernel is being used where a
shard_map is being built (e.g. ``bse_stack_matvec``'s ``_body`` under the
``build_bse_stack_matvec`` shard_map), not as a free-floating eager op.
Call sites that are legitimately eager (single-device reference paths,
replicated diagnostics) live in ``_SANCTIONED_EAGER`` below — a RATCHET per
QUALITY_PATTERNS: every entry pins its exact call count, a stale or
over-count entry fails the suite, and the list can only shrink.

The syntactic rule cannot PROVE the semantic claim (no all-gather in the
compiled program); ``tools/probe_w_densifier_hlo.py`` is the runtime HLO
probe for that, to be run on a compute node.

Pure AST — no jax, no src imports; runs on a login node
(``python3 tests/test_fft_shardmap_context.py``) and under pytest.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: Directories under src/ the gate patrols (the sharded solve families).
SCAN_DIRS = ("bse", "gw")

#: The shard_map-interior kernels being gated.
KERNELS = {"local_ifftn3", "local_fftn3"}


# ===========================================================================
# The ratchet: sanctioned EAGER call sites, (relpath, function-chain) → count.
# Every entry is existing, documented debt or a genuinely single-device path.
# An entry whose count changes in EITHER direction fails the suite: a new
# eager call cannot hide behind an old exception, and a fixed one cannot
# leave a stale excuse behind.  Numbers only go down.
# ===========================================================================
_SANCTIONED_EAGER = {
    # Single-device REFERENCE matvec ("single-device reference, values
    # byte-identical" — its own comment); operands are unsharded by contract.
    ("bse/bse_serial.py", "apply_bse_hamiltonian_single_device"): 3,
    # ζ box is a replicated fit product (host round-trip follows on the next
    # line); no (μ,ν)-sharded operand reaches this FFT.
    ("bse/vq_interp.py", "refit_vq"): 1,
    # (The former KNOWN-DEBT block — bse_feast.ensure_W_R, bse_kpm.run_kpm_dos,
    # bse_pseudopoles._feast_filter/run_pseudopoles — was closed 2026-08-01:
    # every solver W_q→W_R now routes through bse_feast.ensure_W_R →
    # bse_io.make_w_densifier.)
    # Diagnostics on tiny synthetic decks (smoke / correctness cross-checks),
    # run at sizes where a replicated W is intended and harmless.
    ("bse/bse_ring_comm.py", "ring_matvec_smoke_test"): 1,
    ("bse/bse_ring_comm.py", "ring_matvec_correctness_check"): 5,
}


# ===========================================================================
# The instrument
# ===========================================================================

def _is_shardmap_name(name: str) -> bool:
    """A name that constructs (or is) a shard_map'd FFT context."""
    return "shard_map" in name or name.startswith("make_sharded")


def _references_shardmap(fn_node: ast.AST) -> bool:
    """True iff the function's subtree references any shard_map-ish name."""
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Name) and _is_shardmap_name(node.id):
            return True
        if isinstance(node, ast.Attribute) and _is_shardmap_name(node.attr):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if _is_shardmap_name(alias.name.split(".")[-1]) or (
                        alias.asname and _is_shardmap_name(alias.asname)):
                    return True
    return False


def scan_eager_local_fft_calls(tree: ast.AST, relpath: str):
    """All KERNEL call sites with no shard_map context in their fn chain.

    Returns ``[(relpath, chain, lineno)]`` where ``chain`` is the enclosing
    function names joined by ``>`` (``"<module>"`` for top-level calls).  A
    call passes when ANY function in its enclosing chain references a
    shard_map-constructing name — the outer factory building the shard_map
    counts for the inner body it wraps.
    """
    findings = []

    class _V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []          # [(name, references_shardmap)]

        def _visit_fn(self, node):
            self.stack.append((node.name, _references_shardmap(node)))
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _visit_fn
        visit_AsyncFunctionDef = _visit_fn

        def visit_Call(self, node):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name in KERNELS:
                if not any(has for _, has in self.stack):
                    chain = ">".join(n for n, _ in self.stack) or "<module>"
                    findings.append((relpath, chain, node.lineno))
            self.generic_visit(node)

    _V().visit(tree)
    return findings


def _scan_tree():
    findings = []
    for d in SCAN_DIRS:
        for path in sorted((SRC / d).rglob("*.py")):
            relpath = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text(), filename=str(path))
            findings.extend(scan_eager_local_fft_calls(tree, relpath))
    return findings


# ===========================================================================
# The gate
# ===========================================================================

def test_no_eager_local_fft_outside_shardmap_context():
    findings = _scan_tree()
    counts = {}
    for relpath, chain, _ in findings:
        # ratchet key is the INNERMOST function name (stable under nesting)
        key = (relpath, chain.split(">")[-1])
        counts[key] = counts.get(key, 0) + 1

    unsanctioned = {
        f"{rel}:{line} in {chain}()"
        for rel, chain, line in findings
        if (rel, chain.split(">")[-1]) not in _SANCTIONED_EAGER
    }
    assert not unsanctioned, (
        "EAGER local_ifftn3/local_fftn3 call(s) outside any shard_map-"
        "constructing function chain — on a (μ,ν)-sharded operand this "
        "all-gathers an N_μ²-class tile per rank (audit P0-4).  Route "
        "through make_sharded_{i,}fftn_3d / bse_io.make_w_densifier, or if "
        "the operand is provably single-device/replicated, add a ratchet "
        "entry with a reason:\n  " + "\n  ".join(sorted(unsanctioned)))

    stale = {
        f"{key} expected {want}, found {counts.get(key, 0)}"
        for key, want in _SANCTIONED_EAGER.items()
        if counts.get(key, 0) != want
    }
    assert not stale, (
        "_SANCTIONED_EAGER is a ratchet and it no longer matches the tree "
        "(an excuse must not outlive its violation — shrink the entry):\n  "
        + "\n  ".join(sorted(stale)))


# ===========================================================================
# Red twins — the SAME scanner fed synthetic sources that must fail/pass.
# ===========================================================================

_RED_EAGER_IN_PLAIN_FN = '''
from common.fft_helpers import local_ifftn3

def densify(W_q, grid):
    W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")   # EAGER — must flag
    return W_R
'''

_RED_EAGER_AT_MODULE_LEVEL = '''
from common.fft_helpers import local_fftn3
W_G = local_fftn3(load(), axes=(1, 2, 3))                    # EAGER — must flag
'''

_GREEN_SHARDMAP_INTERIOR = '''
from jax.experimental.shard_map import shard_map
from common.fft_helpers import local_ifftn3, make_sharded_fftn_3d

def build_matvec(mesh, spec):
    fftn = make_sharded_fftn_3d(mesh, spec, spec)
    def _body(W_local):                      # shard_map interior: fine
        return local_ifftn3(W_local, axes=(2, 3, 4))
    return shard_map(_body, mesh=mesh, in_specs=(spec,), out_specs=spec), fftn
'''


def test_scanner_flags_eager_call_in_plain_function():
    hits = scan_eager_local_fft_calls(ast.parse(_RED_EAGER_IN_PLAIN_FN), "red.py")
    assert [(h[0], h[1]) for h in hits] == [("red.py", "densify")], (
        "red twin: the scanner failed to flag an eager local_ifftn3 call in "
        f"a plain function — the instrument is void, got {hits!r}")


def test_scanner_flags_module_level_eager_call():
    hits = scan_eager_local_fft_calls(
        ast.parse(_RED_EAGER_AT_MODULE_LEVEL), "red.py")
    assert [(h[0], h[1]) for h in hits] == [("red.py", "<module>")], (
        "red twin: the scanner failed to flag a module-level eager "
        f"local_fftn3 call, got {hits!r}")


def test_scanner_passes_shardmap_interior_call():
    hits = scan_eager_local_fft_calls(
        ast.parse(_GREEN_SHARDMAP_INTERIOR), "green.py")
    assert hits == [], (
        "green control: the scanner flagged a local_ifftn3 call whose "
        "enclosing chain constructs the shard_map — false positive, the "
        f"rule has drifted: {hits!r}")


# ===========================================================================
# Login-node runner (no pytest needed) — same shape as test_layering._main.
# ===========================================================================

def _main():
    n_run, failures = 0, []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        n_run += 1
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            msg = (str(exc).splitlines() or [""])[0]
            failures.append("FAIL %s — %s: %s" % (name, type(exc).__name__, msg))
            print(failures[-1])
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed"
          % (n_run - len(failures), n_run, len(failures)))
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
