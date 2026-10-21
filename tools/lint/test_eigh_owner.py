"""Gate: one owner for the local Hermitian eigensolver.

``distrib_la.eigh_local`` is the only place src code may reach an eigensolver
(it carries the per-leg implementation row, e.g. the ROCm reflector route for
complex operands).  A direct ``jnp.linalg.eigh`` call bypasses that row.

Pure tokenizer -- no jax, no src imports; runs on a login node
(``python3 tools/lint/test_eigh_owner.py``) and under pytest.
"""
import io
import pathlib
import tokenize

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def test_src_calls_no_jnp_eigh_directly():
    """One owner for the local eigensolver: src code calls distrib_la.eigh_local."""
    offenders = []
    for path in SRC.rglob("*.py"):
        toks = list(tokenize.generate_tokens(io.StringIO(path.read_text()).readline))
        for i in range(len(toks) - 4):
            if [t.string for t in toks[i:i + 5]] == ["jnp", ".", "linalg", ".", "eigh"]:
                offenders.append(f"{path.relative_to(SRC)}:{toks[i].start[0]}")
    assert not offenders, "call distrib_la.eigh_local instead of jnp.linalg.eigh: " + ", ".join(offenders)


# ===========================================================================
# Login-node runner (no pytest needed) -- same shape as test_layering._main.
# ===========================================================================

def _main():
    n_run, failures = 0, []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        n_run += 1
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 -- this IS the tally
            msg = (str(exc).splitlines() or [""])[0]
            failures.append("FAIL %s -- %s: %s" % (name, type(exc).__name__, msg))
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
