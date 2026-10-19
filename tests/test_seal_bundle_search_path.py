"""seal_bundle.py refuses a search path a published bundle cannot carry (CPU only, no ELF).

A leg needs one DT_RUNPATH that starts with $ORIGIN; no library may search a relative,
scratch or build directory, including one reached through a symlink.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "seal_bundle", ROOT / "src" / "ffi" / "cpp" / "stage" / "seal_bundle.py")
seal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seal)


@pytest.mark.parametrize("search, leg", [
    ([("RUNPATH", "$ORIGIN")], True),
    ([("RUNPATH", "${ORIGIN}:/opt/site/lib")], True),
    ([("RPATH", "$ORIGIN:$ORIGIN/../lib:/usr/site/lib")], False),
    ([], False),
])
def test_accepts(search, leg):
    seal._refuse_search_path("lib.so", search, root=ROOT, leg=leg)


@pytest.mark.parametrize("search, leg", [
    ([("RPATH", "$ORIGIN")], True),                       # DT_RPATH outranks LD_LIBRARY_PATH
    ([("RUNPATH", "/opt/site/lib:$ORIGIN")], True),       # $ORIGIN not first
    ([], True),
    ([("RUNPATH", "$ORIGIN:lib")], False),                # relative: resolves against the cwd
    ([("RUNPATH", "$ORIGIN:/pscratch/u/stage/lib")], False),
    ([("RPATH", "/home/u/sw/.build/cusolvermp/lib")], False),
    ([("RUNPATH", f"$ORIGIN:{ROOT}/lib")], False),        # inside the sealing checkout
])
def test_refuses(search, leg):
    with pytest.raises(RuntimeError):
        seal._refuse_search_path("lib.so", search, root=ROOT, leg=leg)


def test_refuses_scratch_through_a_symlink(tmp_path, monkeypatch):
    # Only $SCRATCH counts as scratch here, so the literal path is not under it and
    # only its real path is.
    monkeypatch.setattr(seal, "_SCRATCH_ROOTS", ())
    for var in ("PSCRATCH", "TMPDIR"):
        monkeypatch.delenv(var, raising=False)
    target = tmp_path / "scratch"
    target.mkdir()
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "link").symlink_to(target)
    monkeypatch.setenv("SCRATCH", str(target))
    seal._refuse_search_path("lib.so", [("RUNPATH", f"$ORIGIN:{tmp_path}/home/lib")],
                             root=ROOT, leg=True)
    with pytest.raises(RuntimeError):
        seal._refuse_search_path("lib.so", [("RUNPATH", f"$ORIGIN:{tmp_path}/home/link/lib")],
                                 root=ROOT, leg=True)
