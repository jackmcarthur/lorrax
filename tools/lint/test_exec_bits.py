"""Every script that is run by path keeps mode 100755 in git.

``src/ffi/cpp/select_gpu.sh`` lost its exec bit twice in a week and every
``srun ... select_gpu.sh`` launch died with "Permission denied" before Python
started.  A file edited on a filesystem without modes (or recreated by a tool)
comes back 644, and nothing noticed.  This lint reads the modes git records
(``git ls-files -s``) and requires 100755 for a tracked file with a shebang that

* a shell line or a fenced docs command runs by path (first word of a command
  is the file: ``config/frontera/build_mpi_overlay.sh fetch``,
  ``"$ROOT/tools/rename_sweep.sh"``, ``./01-link_files.sh``), or
* sits where scripts live: ``config/*/build_*``, ``config/*/install*``,
  numbered example steps ``examples/**/[0-9]*-*.sh``, ``scripts/``, ``tools/``,
  or is named in :data:`ALWAYS`.

Files that are SOURCED, never executed, stay 644 and are listed in
:data:`SOURCED`; each entry must still match a tracked file (a stale exemption
fails, as in ``test_layering``).
"""
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PLACED = ("config/*/build_*", "config/*/install*", "examples/**/[0-9]*-*.sh",
          "scripts/*", "tools/*")
ALWAYS = ("src/ffi/cpp/select_gpu.sh", "src/ffi/cpp/build_host.sh", "config/frontera/stage_host_pmi.sh")
#: Sourced into the caller's shell (``source`` / ``.``), never executed.
SOURCED = ("*_env.sh", "config/perlmutter/site_config.sh", "config/frontera/stage_runtime.sh",
           "config/local/setup_env.sh", "config/local/stack.sh")
SHELLS = (".sh", ".run", ".sbatch", ".bash")
#: Leading words that run the next word as a command (``srun -n 4 ./x.sh``).
PREFIX = {"exec", "time", "then", "do", "else", "srun", "mpirun", "mpiexec", "env", "nohup", "!"}
ROOTS = re.compile(r"^(\$\{?(ROOT|REPO|LORRAX_ROOT|LORRAX_CHECKOUT|LORRAX_SRC|REPO_ROOT)\}?|"
                   r"\$\(git rev-parse --show-toplevel\))/")
HERE = re.compile(r"^\$\{?(HERE|SCRIPT_DIR|DIR|_here|here)\}?/")


def _tracked():
    """``{path: mode}`` from git's index; skip outside a git checkout."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "-s"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return {line.split("\t", 1)[1]: line.split()[0] for line in out.splitlines()}


def _matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(Path(path).name, p) for p in patterns)


def _command_lines(path, text):
    """Shell lines of a script, or the lines of a doc's fenced code blocks."""
    if path.endswith(SHELLS):
        yield from text.splitlines()
        return
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif fenced:
            yield line


def _run_by_path(path, text, tracked):
    """Tracked files the command lines of ``path`` run by path."""
    base = Path(path).parent
    for line in _command_lines(path, text):
        line = line.split(" #", 1)[0].strip()
        if line.startswith("#"):
            continue
        for cmd in re.split(r"&&|\|\||;|\|", line):
            words = cmd.strip().lstrip("$ ").split()
            while words and (words[0] in PREFIX or re.match(r"^[A-Za-z_]\w*=", words[0])
                             or (words[0].startswith("-") and len(words) > 1)):
                words = words[1:]
            if not words:
                continue
            tok = words[0].strip("\"'`(")
            if "/" not in tok:
                continue
            if ROOTS.match(tok):
                cands = [ROOTS.sub("", tok)]
            elif HERE.match(tok):
                cands = [str(base / HERE.sub("", tok))]
            else:
                cands = [str(base / tok), tok]
            for c in cands:
                c = str(Path(c.removeprefix("./")))
                c = re.sub(r"(^|/)[^/]+/\.\.(/|$)", r"\1", c)
                if c in tracked:
                    yield c
                    break


def _offenders(tracked, read):
    """``(path, why)`` for each shebang file that must be 100755 and is not."""
    def shebang(p):
        return read(p)[:2] == "#!"
    need = {}
    for p in tracked:
        if _matches(p, PLACED) or p in ALWAYS:
            need.setdefault(p, "lives where scripts live")
    for p in tracked:
        if p.endswith(SHELLS) or p.endswith(".md"):
            for hit in _run_by_path(p, read(p), tracked):
                need.setdefault(hit, f"run by path in {p}")
    return sorted((p, why) for p, why in need.items()
                  if tracked[p] != "100755" and not _matches(p, SOURCED) and shebang(p))


def _read(p):
    try:
        return (ROOT / p).read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""


def test_scripts_run_by_path_are_executable():
    tracked = _tracked()
    assert len(tracked) > 500, f"read only {len(tracked)} tracked files: is ROOT right ({ROOT})?"
    bad = _offenders(tracked, _read)
    assert not bad, (
        "scripts that are run by path but are not 100755 in git; fix with "
        "`git update-index --chmod=+x <file>` (or list a sourced file in SOURCED):\n  "
        + "\n  ".join(f"{p}  ({why})" for p, why in bad))


def test_sourced_exemptions_are_still_needed():
    tracked = _tracked()
    stale = [s for s in SOURCED if not any(_matches(p, (s,)) for p in tracked)]
    assert not stale, f"SOURCED entries that match no tracked file: {stale}"


def test_lint_sees_a_lost_exec_bit():
    files = {
        "src/ffi/cpp/select_gpu.sh": ("100644", "#!/bin/bash\n"),
        "config/x/build_thing.sh": ("100755", "#!/bin/bash\n"),
        "config/x/gpu_env.sh": ("100644", "#!/bin/bash\n"),
        "examples/a/b/01-link_files.sh": ("100644", "#!/bin/bash\n"),
        "examples/a/00-run.sh": ("100755", "#!/bin/bash\nb/01-link_files.sh\nsource x/gpu_env.sh\n"),
        "docs/x.md": ("100644", "run it:\n```bash\n$ config/x/fetch.sh --once && echo ok\n```\n"
                                 "`config/x/other.sh` in prose\n"),
        "config/x/fetch.sh": ("100644", "#!/bin/bash\n"),
        "config/x/other.sh": ("100644", "#!/bin/bash\n"),
        "config/x/data.sh": ("100644", "no shebang\n"),
    }
    bad = _offenders({p: m for p, (m, _) in files.items()}, lambda p: files[p][1])
    assert [p for p, _ in bad] == ["config/x/fetch.sh", "examples/a/b/01-link_files.sh",
                                   "src/ffi/cpp/select_gpu.sh"], bad
