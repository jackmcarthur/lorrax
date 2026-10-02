"""Static gate for the documentation tree: the site navigation and the register.

Two properties, both read from source text with no mkdocs install:

1. **Every published page is in the navigation, and every navigation entry
   exists.** A published page is a ``docs/**/*.md`` file outside the
   directories ``mkdocs.yml`` lists under ``exclude_docs``. A page missing
   from ``nav`` is built but unreachable from the site's menus; a ``nav``
   entry with no file breaks the build.
2. **Every link in the register resolves.** The register is the section of
   ``docs/index.md`` headed ``## Where each fact lives``. Every relative link
   in its tables must name an existing file, and an anchor must exist on that
   page: an explicit ``{#id}``, an ``<a id="…">``, or a heading whose
   Python-Markdown ``toc`` slug equals it. The register is the one page every
   other page defers to, so a dead row sends readers nowhere.

Runnable with plain ``python3 tests/test_docs_register.py`` (no jax); also
collects under pytest. Each auditor has a negative control.
"""
import os
import re
import unicodedata

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCS = os.path.join(_REPO, "docs")
_MKDOCS = os.path.join(_REPO, "mkdocs.yml")
_INDEX = os.path.join(_DOCS, "index.md")


# ---------------------------------------------------------------------------
# mkdocs.yml, read as text (no yaml dependency)
# ---------------------------------------------------------------------------

def excluded_dirs(text):
    """Directory prefixes from the ``exclude_docs: |`` block, e.g. ``dev/``."""
    out, inside = [], False
    for line in text.splitlines():
        if re.match(r"^exclude_docs:\s*\|", line):
            inside = True
            continue
        if inside:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            entry = line.strip()
            if entry and not entry.startswith("#"):
                out.append(entry.lstrip("/"))
    return out


def nav_pages(text):
    """Every ``*.md`` path named in the ``nav:`` block, in order."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"^nav:\s*$", l))
    pages = []
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith((" ", "\t", "-")):
            break
        m = re.search(r"(?:^\s*-\s*|:\s*)([\w./-]+\.md)\s*$", line)
        if m:
            pages.append(m.group(1))
    return pages


def published_pages(docs_root, excluded):
    pages = []
    for root, _dirs, files in os.walk(docs_root):
        for name in files:
            if name.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, name), docs_root)
                rel = rel.replace(os.sep, "/")
                if not any(rel.startswith(d.rstrip("/") + "/") for d in excluded):
                    pages.append(rel)
    return sorted(pages)


def nav_problems(mkdocs_text, docs_root):
    excluded = excluded_dirs(mkdocs_text)
    nav = nav_pages(mkdocs_text)
    published = published_pages(docs_root, excluded)
    problems = ["published page not in mkdocs nav: docs/%s" % p
                for p in published if p not in nav]
    problems += ["mkdocs nav entry has no file: docs/%s" % p
                 for p in nav if not os.path.isfile(os.path.join(docs_root, p))]
    return problems


# ---------------------------------------------------------------------------
# the register
# ---------------------------------------------------------------------------

def slugify(text):
    """Python-Markdown ``toc``'s default slug of a heading's text."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def anchors(markdown):
    found = set(re.findall(r"\{#([\w-]+)\}", markdown))
    found |= set(re.findall(r"<a\s+id=\"([\w-]+)\"", markdown))
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = None if in_fence else re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            heading = re.sub(r"\s*\{#[\w-]+\}\s*$", "", m.group(1))
            found.add(slugify(heading))
    return found


def register_rows(index_text):
    start = index_text.index("## Where each fact lives")
    nxt = re.search(r"^## ", index_text[start + 3:], re.M)
    section = index_text[start:start + 3 + nxt.start()] if nxt else index_text[start:]
    return [l for l in section.splitlines() if l.startswith("| **")]


def register_problems(index_text, docs_root):
    rows = register_rows(index_text)
    problems = [] if rows else ["register: no table rows found"]
    for row in rows:
        for target in re.findall(r"\]\(([^)\s]+)\)", row):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            full = os.path.normpath(os.path.join(docs_root, path)) if path else _INDEX
            if not os.path.isfile(full):
                problems.append("register link to a missing file: %s" % target)
                continue
            if anchor and anchor not in anchors(open(full, encoding="utf-8").read()):
                problems.append("register link to a missing anchor: %s" % target)
    return problems


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_every_published_page_is_in_the_nav():
    problems = nav_problems(open(_MKDOCS, encoding="utf-8").read(), _DOCS)
    assert not problems, "\n".join(problems)


def test_every_register_link_resolves():
    problems = register_problems(open(_INDEX, encoding="utf-8").read(), _DOCS)
    assert not problems, "\n".join(problems)


def test_nav_auditor_can_fail(tmp_path=None):
    import tempfile
    root = tmp_path or tempfile.mkdtemp()
    docs = os.path.join(str(root), "docs")
    os.makedirs(os.path.join(docs, "dev"))
    for rel in ("index.md", "orphan.md", "dev/note.md"):
        open(os.path.join(docs, rel), "w").write("# x\n")
    mk = "exclude_docs: |\n  dev/\n\nnav:\n  - Home: index.md\n  - Gone: gone.md\n"
    problems = nav_problems(mk, docs)
    assert any("orphan.md" in p for p in problems), problems
    assert any("gone.md" in p for p in problems), problems
    assert not any("dev/note.md" in p for p in problems), problems


def test_register_auditor_can_fail(tmp_path=None):
    import tempfile
    root = tmp_path or tempfile.mkdtemp()
    docs = os.path.join(str(root), "docs")
    os.makedirs(docs, exist_ok=True)
    open(os.path.join(docs, "a.md"), "w").write("# A page\n\n## Real `anchor` here\n")
    index = ("## Where each fact lives\n\n| q | owner | what |\n|---|---|---|\n"
             "| **ok** | [A](a.md#real-anchor-here) | x |\n"
             "| **bad file** | [B](b.md) | x |\n"
             "| **bad anchor** | [A](a.md#nope) | x |\n")
    problems = register_problems(index, docs)
    assert len(problems) == 2, problems
    assert any("b.md" in p for p in problems) and any("#nope" in p for p in problems)


def _main():
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures.append("FAIL %s: %s" % (name, exc))
                print(failures[-1])
            else:
                print("ok   %s" % name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
