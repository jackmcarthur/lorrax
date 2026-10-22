# Renaming a name across LORRAX

How a deck key, a deck value, a file name or an identifier is renamed
everywhere, with the old spelling still accepted. First used for the
shared-pole W model, which became the tangential W model (2026-10-21). The
BerkeleyGW flag renames reuse it: add rows, run the same commands.

## Tools

- `ruplacer` 0.10 (find and replace across a tree, `--preserve-case` covers
  snake_case, CamelCase, mixedCase, SCREAMING_CASE, kebab-case, Train-Case),
  `ripgrep` and `fd`. Release binaries, checksums verified, into `~/.local/bin`:
  `your-tools/ruplacer` v0.10.0 (`x86_64-unknown-linux-gnu`),
  `BurntSushi/ripgrep` 15.2.0 and `sharkdp/fd` v10.5.0 (`x86_64-unknown-linux-musl`).
  `cargo install ruplacer` works too.
- `tools/rename_sweep.sh`: `git mv` for paths, then ruplacer file by file. It
  never touches `manual/`, `UPGRADE_NOTES.md`, `src/common/renames.py` or
  itself; `RENAME_SKIP=":!path ..."` adds more git pathspecs.
- `src/common/renames.py`: the one table of old spellings that are still
  accepted (`DECK_KEYS`, `DECK_VALUES`, `DISK_NAMES`) and the spellings kept on
  purpose (`FROZEN`). The GW and TD deck readers, the restart-member reader,
  the store's schema checks and the run-directory lookup go through it; an old
  spelling logs one NOTE (`renamed_name`) and runs under the new name.

## Procedure

1. **Inventory.** List every spelling and every file name (ripgrep reads stdin
   when no path is given, so always pass `.` in a script):

       rg -n -i 'old[-_ ]?name|oldname' . -g '!manual/**'
       fd -i 'old.?name'

   Sort the hits: deck keys and values, names on disk (HDF5 datasets,
   schemas, run-directory names), GATE ids and log text, Python identifiers,
   prose. Anything that feeds a stored hash or a version tag a restart
   compares stays as it is: add it to `FROZEN`.
2. **Alias rows** in `src/common/renames.py` for every user-visible old
   spelling (deck keys, deck values, names on disk).
3. **Record the frozen hashes** before the sweep (for this rename:
   `RECIPE_HASH`, `GATE_HASH`, `ORDERED_GATE_HASH`, `CHARGE4_*_GATE_HASH` and
   the version tags of `gw.tangential_recipe`).
4. **Sweep**, dry run first, then apply. Pairs are applied in order, so a
   name goes before any shorter name it contains:

       tools/rename_sweep.sh      shared_poles=tangential_poles shared_pole=tangential
       tools/rename_sweep.sh --go shared_poles=tangential_poles shared_pole=tangential

   Commit the sweep output alone, so it can be regenerated (step 7).
5. **By hand**: what the sweep leaves (its last section lists it). Spaced
   spellings ("shared pole") are read in context: physics prose stays as it is.
   Restore any frozen text the sweep changed. Then wire the
   alias rows into any reader that is new, regenerate
   `docs/input_reference.md` (`python tools/gen_input_reference.py`) and add
   an `UPGRADE_NOTES.md` entry.
6. **Verify**:
   - the old spelling appears only in `renames.py`, `UPGRADE_NOTES.md` and the
     `FROZEN` spellings (`rg -n -i 'old[-_ ]?name' . -g '!manual/**'`);
   - the frozen hashes are byte-identical to step 3;
   - the lint suites (`python tools/lint/test_*.py`, as `tools/release_check.sh`);
   - no new undefined names: `uvx pyflakes src services tools tests`, compared
     with the same run on `origin/main`;
   - the unit layer: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests`;
   - an old deck and an old run directory still run (the hsuite chain, on an
     allocation).
7. **Landing on a moved main.** Do not rebase the sweep commit: rerun step 4 on
   the new main, restore the `FROZEN` text the sweep changed (for this rename,
   the one gate-table line, `gw/tangential_recipe.py`, "new shared-pole
   objects"), check the hashes, then cherry-pick the hand commit. Every other
   open branch that touches the renamed files needs the same sweep before it
   merges.

## The manual and the flowcharts

The manual source (`~/projects/lorrax_docs_site`) has no git. Copy
`docs/ mkdocs.yml hooks/ figures_src/ CODE_CHANGES.md` into a scratch repo,
commit that snapshot, branch, and run the same sweep there
(`bash <lorrax>/tools/rename_sweep.sh --go ...`). Then read the spaced
spellings by hand and build with `.venv/bin/mkdocs build --strict -f <copy>/mkdocs.yml`.
A renamed page keeps its old URL through `hooks/redirects.py` (`REDIRECTS`).
Apply the diff to the source when the code lands, then
`publish.sh --deploy`. `manual/` in this repo is that deploy's output, so it is
never swept here.

The flowcharts (`~/projects/lorrax_flowcharts`) take the same sweep, with
`RENAME_SKIP=":!REVISIONS_R3.md"` for the revision logs. A spec whose visible
labels changed is re-rendered (`bash render/r.sh <ids>`).

## Gotchas met on the first run

- ruplacer's `-T/--type-not` globs match file basenames only, and the option
  swallows the arguments after it. Choose files with git pathspecs instead,
  as the sweep does.
- A tracked script that names the old spelling edits itself mid-run; bash
  reads a running script from disk. The sweep skips itself.
- Words in a hashed table are data: "new shared-pole objects" in the gate
  table feeds `GATE_HASH`, so it stays.
