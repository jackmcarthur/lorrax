# Renaming a name across LORRAX

**The rule.** Renaming a public identifier, a deck key or value, a CLI flag, or
an HDF5 group, dataset or attribute goes through the alias table
`src/runtime/renames.py` plus the sweep below. Never through an ad-hoc `sed`.
The old name stays readable: it is accepted with one NOTE (`renamed_name`)
saying what to write instead, and the run continues under the new name.

First used for the shared-pole W model, which became the tangential W model
(2026-10-21). The BerkeleyGW flag renames reuse it: add rows, run the same
commands.

## The alias table

`src/runtime/renames.py` lives in `runtime`, the bottom layer, so every
package may import it. It holds:

| table | key | read by |
|---|---|---|
| `DECK_KEYS` | `(deck, old key)` | `gw_config.read_lorrax_input`, `td_config.parse_td_deck`, before any other check |
| `DECK_VALUES` | `(deck, new key, old value)` | the same readers, and `coerce_bispinor_gw_mode` |
| `CLI_FLAGS` | `(driver module or "*", "--old")` | `runtime.read_verbose_switches`, which every driver runs at start-up, before its parser |
| `DISK_NAMES` | old name | readers of HDF5 datasets, header schemas and run-directory suffixes (`disk_name`, `disk_names`, `run_dir_entry`, `suffix_pattern`) |
| `FROZEN` | spelling kept on purpose | nothing: it documents what the sweep must not change |

A row's value is the new name, or a `Rename(new, note=..., convert=...)` when
the meaning moved as well as the name: `note` is appended to the NOTE, and
`convert` maps the old value to the new one (a BerkeleyGW band count that
counts from another origin, say). A deck-key converter may return
`{new key: value, ...}` to fill several keys from one old key. A flag's
converter takes and returns the argument string; `--old`, `--old=v` and
`--old v` are all rewritten. A rename to a key that the deck also names under
its new spelling refuses (`GATE renamed_key_twice`).

## Tools

- `ruplacer` 0.10 (find and replace across a tree; `--preserve-case` covers
  snake_case, CamelCase, mixedCase, SCREAMING_CASE, kebab-case, Train-Case),
  `ripgrep` and `fd`. Release binaries, checksums verified, into `~/.local/bin`:
  `your-tools/ruplacer` v0.10.0 (`x86_64-unknown-linux-gnu`),
  `BurntSushi/ripgrep` 15.2.0 and `sharkdp/fd` v10.5.0 (`x86_64-unknown-linux-musl`).
  `cargo install ruplacer` works too.
- `tools/rename_sweep.sh`: `git mv` for paths, then ruplacer file by file. It
  never touches `manual/`, `UPGRADE_NOTES.md`, `src/runtime/renames.py`, this
  page or itself; `RENAME_SKIP=":!path ..."` adds more git pathspecs.

## Procedure

1. **Inventory.** List every spelling and every file name (ripgrep reads stdin
   when no path is given, so always pass `.` in a script):

       rg -n -i 'old[-_ ]?name|oldname' . -g '!manual/**'
       fd -i 'old.?name'

   Sort the hits: deck keys and values, CLI flags, names on disk (HDF5
   datasets, schemas, run-directory names), GATE ids and log text, Python
   identifiers, prose. Anything that feeds a stored hash or a version tag a
   restart compares stays as it is: add it to `FROZEN`.
2. **Alias rows** in `src/runtime/renames.py` for every user-visible old
   spelling. A reader of a new kind of name calls the helpers there.
3. **Record the frozen hashes** before the sweep (for the tangential rename:
   `RECIPE_HASH`, `GATE_HASH`, `ORDERED_GATE_HASH`, `CHARGE4_*_GATE_HASH` and
   the version tags of `gw.tangential_recipe`).
4. **Sweep**, dry run first, then apply. Pairs are applied in order, so a
   name goes before any shorter name it contains:

       tools/rename_sweep.sh      shared_poles=tangential_poles shared_pole=tangential
       tools/rename_sweep.sh --go shared_poles=tangential_poles shared_pole=tangential

   Commit the sweep output alone, so it can be regenerated (step 7).
5. **By hand**: what the sweep leaves (its last section lists it). Spaced
   spellings ("shared pole") are read in context: physics prose stays as it is.
   Restore any frozen text the sweep changed. Regenerate
   `docs/input_reference.md` (`python tools/gen_input_reference.py`) and add an
   `UPGRADE_NOTES.md` entry.
6. **Verify**:
   - the old spelling appears only in `renames.py`, `UPGRADE_NOTES.md`, this
     page and the `FROZEN` spellings (`rg -n -i 'old[-_ ]?name' . -g '!manual/**'`);
   - the frozen hashes are byte-identical to step 3;
   - the lint suites (`python tools/lint/test_*.py`, as `tools/release_check.sh`);
   - no new undefined names: `uvx pyflakes src services tools tests`, compared
     with the same run on `origin/main`;
   - the unit layer: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests`;
   - **an old deck and an old run directory**, read by the new code, on a
     copy. A deck: `gw_config.read_lorrax_input(<old deck>)` prints the NOTE
     and returns the new spelling. A restart: the production entry point
     (for the tangential rename, `gw.tangential_recipe.tangential_restart_handle`,
     with `JAX_ENABLE_X64=1` and a CPU mesh) authenticates the old bundle.
     Done for the tangential rename on a Frontier ZnO run of 2026-10-16
     (`spx_ab/r2_plain`): `sp_export.in` and `bisp_sc.in` load with one NOTE
     each; the restart bundle's `shared_pole_member` and its
     `lorrax.shared-real-pole.v1` model authenticate, payload digest included,
     from `oneshot_shared_pole/`, used in place.
7. **Landing on a moved main.** Do not rebase the sweep commit: rerun step 4 on
   the new main, restore the `FROZEN` text the sweep changed (for the
   tangential rename, the one gate-table line in `gw/tangential_recipe.py`,
   "new shared-pole objects"), check the hashes, then cherry-pick the hand
   commit. Every other open branch that touches the renamed files needs the
   same sweep before it merges.

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
- An old deck can also stop on a key retired some other way:
  `head_correction = no_local_fields` in the 2026-10-16 decks refuses on
  current main. A removal that should keep old decks running belongs in this
  table too, as a `Rename` with a converter.
