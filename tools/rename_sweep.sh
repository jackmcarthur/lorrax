#!/usr/bin/env bash
# Rename name families across the tree: file and directory names (git mv) and
# file contents (ruplacer --preserve-case).  See RENAMING.md.
#
#   tools/rename_sweep.sh        old_name=new_name [old2=new2 ...]   # dry run
#   tools/rename_sweep.sh --go   old_name=new_name [old2=new2 ...]   # apply
#
# Pairs are snake_case and are applied in the order given, so put a longer
# name before a name it contains (shared_poles=tangential_poles before
# shared_pole=tangential).  --preserve-case also covers CamelCase, mixedCase,
# SCREAMING_CASE, kebab-case and Train-Case; this script adds the sentence-case
# kebab spelling (Shared-pole -> Tangential), which ruplacer leaves alone.
# Spellings with a space ("shared pole") are never touched: review them by hand.
#
# Never swept (git pathspecs below): manual/ (a generated copy of the manual,
# rebuilt by the manual's own publish step), UPGRADE_NOTES.md (history),
# src/common/renames.py (the alias table, which keeps the old spellings) and
# this script (its examples name the first rename; bash also reads a running
# script from disk, so a script that edits itself breaks mid-run).
# ruplacer's own -T globs match file basenames only, so they cannot skip a
# directory; the files are chosen with git instead and passed one at a time.
# Needs ruplacer >= 0.10 and ripgrep on PATH (RENAMING.md says where from).
set -euo pipefail
go=""
if [[ ${1:-} == --go ]]; then go=--go; shift; fi
(( $# )) || { sed -n 2,22p "$0"; exit 2; }
cd "$(git rev-parse --show-toplevel)"
skip=(':!manual' ':!UPGRADE_NOTES.md' ':!src/common/renames.py' ':!tools/rename_sweep.sh')

kebab() { echo "${1//_/-}"; }
sentence() { local k; k=$(kebab "$1"); echo "${k^}"; }

for pair in "$@"; do
  old=${pair%%=*}; new=${pair#*=}
  [[ $old && $new && $old != "$pair" ]] || { echo "bad pair: $pair (want old=new)"; exit 2; }
  echo "### $old -> $new"
  # 1. paths: every tracked path holding the snake or kebab form.
  { git ls-files -- . "${skip[@]}" | grep -E "$old|$(kebab "$old")" || true; } | while read -r path; do
      dest=$(echo "$path" | sed -e "s/$old/$new/g" -e "s/$(kebab "$old")/$(kebab "$new")/g")
      echo "  git mv $path $dest"
      if [[ $go ]]; then mkdir -p "$(dirname "$dest")"; git mv "$path" "$dest"; fi
    done
  # 2. contents, one file at a time (a dry run lists files under their old paths).
  joined=${old//_/}
  { git grep -l -I -i -E "${old//_/[-_]?}|$joined" -- . "${skip[@]}" || true; } | while read -r f; do
      ruplacer $go -e --color never --preserve-case -- "$old" "$new" "$f" | grep -v -E '^$|^Would perform|^Re-run|^Performed' || true
      ruplacer $go -e --color never --no-regex -- "$(sentence "$old")" "$(sentence "$new")" "$f" | grep -v -E '^$|^Would perform|^Re-run|^Performed' || true
    done
done

echo "### left for review (spaced or joined spellings, and anything the sweep skipped)"
for pair in "$@"; do
  old=${pair%%=*}
  git grep -n -I -i -E "${old//_/[-_ ]?}" -- . "${skip[@]}" || true
done
