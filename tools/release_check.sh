#!/bin/bash
# release_check.sh — one command reproducing the pre-push bar.
#
# Login-node checks (always run; python3 3.7 is enough — the suites are the
# AST gates, which run via their __main__ runners without jax):
#   1-5. the five login AST suites: test_layering, test_crossfile_requests,
#        test_env_registry, test_env_grammar, test_fft_shardmap_context
#   6.   tools/gen_input_reference.py drift check (the tool refuses on key
#        drift vs gw_config._DEFAULTS; a content diff vs the committed
#        docs/input_reference.md also fails — regenerate and commit).
#        The pre-run file is restored either way: the check never mutates
#        the tree.
#   7.   origin-delta blob scan: no blob > 1 MB introduced since $BASE_REF
#   8.   secrets grep over the added lines of the $BASE_REF..HEAD diff
#   9.   git status cleanliness (porcelain empty — no uncommitted work)
#
# --with-allocation additionally submits the fastloop mini-deck chain in
# check mode (sbatch, 1 dev job) and waits for its verdict.  Exit codes of
# the fastloop runner: 0 pass | 1 parity drift | 2 stage failure | 3 refusal.
#
# Exit: 0 = every check passed; 1 = at least one failed.  A one-line-per-
# check summary always prints at the end.
#
# Env overrides:
#   RC_BASE_REF          delta base (default origin/main)
#   RC_FASTLOOP_SBATCH   fastloop script (default: the sandbox copy)
#   RC_FASTLOOP_TIMEOUT  seconds to wait for the fastloop job (default 3600)

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

BASE_REF="${RC_BASE_REF:-origin/main}"
FASTLOOP_SBATCH="${RC_FASTLOOP_SBATCH:-/scratch2/08271/jackmc/lorrax_sandbox/fastloop/run_fastloop.sbatch}"
FASTLOOP_TIMEOUT="${RC_FASTLOOP_TIMEOUT:-3600}"
WITH_ALLOCATION=0
for arg in "$@"; do
  case "$arg" in
    --with-allocation) WITH_ALLOCATION=1 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg (only --with-allocation)"; exit 1 ;;
  esac
done

NAMES=()
VERDICTS=()
FAILED=0
record() {  # record <name> <rc> [detail]
  NAMES+=("$1")
  if [ "$2" -eq 0 ]; then VERDICTS+=("PASS")
  else VERDICTS+=("FAIL${3:+ — $3}"); FAILED=1; fi
}

# --- 1-5. the five login AST suites ---------------------------------------
for t in test_layering test_crossfile_requests test_env_registry \
         test_env_grammar test_fft_shardmap_context; do
  out=$(python3 "tests/$t.py" 2>&1); rc=$?
  tail_line=$(printf '%s\n' "$out" | tail -1)
  record "$t" "$rc" "$tail_line"
  [ "$rc" -ne 0 ] && printf '%s\n' "$out" | tail -20
done

# --- 6. input-reference drift ---------------------------------------------
REF_DOC="docs/input_reference.md"
SAVED=$(mktemp); cp "$REF_DOC" "$SAVED"
gen_out=$(python3 tools/gen_input_reference.py 2>&1); gen_rc=$?
if [ "$gen_rc" -ne 0 ]; then
  record "input_reference drift" 1 "generator refused (key drift): ${gen_out}"
elif ! git diff --quiet -- "$REF_DOC"; then
  record "input_reference drift" 1 \
    "regenerated $REF_DOC differs from the committed one — rerun tools/gen_input_reference.py and commit"
  git --no-pager diff --stat -- "$REF_DOC"
else
  record "input_reference drift" 0
fi
cp "$SAVED" "$REF_DOC"; rm -f "$SAVED"   # never mutate the tree

# --- 7. origin-delta blob scan (> 1 MB) -----------------------------------
big=$(git rev-list --objects "$BASE_REF"..HEAD 2>/dev/null \
      | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
      | awk '$1=="blob" && $3>1048576 {printf "%.1fMB %s\n", $3/1048576, $4}')
if [ -z "$big" ]; then
  record "blob scan ($BASE_REF..HEAD, >1MB)" 0
else
  record "blob scan ($BASE_REF..HEAD, >1MB)" 1 "$(printf '%s' "$big" | wc -l | tr -d ' ') blob(s)"
  printf '%s\n' "$big"
fi

# --- 8. secrets grep over the added lines of the delta --------------------
# Pattern set: private-key blocks, AWS/GitHub/GitLab/Slack/OpenAI-style
# tokens, and assignment-shaped credentials.
SECRET_RE='BEGIN (RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|glpat-[A-Za-z0-9_-]{20}|xox[baprs]-[A-Za-z0-9-]+|sk-[A-Za-z0-9]{40,}|(password|passwd|api_key|apikey|secret_key|auth_token|access_token)[[:space:]]*[=:][[:space:]]*["'"'"'][^"'"'"']{6,}'
hits=$(git diff "$BASE_REF"..HEAD | grep -E '^\+[^+]' | grep -cEi "$SECRET_RE")
if [ "$hits" -eq 0 ]; then
  record "secrets grep ($BASE_REF..HEAD)" 0
else
  record "secrets grep ($BASE_REF..HEAD)" 1 "$hits added line(s) match — inspect before pushing"
  git diff "$BASE_REF"..HEAD | grep -E '^\+[^+]' | grep -Ei "$SECRET_RE" | head -5
fi

# --- 9. git status cleanliness --------------------------------------------
dirty=$(git status --porcelain)
if [ -z "$dirty" ]; then
  record "git status clean" 0
else
  record "git status clean" 1 "$(printf '%s\n' "$dirty" | wc -l | tr -d ' ') path(s) uncommitted"
  printf '%s\n' "$dirty" | head -10
fi

# --- optional: fastloop check-mode on an allocation -----------------------
if [ "$WITH_ALLOCATION" -eq 1 ]; then
  if [ ! -f "$FASTLOOP_SBATCH" ]; then
    record "fastloop check" 1 "sbatch script not found: $FASTLOOP_SBATCH"
  else
    # sbatch --parsable prints the job id as its last line.
    jid=$(FASTLOOP_MODE=check sbatch --parsable "$FASTLOOP_SBATCH" 2>&1 | tail -1)
    case "$jid" in
      *[!0-9]*|"")
        record "fastloop check" 1 "sbatch failed: $jid" ;;
      *)
        echo "[release_check] fastloop check job $jid submitted; waiting (cap ${FASTLOOP_TIMEOUT}s)..."
        waited=0
        while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
          sleep 30; waited=$((waited + 30))
          if [ "$waited" -ge "$FASTLOOP_TIMEOUT" ]; then break; fi
        done
        flout="$(dirname "$FASTLOOP_SBATCH")/work/fastloop.$jid.out"
        if squeue -h -j "$jid" 2>/dev/null | grep -q .; then
          record "fastloop check (job $jid)" 1 "still running after ${FASTLOOP_TIMEOUT}s — read $flout"
        elif [ ! -f "$flout" ]; then
          record "fastloop check (job $jid)" 1 "no output file at $flout"
        else
          flrc=$(grep -o '\[fastloop rc=[0-9]*\]' "$flout" | tail -1 | grep -o '[0-9]*')
          record "fastloop check (job $jid)" "${flrc:-1}" \
            "runner rc=${flrc:-unknown} (0 pass|1 drift|2 stage|3 refusal) — $flout"
        fi ;;
    esac
  fi
fi

# --- summary ---------------------------------------------------------------
echo
echo "=== release_check summary ($(git rev-parse --short HEAD) vs $BASE_REF) ==="
i=0
while [ "$i" -lt "${#NAMES[@]}" ]; do
  printf '  %-42s %s\n' "${NAMES[$i]}" "${VERDICTS[$i]}"
  i=$((i + 1))
done
if [ "$FAILED" -eq 0 ]; then
  echo "=== ALL CHECKS PASSED ==="
else
  echo "=== AT LEAST ONE CHECK FAILED (exit 1) ==="
fi
exit "$FAILED"
