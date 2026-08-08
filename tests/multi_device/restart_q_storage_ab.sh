#!/usr/bin/env bash
# restart_q_storage full-vs-wedge A/B — Perlmutter driver.
#
# The gate `4e8cfd70` (the restart producer) names as its own: si_bse_debug run
# at MATCHED settings with `restart_q_storage = full` and again with `auto`,
# with the BSE eigenvalues AND the on-disk tensors diffed between the arms.  It
# lives here rather than in the pytest suite because SlabIO needs the phdf5 FFI
# `.so` pair: every restart-writer cell in the tree is red on a box without it,
# so until this ran the q_irr format's BYTES had never been measured anywhere.
#
# The measurement contract and the physics numbers are in
# `restart_q_storage_ab.py`'s module docstring.  Read that first.  Its two
# requirements, one line each, because getting either wrong turns this from a
# measurement into a story:
#
#   * MATCHED SETTINGS.  The frozen reference moves by meV under an iteration
#     or process count change with no code change at all, so only a matched
#     A/B cancels the fragility; the reference is a sanity line, never the
#     acceptance criterion.
#   * THE WEDGE ARM'S BSE LEG IS P=1.  The sharded reader refuses a wedge file
#     by design (`bse_io._MunuSlabPlan`); only the serial h5py reader unfolds.
#     The P=4 leg runs on both arms anyway — on `full` because that is the
#     configuration the frozen reference was cut at, and on `auto` because the
#     refusal is itself an assertion.
#
# Needs: a GPU allocation, `LX_BASE_MODULE=lorrax_J070`, and the BUILD_NOTES
# `.so` pin for the tree under test.  All three live in the LXRUN hook, which
# is the ONE launcher-specific thing in this file — point it at your own
# wrapper elsewhere.
#
#   LXRUN=/path/to/lxr.sh bash tests/multi_device/restart_q_storage_ab.sh
#
# where `$LXRUN <ngpu> <nranks> <workdir> <cmd...>` runs <cmd> on <ngpu> GPUs
# as <nranks> processes with <workdir> as cwd.  GIVE THAT HOOK A WAIT: on a
# shared pool the launcher REFUSES rather than queues when the GPUs are busy,
# and a refusal is not a leg that failed, it is a leg that never ran.  This
# script aborts on one instead of writing an empty row (measured the hard way
# on 2026-08-08: a first pass produced a whole table of blank walls because a
# sibling agent held the node).
#
# THE WALL SWEEP, and what it is for.  The third arm sets
# `write_restart_tensors = false` (`67eda567`), the suppress key whose default
# is an OPEN OWNER DECISION; this driver produces the number that decision
# needs.  It is a round-robin over the arms so a drift in node state hits all
# three equally instead of landing on whichever ran last, and every leg clears
# tmp/ first, because gw_jax REUSES an existing tmp/zeta_q.h5 and prints
# "FIT SKIPPED" — an uncleared repeat does LESS work and its wall is not a
# repeat of anything.
#
# MEASURED 2026-08-08, lx job 56499811, 1 node x 4 A100 (nid001028), branch
# `svc/symmetry_maps-followup-2026-08-08` @ 54d25712, BUILD_NOTES merge_ckpt
# `.so` pair, 5 reps after one unreported warm-up leg per arm, mean +/- sample
# stdev over the 5:
#
#   arm       TOTAL (wall) s     persist_w0 s      write phase s
#   full      22.263 +/- 1.145   0.325 +/- 0.007   1.376 +/- 0.036
#   auto      22.320 +/- 0.935   0.096 +/- 0.004   1.171 +/- 0.158
#   nowrite   21.568 +/- 0.981   0.000 +/- 0.000   0.627 +/- 0.068
#
# "write phase" is `gw_jax.isdf` SELF time — which is where the V / G0 / enk /
# psi writes and the W0 placeholder land, there being no finer timer around
# them — plus `gw_jax.persist_w0`.  Read the three rows in that order of
# confidence:
#
#   * `persist_w0` is a clean isolated timer and the sharpest number here:
#     0.325 -> 0.096 s for the wedge (3.4x) and -> 0.000 s suppressed.  The
#     wedge cuts W0's BYTES by 8x and its TIME by 3.4x, the gap being the
#     fixed per-write cost — dataset create, collective open, H5Fclose — that
#     does not shrink with the payload.
#   * the write phase says the suppress key saves 0.749 s and the wedge saves
#     0.205 s of it, the latter inside its own scatter.
#   * THE TOTAL STEP DOES NOT RESOLVE EITHER CHANGE on this deck, and saying so
#     is the honest report: 0.695 s of mean separation between `full` and
#     `nowrite` against ~1.0 s of run-to-run scatter.  si_bse_debug spends ~20
#     of its ~22 s on imports, runtime bring-up and 219 XLA compiles, so a
#     0.5 GB write is 6% of it.  The spec's "~21% of wall" is the PRODUCTION
#     deck's ratio (2.01 GB), not this one's, and this campaign does not
#     measure that deck.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FIX="$REPO/tests/regression/si_bse_debug"
W="${RESTART_AB_WORKDIR:-$REPO/.restart_q_storage_ab}"
LXRUN="${LXRUN:-}"
REPS="${RESTART_AB_REPS:-5}"
INPUT=bse_si_test.in

# Solver settings, verbatim from tests/test_bse_bgw_regression.py.  MATCHED
# across arms on purpose — see the header.
BSE_FLAGS=(--bse --lanczos --tda --matvec-kind=ring
           --n-val 4 --n-cond 4 --n-occ 8 --n-reorth -1
           --max-lanczos-iter 200 --n-eig 20)

if [ -z "$LXRUN" ] || [ ! -x "$LXRUN" ]; then
    echo "restart_q_storage_ab.sh: set LXRUN to an executable launcher hook" >&2
    echo "  usage: \$LXRUN <ngpu> <nranks> <workdir> <cmd...>" >&2
    exit 2
fi

# A launcher refusal is not a result.  lx reserves 90-98 for its own states
# (93 ALLOCFAIL, 96 POOLFULL, 98 EXPIRED); anything in that band means the leg
# never started, and continuing past it fills the table with blanks that read
# like measurements.
#
# THE SETTLE BEFORE A MULTI-PROCESS LEG IS NOT SUPERSTITION.  Back-to-back
# multi-rank steps race on the JAX distributed coordination service: the new
# step's tasks reach RegisterTask while the previous step's coordinator is
# still up and get "unexpectedly tried to connect with a different
# incarnation", after which the whole step aborts at 134/143.  Measured on
# 2026-08-08, and the reason it is guarded rather than tolerated is WHAT it
# swallowed: it took out the `auto` arm's P=4 leg, whose entire job is to
# REFUSE, and reported an abort where the refusal should have been.  An arm
# that must refuse is exactly the arm where a spurious failure is invisible.
run_leg () {                      # run_leg <ngpu> <nranks> <dir> <log> <cmd...>
    local g="$1" n="$2" d="$3" log="$4"; shift 4
    [ "$n" -gt 1 ] && sleep 10
    "$LXRUN" "$g" "$n" "$d" "$@" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ge 90 ] && [ "$rc" -le 98 ]; then
        echo "*** launcher refused this leg (rc=$rc) — it never ran." >&2
        echo "*** Give \$LXRUN a wait (lx: --wait N) or an allocation of your" >&2
        echo "*** own, then re-run.  Refusing to report a partial table." >&2
        exit "$rc"
    fi
    return "$rc"
}

# --------------------------------------------------------------------------
# arms: the committed fixture, plus ONE deck line each
# --------------------------------------------------------------------------
setup_arms () {
    local a
    for a in full auto nowrite; do
        rm -rf "$W/arms/$a"; mkdir -p "$W/arms/$a"
        cp "$FIX"/WFN.h5 "$FIX"/kin_ion.h5 "$FIX"/$INPUT \
           "$FIX"/centroids_frac_480_orbitclosed.txt \
           "$FIX"/bse_eigenvalues_ref.dat "$FIX"/bgw_eigenvalues_dft_ref.dat \
           "$W/arms/$a"/
        chmod u+w "$W/arms/$a"/*
        printf '\n# --- restart_q_storage_ab arm: %s ---\n' "$a" \
            >> "$W/arms/$a/$INPUT"
    done
    echo 'restart_q_storage = full' >> "$W/arms/full/$INPUT"
    echo 'restart_q_storage = auto' >> "$W/arms/auto/$INPUT"
    # The suppress-key arm pins `full` so the ONLY variable against the `full`
    # arm is the suppress key itself.
    printf 'restart_q_storage = full\nwrite_restart_tensors = false\n' \
        >> "$W/arms/nowrite/$INPUT"
    md5sum "$W/arms"/*/centroids_frac_480_orbitclosed.txt
}

# GW leg twice, tmp/ cleared before EACH.  The second is the warm one.
gw_arm () {
    local a="$1" leg
    for leg in 1 2; do
        rm -rf "$W/arms/$a/tmp"
        run_leg 4 4 "$W/arms/$a" "$W/arms/$a/gw$leg.log" \
            python -u -m gw.gw_jax -i $INPUT
        echo "  gw$leg rc=$?  $(grep -E 'TOTAL \(wall\)' \
            "$W/arms/$a/gw$leg.log" | tail -1)"
    done
    # Read the source-tree line back.  A stale LORRAX_CHECKOUT silently ran a
    # DIFFERENT worktree on 2026-08-08 and produced a convincing false physics
    # red; this log line is the only thing that catches it.
    grep -E '^\[lx\] source tree:' "$W/arms/$a/gw2.log" | head -1
    grep -E '\[restart_write\] (restart_q_storage|write_restart_tensors)' \
        "$W/arms/$a/gw2.log" | head -2
    ls -l "$W/arms/$a"/tmp/isdf_tensors_*.h5 2>&1
}

# P=1 + TDA is the serial-h5py branch of _preview_lanczos — the only reader
# that unfolds a wedge.  The physics A/B is measured on this leg.
bse_p1 () {
    local a="$1"
    run_leg 1 1 "$W/arms/$a" "$W/arms/$a/bse_p1.log" \
        python -u -m bse.bse_jax -i $INPUT "${BSE_FLAGS[@]}" --px 1 --py 1
    echo "  bse_p1 rc=$?"
    grep -A4 'Lowest 20 eigenvalues (eV)' "$W/arms/$a/bse_p1.log" | tail -5
}

# P=4 is the configuration the frozen reference was cut at.  On `full` it must
# succeed; on the other two it must REFUSE, and the refusals are asserted.
bse_p4 () {
    local a="$1"
    run_leg 4 4 "$W/arms/$a" "$W/arms/$a/bse_p4.log" \
        python -u -m bse.bse_jax -i $INPUT "${BSE_FLAGS[@]}" --px 2 --py 2
    echo "  bse_p4 rc=$?"
}

# Round-robin so node drift hits every arm equally instead of landing on
# whichever arm ran last.
sweep () {
    local r a tot pw0 isdf sz
    for a in full auto nowrite; do
        rm -rf "$W/arms/$a/tmp"
        run_leg 4 4 "$W/arms/$a" "$W/arms/$a/sweep_warmup.log" \
            python -u -m gw.gw_jax -i $INPUT
    done
    for r in $(seq 1 "$REPS"); do
        for a in full auto nowrite; do
            rm -rf "$W/arms/$a/tmp"
            run_leg 4 4 "$W/arms/$a" "$W/arms/$a/sweep_r$r.log" \
                python -u -m gw.gw_jax -i $INPUT
            tot=$(grep -E 'TOTAL \(wall\)' "$W/arms/$a/sweep_r$r.log" \
                  | tail -1 | awk '{print $3}')
            isdf=$(grep -E '^gw_jax\.isdf ' "$W/arms/$a/sweep_r$r.log" \
                   | awk '{print $3, $4}')
            pw0=$(grep -E '^gw_jax\.persist_w0 ' "$W/arms/$a/sweep_r$r.log" \
                  | awk '{print $4}')
            sz=$(stat -c %s "$W/arms/$a"/tmp/isdf_tensors_*.h5 2>/dev/null \
                 || echo 0)
            echo "rep=$r arm=$a total=$tot isdf(tot,self)=$isdf" \
                 "persist_w0=$pw0 restart_bytes=$sz"
        done
    done
}

mkdir -p "$W/arms"
echo "===== setup ====="; setup_arms
for a in full auto nowrite; do echo "===== GW: $a ====="; gw_arm "$a"; done
for a in full auto; do echo "===== BSE P=1: $a ====="; bse_p1 "$a"; done
echo "===== BSE P=4: full (the frozen reference's configuration) ====="
bse_p4 full
grep -A4 'Lowest 20 eigenvalues (eV)' "$W/arms/full/bse_p4.log" | tail -5
echo "===== BSE P=4: auto (MUST refuse — the sharded reader cannot unfold) ====="
bse_p4 auto
grep -m1 -o '_MunuSlabPlan: kgrid.*' "$W/arms/auto/bse_p4.log" \
    || echo "*** the sharded reader did NOT refuse the wedge — investigate" >&2
echo "===== BSE P=4: nowrite (MUST refuse — the file is absent) ====="
bse_p4 nowrite
grep -m1 -o 'FileNotFoundError: Could not find.*' "$W/arms/nowrite/bse_p4.log" \
    || echo "*** no FileNotFoundError — investigate" >&2
echo "===== wall sweep ($REPS reps, round-robin) ====="; sweep
echo "===== compare ====="
run_leg 1 1 "$W" "$W/compare.log" \
    python -u "$REPO/tests/multi_device/restart_q_storage_ab.py" \
    compare "$W/arms/full" "$W/arms/auto" \
    --ref "$FIX/bse_eigenvalues_ref.dat"
rc=$?
cat "$W/compare.log"
exit "$rc"
