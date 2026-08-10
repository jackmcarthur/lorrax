# SPEC — an unknown deck key should REFUSE, not log-and-proceed (2026-08-10) — **PROPOSAL, NOTHING IMPLEMENTED**

**This is a specification for the deck-parser / loader owners, not a change and not a
defect report.** It is registered here because measurement-discipline rule 1 in
`AGENT_PREAMBLE.md` cites it, and a rule that cites an unregistered corollary decays.

**The incident** (peer fleet, 2026-08-10, reported to this lane rather than reproduced
by it — treat as reported until an owner reproduces it): a worktree one commit behind
the deck key it was testing ran **both arms of an A/B flag-off**. The key was unknown
to that tree's parser, the parser logged and proceeded, and the A/B came back green
having measured nothing. The `[lx] source tree:` line was correct throughout, which is
why it is only the necessary half of the instrument check.

**The proposal.** An unknown key in a deck is a refusal at parse time, naming the key,
before any stage runs.

**Why this is an owner ruling and not a patch.** Warn-and-ignore is currently
*deliberate* for retired keys — `docs/index.md` records `use_ffi_io = false` becoming
warn-and-ignore on 2026-08-06 when the three I/O tiers collapsed to one transport. A
blanket refusal would break every deck carrying a retired key. So the ruling needed is
the shape of the split, not whether to refuse:

| option | behaviour | cost |
|---|---|---|
| A — retired-key allowlist | known-retired keys warn; everything else refuses | one allowlist to maintain, and it is the record of what was retired |
| B — strict by env | refuse under `LORRAX_DECK_STRICT=1`, warn otherwise; A/B harnesses set it | opt-in, so the default still measures nothing when a lane forgets |
| C — refuse on unknown, warn on retired-and-dated | as A, but each allowlist row carries the date and the commit that retired it | most work, and the only one that also answers "when did this key stop meaning anything" |

**Interim, binding on lanes now, and it needs no code:** per measurement-discipline
rule 1, grep every A/B log for `ignored` and for unknown-key lines, and run
`git rev-parse HEAD` inside the leg. An A/B whose logs were not grepped for those has
not established that its arms differ.

**Owners:** the deck parser (`src/gw/gw_init.py`) and whoever owns
`docs/input_reference.md`, which is generated from that parser. **Status: OPEN,
awaiting an owner ruling between A, B and C.**
