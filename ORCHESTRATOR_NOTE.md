# ORCHESTRATOR NOTE — 2026-08-16 (rev 2, supersedes rev 1)

Rev 1 told you the 0.99 threshold was wrong and to use nonzero-fp-weight
instead. That was too strong. Here is the settled position.

## The two candidate rules AGREE EXACTLY on an insulator

  * rule A (owner's words):  include a band iff occupancy < 0.99
  * rule B (exact):          include a band in a branch iff its fp weight
                             in that branch is nonzero

On a gapped system every occupancy is exactly 1.000 or 0.000, so both rules
exclude exactly the same bands and produce identical numbers. **Your audit is
scoped to the insulator, so this choice CANNOT affect your merge verdict.**
Do not spend time adjudicating it, and do not let it delay Tasks 2 and 3.

They differ only on metals, where MP1 smearing puts bands between 0.99 and
1.000: rule A drops their small nonzero contribution, rule B keeps it. That
is a numerics-vs-cost tradeoff and it is the owner's call, not yours and not
mine. It has been put to him.

## What to do

1. **Localize.** Find where the Green's-function band range consumed by Sigma
   is decided on this branch, and state with file:line whether metallic
   support widened it unconditionally on gapped decks. This is the finding
   that matters and it is rule-independent.
2. **Quantify the waste**, if there is any: bands carried that contribute
   exactly zero, and what they cost in wall time and memory on the Si deck.
   A zero-weight band contributes exactly zero, so on the insulator this is a
   COST question, not a correctness one — report it that way.
3. **Propose the minimal diff** implementing the restriction. Write it against
   rule B (exact, nonzero weight) since it is correct under both readings and
   collapses to rule A on every gapped deck; note in your report that rule A
   differs only on metals and is pending an owner decision.
4. **Do NOT land anything on `integ/metal-mpa-qsgw-2026-08-15`.** Commit to
   your own branch off `41742d17`. The branch owner will take your diff or
   your localization and land it themselves, coordinated with their merge.

## Unchanged, and now confirmed as the highest-value item

The branch owner has confirmed they have **no** insulator evidence in their
lane and that the `src/gw/sigma_dispatch.py` bit-for-bit projector comment is
an assertion with no test behind it. **Your audit is the first thing to
exercise it, and their merge to main is now HELD on your verdict.** If the
claim is false it is a hard merge blocker and they want the failing case
verbatim — exact states, exact numbers, exact tolerance.
