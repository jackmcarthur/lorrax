# The BSE head-override deck reader is a second parser, and it stays one (2026-08-11)

**Not a defect and not a failure.** This records a deduplication that was considered,
measured, and deliberately not made, so that the next cleanup lane does not spend a leg
re-deriving the same answer.

## What looked duplicated

`bse/bse_head.py:_parse_head_overrides` hand-parses the `vhead` and `whead_0freq` keys
out of `cohsex.in` by scanning lines. Those are the same two deck keys the GW side reads
through `gw.gw_config.read_lorrax_input`, which builds a `configparser.ConfigParser` with
`inline_comment_prefixes=('#',)` and is the parser `docs/input_reference.md` is generated
from. Two readers of one key is a standing hazard and it has already bitten once: until
2026-08-09 the BSE reader did not strip inline comments, so a deck written
`vhead = 3303.748102  # BGW` pinned the GW side's head and silently left the BSE using the
restart's own, which is one run screening with two different q=0 heads.

## What the measurement showed

Both readers were run over the same five decks at `72945497`. On a deck carrying a
`[cohsex]` section they agree on the value, with or without an inline comment — that much
is already pinned by `tests/test_bse_head_override.py::test_agrees_with_the_gw_side_reader`.
They diverge in two ways that matter:

| deck | `_parse_head_overrides` | `read_lorrax_input` |
|---|---|---|
| `vhead = 1.0d3` | `ValueError` naming the file, the line number and the key, and explaining that the key pins a cross-code comparison | `ValueError: could not convert string to float: '1.0d3'` |
| head keys with no `[cohsex]` section | reads the override | returns `None` |

The refusal message is the whole point of the feature rather than a nicety. `1.0d3` is
Fortran exponent notation, which is exactly what a hand carrying a number over from a
BerkeleyGW input writes, and a validation knob that fails without naming itself is close
to a knob that no-ops. Three tests in `tests/test_bse_head_override.py` pin this:
`test_malformed_value_refuses_rather_than_falling_back` matches on the phrase
`head override`, and `test_refusal_names_the_file_and_line` asserts both `cohsex.in:3`
and the key name appear. Swapping in `read_lorrax_input` fails all of them.

## Why it was left alone

Beyond the message, `read_lorrax_input` returns the full defaults dictionary and runs the
retired-key and unknown-key reports as a side effect, so calling it here would make the
BSE head resolution print a deck audit and would pull `gw.gw_config` and its import chain
into `bse_head` — a module both restart loaders import. That is a large import-graph move
in exchange for no behavioural gain, on a seam where the two readers are already held to
each other by an explicit agreement test.

**The contract that matters is enforced, and it is enforced by the test rather than by
sharing code.** If an owner wants one reader instead of two, the shape is to give
`gw_config` a small, side-effect-free entry point that reads named keys and refuses with
provenance, and then point both sides at it — not to call the full deck parser from the
BSE.

**Status: filed, no action taken. Owner call.**
