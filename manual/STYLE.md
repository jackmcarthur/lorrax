# Style conventions for manual writing

Match the first four paragraphs of the repo `README.md`. They are the reference for
every section of this manual. Read them immediately before writing and imitate them
directly: sentence rhythm, punctuation habits, and information density, not a vague
"tone".

That prose is dense declarative sentences, each adding new technical content.
Tradeoffs appear in the same sentence as the feature they qualify ("significant
speedups, with potential memory tradeoffs, relative to the canonical $O(N^4)$
scaling plane-wave formalism"). Limitations are stated flatly ("Symmetries are not
used in the evaluation of the quasiparticle energies and will not reduce
computational cost"). Novelty is claimed once with the word "novel" and never
argued. Short lists go inline as "1.) ..., and 2.) ...". Parentheses carry status
and asides: (WIP), (default), (Ry). Punctuation is commas, colons, and parentheses;
the reference prose contains no em dashes, and several per paragraph is a reliable
sign the writing has drifted back to machine register.

The failure mode to avoid is decoration. A bullet must not open with a bolded catchy
label. A header says what a person would say ("Pros of LORRAX", not "Where LORRAX
wins"). Never narrate ("this section explains", "worth remembering", "a fair summary
for the impatient") and never restate what the previous paragraph established. In
lists and tables, fragments are correct and complete sentences are padding. Bullets
are for genuinely enumerable things (files, flags, steps), nothing conceptual.

House facts: the full name is "the LOw-scaling Real-space Real-Axis eXcited state
package in BerkeleyGW". Quantum ESPRESSO is the "DFT starting point", never the
"front end". No code besides Quantum ESPRESSO and BerkeleyGW is named; write "the
majority of $O(N^3)$-scaling GW codes" and similar. Forward references are bare
parentheticals like (§5.4). Bold marks file paths and input keys, nothing else.

When in doubt, cut the sentence in half and drop the label. Sections 1.1 and 1.3
are the reference implementations.
