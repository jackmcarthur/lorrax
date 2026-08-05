# Frontera campaign notes

Measurement and design notes written on TACC Frontera between 2026-06 and
2026-08.  Committed here because they lived only on `/scratch2`, which is
purgeable, while shipped code cites them: a grep for `.md` filenames across
`src/`, `tests/`, `config/` and `tools/` returns 155 distinct names, and a
substantial fraction resolved only to the scratch copy.

Provenance:

| here | was |
| --- | --- |
| `notes/` | `/scratch2/08271/jackmc/lorrax_setup/docs/` |
| `wk_REL/` | `/scratch2/08271/jackmc/lorrax_setup/wk_REL/docs/` |
| `wk_REL/reference/` | `/scratch2/08271/jackmc/lorrax_setup/wk_REL/reference/` |
| `LORRAX_CONTEXT_BRIEF.md` | `/scratch2/08271/jackmc/lorrax_setup/` |
| `LORRAX_FRONTERA_ADVICE.md` | `/work2/08271/jackmc/frontera/` |
| `FFI_BUILD_CANONICAL.md` | `/work2/08271/jackmc/frontera/lorrax_ffi_unified/CANONICAL.md` |

These are a record of what was measured and decided at the time, not current
documentation.  Where one contradicts `docs/`, `docs/` wins.  Several record
REFUTED results — `wk_REL/patches/` holds patches kept as evidence that an
approach did not work, and they are not meant to be applied.

`FFI_BUILD_CANONICAL.md` is the exception that stays live: it records which
`liblorrax_ffi_host.so` build is canonical and why each other one is
superseded.  Update it in place rather than treating it as history.
