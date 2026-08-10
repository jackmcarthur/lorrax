# Archived Documentation

This folder contains older documentation that has been **superseded** by the comprehensive docs. These files are kept for historical reference but may contain outdated, incomplete, or redundant information.

## Superseded Files

| Old File | Superseded By | Notes |
|----------|---------------|-------|
| `formalism.md` | `../PHYSICS_COMPREHENSIVE.md` §1-3 | Terse, missing spin details |
| `isdf_context.md` | `../PHYSICS_COMPREHENSIVE.md` §2-3 | Incomplete (cuts off mid-sentence) |
| `cohsex_jax_physics.md` | `../PHYSICS_COMPREHENSIVE.md` §6-7 | Verbose, streamlined in comprehensive doc |
| `ZETA_FITTING_ALGORITHM.md` | `../PHYSICS_COMPREHENSIVE.md` §4-5 | Redundant with comprehensive doc |
| `isdf_spin_galerkin_derivation.md` | `../PHYSICS_COMPREHENSIVE.md` §3 | CCT/ZCT derivation already included |
| `ctsp_revised.md` | `../MINIMAX_QUADRATURE.md` | Overlaps with minimax quadrature reference |
| `cold_start_2026-07.md` | `../../environment/machines/frontera.md` §3 | Point-in-time measurement record (jobs 7882055/7882076/7882070/7882139) — preserved verbatim; the operative recipe moved |
| `HANDOFF_2026-07-28.md`, `HANDOFF_2026-07-29.md`, `HANDOFF_cpu_frontera_2026-07.md` | current docs + `SPEEDUP_SCORECARD.md` | Point-in-time campaign handoffs |

## Current Documentation

For up-to-date information, see:

### Primary Guides
- **Physics/theory**: [`physics.md`](../../theory/physics.md)
- **Codebase structure**: [`codebase.md`](../../architecture/codebase.md)
- **Environment setup**: [`environment/overview.md`](../../environment/overview.md)

### Specialized Topics
- **Memory model**: [`memory-model.md`](../../architecture/memory-model.md)
- **CTSP quadrature / minimax**: [`minimax-quadrature.md`](../../theory/minimax-quadrature.md)

### Other Resources
- **Reference papers**: [`misc/references/`](misc/references/)

Two files that used to be listed here as current resources are not current, and
as of 2026-08-09 each carries a banner saying so: `../notes/AGENT_TODO.md`
(written against a `src/isdf/` package layout the tree no longer has) and
`../notes/advanced_README_legacy.md` (the README of a folder that no longer
exists). Both belong with the superseded files above rather than under current
documentation, and both are candidates for deletion.
