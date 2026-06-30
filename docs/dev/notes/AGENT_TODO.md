# Agent TODO Suggestions

**NOTE**: These are **suggested improvements** identified by AI agents during code analysis. They are **NOT the user's current priorities**. This document serves as a parking lot for ideas that may be useful in the future.

---

## 🏗️ Code Structure Refactoring

### High Priority

**Problem**: Two massive files with too many responsibilities
- `src/isdf/common/load_wfns.py`: 2796 lines, 39 functions
- `src/gw/gw_jax.py`: 2389 lines, 30 functions

#### Suggested: Split `load_wfns.py` into module package

```
src/isdf/common/wfn_loading/
├── __init__.py          # Public API exports
├── fft_backend.py       # shard_map FFT (lines 73-96)
├── nufft_backend.py     # NUFFT wrappers (lines 97-249)
├── pair_density.py      # P_k,ab computation (lines 552-664)
├── transforms.py        # k↔R transforms (lines 748+)
├── cct_zct.py          # CCT/ZCT accumulation
└── fitting.py          # fit_zeta_chunked_to_h5 orchestrator
```

**Benefits**:
- Each module ~300-500 lines
- Clear separation of concerns
- Easier testing and maintenance
- Agents can read only relevant modules

#### Suggested: Extract sub-modules from `gw_jax.py`

```
src/gw/
├── gw_jax.py       # Main driver only (~500 lines)
├── sigma_compute.py    # get_sigma_static_*, project_potential_to_bands
└── restart_pipeline.py # restart load/validation/stage resume helpers
```

**Benefits**:
- Main driver becomes ~500 lines, much more readable
- Clear module responsibilities
- Easier to locate and modify specific functionality

---

## 🧹 Code Quality Improvements

### Environment Variable Management

**Current issue** (gw_jax.py lines 3-13): Module-level side effects

```python
# ❌ BAD: Side effects at module import
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
```

**Suggested**:
```python
# ✅ GOOD: Explicit configuration function
# src/isdf/config/jax_env.py
def configure_jax_environment(enable_x64=True, platforms="cuda,cpu", ...):
    """Configure JAX environment variables before JAX import."""
    os.environ.setdefault("JAX_ENABLE_X64", "1" if enable_x64 else "0")
    ...

# Call explicitly in main() or __init__.py
```

### Global State Reduction

**Current issue** (gw_jax.py line 72): Global mesh makes testing harder

```python
# ❌ BAD: Global mesh
mesh_bands = Mesh(np.asarray(_default_devices), ("bands",))
```

**Suggested**:
```python
# ✅ GOOD: Pass mesh as parameter or use context manager
def create_mesh_context(devices=None):
    devices = devices or jax.devices()
    return Mesh(np.asarray(devices), ("bands",))
```

### Function Size Guidelines

**Target**: Functions should be <100 lines
**Max**: No function >200 lines without strong justification

**Likely violations**:
- `fit_zeta_chunked_to_h5`
- `main()`
- `compute_sigma_pipeline_jax`

**Suggested**: Extract helper functions when logic becomes complex

### Type Hints & Documentation

**Current**: Many functions lack type hints

**Suggested**: Add comprehensive type hints
```python
def compute_CCT_from_left_right(
    left: jax.Array,  # (nk, 2, nmu)
    right: jax.Array,  # (nk, 2, nnu)
    *,
    fft_norm: str = "ortho"
) -> jax.Array:  # (nk, nmu, nnu)
    """Compute CCT matrix from pair density components.

    Args:
        left: Left pair density component, shape (n_k, 2, n_mu)
        right: Right pair density component, shape (n_k, 2, n_nu)
        fft_norm: FFT normalization ('ortho', 'forward', 'backward')

    Returns:
        CCT matrix of shape (n_k, n_mu, n_nu)
    """
```

---

## 🧪 Testing Infrastructure

### Suggested Test Organization

- Each module should have corresponding `test_<module>.py`
- Use fixtures for expensive setup (wavefunction loading, etc.)
- Add unit tests for pure functions (transforms, pair density)
- Integration tests for full pipelines

### Suggested Test Files

```
tests/
├── unit/
│   ├── test_pair_density.py
│   ├── test_transforms.py
│   ├── test_cct_zct.py
│   └── test_sigma_compute.py
├── integration/
│   ├── test_zeta_fitting_pipeline.py
│   └── test_cohsex_full.py
└── fixtures/
    └── conftest.py  # Shared fixtures
```

---

## 📦 Code Organization

### Constants Module

**Suggested**: `src/isdf/common/constants.py`
```python
DEFAULT_FFT_NORM = "ortho"
DEFAULT_MEMORY_BUFFER_GB = 2.0
MAX_CHUNK_SIZE_DEFAULT = 100
HARTREE_TO_EV = 27.211386245988
```

### Error Handling

**Suggested**: Custom exceptions
```python
# src/isdf/common/exceptions.py
class ISDFError(Exception):
    """Base exception for ISDF package."""

class ChunkingError(ISDFError):
    """Raised when chunking constraints cannot be satisfied."""

class ConvergenceError(ISDFError):
    """Raised when iterative solver fails to converge."""
```

### Logging Instead of Print

**Current**: Many `print()` statements scattered throughout

**Suggested**: Use Python logging
```python
import logging
logger = logging.getLogger(__name__)

# Instead of print()
logger.info("Computing zeta for q-point %d/%d", iq, nq)
logger.warning("Memory usage exceeds threshold: %.1f GB", mem_gb)
logger.debug("CCT matrix shape: %s", CCT.shape)
```

---

## 📚 Documentation Gaps

### Suggested Additional Docs

- **TESTING_GUIDE.md** — How to run tests, what's tested, benchmark suite
- **CONTRIBUTING.md** — Contribution guidelines, code style, PR process
- **TROUBLESHOOTING.md** — Common errors and solutions
- **API_REFERENCE.md** — Auto-generated from docstrings (via pdoc/mkdocs)

### Suggested Docstring Coverage

- All public functions should have docstrings
- All public classes should have docstrings
- Module-level docstrings explaining purpose

---

## 🚀 Performance Optimization Ideas

### Profiling Infrastructure

**Suggested**: Add profiling utilities
```python
# src/isdf/common/profiling.py
@contextmanager
def profile_section(name: str):
    """Profile a code section and log results."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name}: {elapsed:.2f}s")
```

### Memory Tracking

**Suggested**: Add memory monitoring
```python
def log_memory_usage(tag: str):
    """Log current GPU/CPU memory usage."""
    # JAX device memory
    # System memory
    # Log with tag
```

---

## 📊 Data Validation

### Input Validation

**Suggested**: Validate inputs at public API boundaries
```python
def validate_wfn_file(path: str) -> None:
    """Validate WFN.h5 file format and required fields."""
    # Check file exists
    # Check required datasets
    # Check shapes are consistent
    # Raise helpful error messages
```

### Runtime Assertions

**Suggested**: Add shape/dtype assertions in critical paths
```python
def compute_CCT(P_k: jax.Array) -> jax.Array:
    assert P_k.ndim == 5, f"Expected 5D array, got {P_k.ndim}D"
    assert P_k.shape[1] == P_k.shape[2] == 2, "Expected spin dimension = 2"
    # ... computation
```

---

## Priority Tiers (Agent Suggestion)

**Phase 1: Documentation** (~1-2 hours)
- Already completed: Restructured docs with archive/, references/, advanced/
- Add missing docs: TESTING_GUIDE.md, TROUBLESHOOTING.md

**Phase 2: Easy Wins** (~2-4 hours)
- Extract environment configuration
- Add type hints to public functions
- Replace print() with logging
- Add constants module

**Phase 3: Major Refactoring** (~1-2 weeks)
- Split load_wfns.py into package
- Refactor gw_jax.py sub-modules
- Remove global state
- Add comprehensive test suite

---

**Remember**: These are suggestions, not requirements. Prioritize based on actual development needs.
