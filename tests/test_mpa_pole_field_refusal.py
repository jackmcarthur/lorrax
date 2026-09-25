"""``mpa_store.refuse_bad_pole_fields``: one program, same refusals.

The check used to run eagerly, one field-sized temporary per ``jnp`` op, and
died allocating 6.44 GiB on a CrI3 8x8 GN-PPM SC field.  It is now one
jitted reduction.  These cells hold both halves: the refusals still fire,
and the compiled program's scratch does not scale with the field.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")
jnp = jax.numpy

from file_io.mpa_store import _pole_field_counts, refuse_bad_pole_fields  # noqa: E402

_SHAPE = (3, 4, 16, 16)          # (n_p, n_q, mu, nu)


def _field(rng):
    omega = (rng.uniform(0.1, 2.0, _SHAPE)
             - 1j * rng.uniform(0.01, 0.1, _SHAPE))
    residue = rng.normal(size=_SHAPE) + 1j * rng.normal(size=_SHAPE)
    return jnp.asarray(omega), jnp.asarray(residue)


def test_a_healthy_field_passes_and_a_dormant_pole_may_sit_at_zero():
    omega, residue = _field(np.random.default_rng(0))
    omega = omega.at[0, 0, 0, 0].set(0.0)
    residue = residue.at[0, 0, 0, 0].set(0.0)
    refuse_bad_pole_fields(omega, residue, where="cell")
    refuse_bad_pole_fields(omega, residue, jnp.zeros_like(residue), where="cell")


@pytest.mark.parametrize("odd", [False, True])
def test_a_nonfinite_element_refuses(odd):
    omega, residue = _field(np.random.default_rng(1))
    odd_residue = jnp.zeros_like(residue) if odd else None
    bad = residue.at[1, 2, 3, 4].set(np.nan)
    with pytest.raises(ValueError, match="refuses 1 non-finite"):
        refuse_bad_pole_fields(omega, bad, odd_residue, where="cell")
    if odd:
        with pytest.raises(ValueError, match="refuses 1 non-finite"):
            refuse_bad_pole_fields(
                omega, residue, odd_residue.at[0, 0, 0, 1].set(np.inf),
                where="cell")


def test_a_live_pole_off_the_causal_sheet_refuses():
    omega, residue = _field(np.random.default_rng(2))
    with pytest.raises(ValueError, match="refuses 2 live poles"):
        refuse_bad_pole_fields(
            omega.at[0, 1, 2, 3].set(-0.5 - 0.1j).at[2, 3, 4, 5].set(0.5 + 0.1j),
            residue, where="cell")
    # The odd residue makes a pole live even where the even residue is zero.
    zero_even = residue.at[0, 0, 0, 0].set(0.0)
    odd_residue = jnp.zeros_like(residue).at[0, 0, 0, 0].set(1.0)
    with pytest.raises(ValueError, match="refuses 1 live poles"):
        refuse_bad_pole_fields(omega.at[0, 0, 0, 0].set(0.0), zero_even,
                               odd_residue, where="cell")


def test_the_program_allocates_no_field_sized_scratch():
    omega, residue = _field(np.random.default_rng(3))
    field_bytes = int(residue.size) * residue.dtype.itemsize
    for odd_residue in (None, jnp.zeros_like(residue)):
        compiled = _pole_field_counts().lower(
            omega, residue, odd_residue).compile()
        stats = compiled.memory_analysis()
        if stats is None:
            pytest.skip("backend reports no memory analysis")
        # The eager form held several whole-field temporaries at once.
        assert int(stats.temp_size_in_bytes) < field_bytes // 4, (
            f"scratch {stats.temp_size_in_bytes} B against a "
            f"{field_bytes} B field")
