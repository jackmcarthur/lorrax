"""Pure capacity gates for pivoted-Cholesky Gram column blocking."""

_GRAM_PRICING_WORKER = r'''
import centroid.pivoted_cholesky as pc

print("GRAM_PRICING_MODULE=" + pc.__file__)

budget = 8 * 2**30
expected = {8: 2896, 81: 908}
widths = {}
for nk, expected_width in expected.items():
    width = pc.auto_gram_col_block_width(
        nk, nspinor=2, budget_bytes=budget, divisor=4,
    )
    widths[nk] = width
    assert width == expected_width, (nk, width, expected_width)
    assert width % 4 == 0
    assert pc.gram_col_block_bytes(nk, 2, width) <= budget
    assert pc.gram_col_block_bytes(nk, 2, width + 4) > budget

assert widths[81] < widths[8], widths

# The implementation now tiles BOTH candidate axes, so the registered square
# law and the physical pair tensors describe the same object.  On 4x4 the
# exact local two-pair footprint is 1/16 of the global square model.
tile = widths[81]
assert pc.gram_col_block_device_bytes(
    81, 2, 3600, tile, x_shards=4, y_shards=4,
) == pc.gram_col_block_bytes(81, 2, tile) // 16

# Complete-live-set accounting: the fused compiled peak includes its four
# input slices and internal pair workspaces; final Hermitian fold has three
# local-G slots. The max is explicit.
live = pc.gram_block_live_set_bytes(
    resident_bytes=1000,
    fused_gram_peak_bytes=400,
    gram_matrix_local_bytes=25,
    extracted_input_bytes=30,
    extract_increment_bytes=150,
)
assert live == {
    "extract": 1205,
    "fused_gram": 1425,
    "final_fold": 1075,
    "peak": 1425,
}

# Geometric selection returns a rung it actually queried and never rounds a
# width off the common x/y divisor.
seen = []
def peak_for_width(width):
    seen.append(width)
    return {"peak": width * 100}

chosen, chosen_facts = pc._auto_gram_width_from_compiled_peaks(
    256, max_width=1020, divisor=4, budget_bytes=70000,
    peak_for_width=peak_for_width,
)
assert chosen == 512, (chosen, seen)
assert chosen_facts["peak"] == 51200
assert seen == [256, 512, 1020], seen

# A full-width tile is admissible when its compiled live set fits.
chosen, _ = pc._auto_gram_width_from_compiled_peaks(
    488, max_width=3008, divisor=4, budget_bytes=4000,
    peak_for_width=lambda width: {"peak": width},
)
assert chosen == 3008
assert pc.gram_tile_schedule(3008, chosen) == (1, 3008, 1.0)

# If the 3008-wide rung fails but 1952 fits, compact to the minimum width
# that preserves two tiles. This removes the old ~1.68x padded square work.
seen_compact = []
def compact_peak(width):
    seen_compact.append(width)
    return {"peak": width}

chosen, _ = pc._auto_gram_width_from_compiled_peaks(
    488, max_width=3008, divisor=4, budget_bytes=2000,
    peak_for_width=compact_peak,
)
assert chosen == 1504, (chosen, seen_compact)
assert pc.gram_tile_schedule(3008, chosen) == (2, 3008, 1.0)
assert seen_compact == [488, 976, 1952, 3008, 1504], seen_compact

chosen, _ = pc._auto_gram_width_from_compiled_peaks(
    488, max_width=3008, divisor=4, budget_bytes=500,
    peak_for_width=lambda width: {"peak": width},
)
ntiles, executed, inflation = pc.gram_tile_schedule(3008, chosen)
assert (chosen, ntiles, executed) == (432, 7, 3024)
assert inflation < 1.011

required = pc.gram_col_block_bytes(81, 2, 256)
try:
    pc.auto_gram_col_block_width(81, 2, required - 1, divisor=4)
except MemoryError as exc:
    assert "refuses before pair-density" in str(exc), exc
else:
    raise AssertionError("undersized Gram budget did not refuse")

print(f"GRAM_COL_BLOCK_PRICING_OK nk8={widths[8]} nk81={widths[81]} "
      f"compiled_rung={chosen}")
'''


def test_auto_gram_col_block_pricing_at_nk8_and_nk81():
    """The registered square-law model fits its budget and shrinks at nk81."""
    import os
    import subprocess
    import sys

    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _GRAM_PRICING_WORKER],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, (
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}")
    print(out.stdout, end="")
    assert "GRAM_COL_BLOCK_PRICING_OK" in out.stdout, out.stdout
