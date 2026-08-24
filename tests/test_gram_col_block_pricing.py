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

# Complete-live-set accounting: right pair build carries the completed left
# pair tile; final assembly has four local-G slots.  The max is explicit.
live = pc.gram_block_live_set_bytes(
    resident_bytes=1000,
    pair_left_peak_bytes=200,
    pair_right_peak_bytes=300,
    gram_fold_peak_bytes=400,
    one_pair_tile_bytes=50,
    gram_matrix_local_bytes=25,
)
assert live == {
    "pair_left": 1225,
    "pair_right": 1375,
    "gram_fold": 1425,
    "final_fold": 1100,
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
