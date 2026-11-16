"""The screening resolvent prints one progress line per solved probe block."""
import numpy as np

from bse.w_ladder import ResolventProgress


def test_progress_lines_count_blocks_and_estimate_time_left():
    ticks = iter([0.0, 30.0, 40.0, 50.0, 60.0])      # first block carries a 30 s compile
    lines = []
    p = ResolventProgress(label="ladder W (w_bse)", nz=1, nq=2, n_logical=10, tol=1e-6,
                          max_iter=300, clock=lambda: next(ticks), emit=lines.append)
    it, res = np.array([12, 14, 13, 9, 9]), np.array([3e-7, 9e-7, 1e-7, 2e-7, 5e-7])
    p.update(0, 0, 0, 5, it, res, chunk=5)
    assert p.total == 4                                # 1 z x 2 q x ceil(10/5) blocks
    assert "z 1/1, q 1/2, block 1/2 (5 columns)" in lines[0]
    assert "mean 11.4 max 14/300" in lines[0] and "9.00e-07" in lines[0]
    assert "block 30.0 s; done 1/4 in 30 s, ~90 s left" in lines[0]
    p.update(0, 0, 5, 5, it, res, chunk=5)
    assert "block 2/2" in lines[1] and "~20 s left" in lines[1]   # 10 s per later block
    p.update(0, 1, 0, 5, it, res, chunk=5)
    last = p.update(0, 1, 5, 3, it, res, chunk=5)      # a short final block: 3 live columns
    assert "q 2/2, block 2/2 (3 columns)" in last and "done 4/4 in 60 s" in last
    assert "left" not in last
    assert "max 14/300" in last and "mean 13.0" in last
