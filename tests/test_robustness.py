"""Statistical checks for the paired comparisons in Appendix C."""
from pathlib import Path
import numpy as np

ns = {'np': np}
exec((Path(__file__).resolve().parents[1] / 'src/robustness.py').read_text(), ns)


def test_pairing_cancels_identical_residuals_exactly():
    d = np.random.default_rng(1).normal(size=(100, 12))
    assert np.array_equal(ns['paired_interval'](d, d), np.zeros((2, 13)))


def test_interval_detects_known_scale_change_and_ignores_bias():
    d = np.random.default_rng(2).normal(size=(2000, 12))
    lo, hi = ns['paired_interval'](2 * d + 10, d, n_boot=200)
    assert np.all(lo > 0.8) and np.all(hi < 1.2)
    assert np.allclose(ns['robust_std'](d + 10), ns['robust_std'](d))
