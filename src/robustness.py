# --- Validation helpers for Appendix C ---------------------------------------------------------------
from numpy.lib.stride_tricks import sliding_window_view


def training_continuum(labels, flux, err):
    """Chapter 3's selection, using ONLY the supplied training stars (three labels, unnormalised spectra)."""
    cq = np.full_like(flux, np.nan)
    for lo, hi in CHIPS:
        c = np.flatnonzero((wave > lo) & (wave < hi))
        width = int(50 / (np.median(wave[c]) * np.log(10) * 6e-6))
        f = np.where(np.isfinite(err[:, c]), flux[:, c], np.nan)
        win = sliding_window_view(f, width, axis=1)[:, ::10]
        quantile = np.nanpercentile(win, 90, axis=2)
        centres = np.arange(win.shape[1]) * 10 + width // 2
        for n in range(len(f)):
            cq[n, c] = np.interp(np.arange(len(c)), centres, quantile[n])
    pivot = np.median(labels, axis=0)
    scale = np.diff(np.percentile(labels, [2.5, 97.5], axis=0), axis=0)[0] / 2
    fq = flux / cq
    eq = np.where(np.isfinite(fq), err / cq, np.inf)
    theta, _, _ = train((labels - pivot) / scale, np.nan_to_num(fq), eq)
    return (np.abs(theta[:, 0] - 1) < 0.015) & np.all(
        np.abs(theta[:, 1:4] / scale) < [1e-5, 0.0045, 0.0085], axis=1)


def robust_std(d):
    """One robust standard deviation per column (label); rows are stars."""
    return 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)


def paired_interval(a, b, n_boot=500, seed=42):
    """95% percentile interval for robust_std(a) - robust_std(b), resampling the SAME stars in both.
    Inputs are residuals (stars, labels). The last column is the median scatter over abundances (columns 3:).
    This measures held-out sampling uncertainty conditional on the two fitted models, not training uncertainty.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(len(a), size=len(a))
        sa, sb = robust_std(a[i]), robust_std(b[i])
        draws.append(np.r_[sa - sb, np.median(sa[3:]) - np.median(sb[3:])])
    return np.percentile(draws, [2.5, 97.5], axis=0)
