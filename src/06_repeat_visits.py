# %% [markdown]
# # Chapter 6 — Low S/N: repeat visits of the same stars
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/06_repeat_visits.ipynb)
#
# **Goal.** Measure how the precision of The Cannon degrades with S/N using real low-S/N spectra: the single
# visits that APOGEE combines into each star's spectrum. This is the test of Ness et al. 2015 (N15, §5.6), who
# chose single visits over spectra with artificially added noise because adding noise "would bypass the question
# of how consistently the continuum can be defined at different S/N".
#
# **You will learn**
# 1. to measure precision from **repeat observations**, independently of any reference labels, with pairs of
#    visits of similar S/N;
# 2. why a single visit compared with the combined spectrum that contains it underestimates the scatter, and why
#    visits of very different S/N should not be pooled;
# 3. how the normalisation affects labels at low S/N (quantile vs continuum pixels, Chapter 3).

# %%
#@@LOADER
#@@NORM
#@@CANNON
import warnings
from numpy.lib.stride_tricks import sliding_window_view


def quantile_continuum(flux, err, q=90, width=50.0, step=10):
    """Running q-th percentile pseudo-continuum, per detector (Chapter 3)."""
    cont = np.full(flux.shape, np.nan)
    for lo, hi in CHIPS:
        c = np.flatnonzero((wave > lo) & (wave < hi))
        npw = int(width / (np.median(wave[c]) * np.log(10) * 6e-6))
        f = np.where(np.isfinite(err[:, c]), flux[:, c], np.nan)
        win = sliding_window_view(f, npw, axis=1)[:, ::step]
        with warnings.catch_warnings():                       # windows with no good pixel give NaN, silently
            warnings.simplefilter('ignore', RuntimeWarning)
            qc = np.nanpercentile(win, q, axis=2)
        centre = np.arange(win.shape[1]) * step + npw // 2
        for n in range(len(f)):
            cont[n, c] = np.interp(np.arange(len(c)), centre, qc[n])
    return cont


def quantile_normalise(flux, err):
    cq = quantile_continuum(flux, err)
    return np.nan_to_num(flux / cq), np.where(np.isfinite(cq), err / cq, np.inf)


labels = load('labels')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES, TEX, UNITS = ['teff', 'logg', 'fe_h'], [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]'], ['K', 'dex', 'dex']
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
L = tr[NAMES].to_numpy()
raw_tr = prepare(load('spectra_train'))
vis = load('spectra_visits')
vl = labels[labels['sample'] == 'visits'].reset_index(drop=True)
raw_v = prepare(vis)
star, visit, snr = vis['star'], vis['visit'], vis['snr']        # star index, 0 = combined spectrum, per-spectrum S/N
print(f"{len(vl)} stars, {(visit > 0).sum()} single visits; visits per star: {np.bincount(star).min() - 1}-{np.bincount(star).max() - 1}")

# %% [markdown]
# ## 6.1 Two models, two normalisations
#
# We train two three-label models on the whole training set: one on spectra normalised with the continuum pixels
# (our standard), one on spectra normalised with the running 90th percentile. Each model is then applied to the
# visit spectra normalised **the same way** as its training spectra.

# %%
pv = np.median(L, axis=0)
sc = (np.percentile(L, 97.5, axis=0) - np.percentile(L, 2.5, axis=0)) / 2
models, lab_v, sig_v = {}, {}, {}
for nm, norm in [('continuum pixels', lambda f, e: cheb_normalise(f, e, cont)), ('running quantile', quantile_normalise)]:
    th, _, s, _ = train_clean((L - pv) / sc, *norm(*raw_tr))
    fv, ev = norm(*raw_v)
    out = [test(fv[n], ev[n], th, s) for n in range(len(fv))]
    lab_v[nm] = np.array([o[0] for o in out]) * sc + pv
    sig_v[nm] = np.array([np.sqrt(np.diag(o[1])) for o in out]) * sc     # formal errors, physical units
    print(f'{nm}: labelled {len(fv)} spectra')

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(snr[visit > 0], bins=np.arange(0, 200, 5), label='single visits')
ax.hist(snr[visit == 0], bins=np.arange(0, 200, 5), histtype='step', color='k', label='combined spectra')
ax.set(xlabel='S/N per pixel', ylabel='spectra')
ax.legend()
plt.tight_layout()

# %% [markdown]
# **Figure 6.1.** S/N of the single-visit spectra (blue) and of the combined spectra of the same 200 stars (black).
# A star with $n$ visits of similar quality has a combined S/N ≈ $\sqrt n$ times that of one visit; the visits span
# S/N ≈ 5–130, most of them 25–90. (Bins with fewer than 20 visits are not used below.)
#
# ## 6.2 Precision from repeat visits
#
# Two visits $i$ and $j$ of the same star have the same true labels, so if their errors are independent with
# variances $v_i$ and $v_j$, the difference of their labels has variance $v_i + v_j$. For two visits of *similar*
# S/N, $v_i \approx v_j = v$ and $\tfrac12(\ell_i - \ell_j)^2$ estimates $v$, the single-visit variance at that S/N —
# without reference labels, and without comparing a visit with a combination that contains it (they would share its
# noise). We use every pair of visits of a star whose S/N differ by less than 25%, bin the pairs by their mean S/N,
# and get uncertainties by resampling the *stars* (a bootstrap: pairs of one star are not independent).

# %%
sel = visit > 0
bins = np.array([20, 30, 40, 50, 60, 80, 100, 150])
mid = np.sqrt(bins[1:] * bins[:-1])                                   # geometric bin centres
pairs = []                                                            # (star, i, j) for matched-S/N pairs of visits
for st in np.unique(star):
    v = np.flatnonzero(sel & (star == st))
    for a_ in range(len(v)):
        for b_ in range(a_ + 1, len(v)):
            if abs(np.log(snr[v[a_]] / snr[v[b_]])) < np.log(1.25):
                pairs.append((st, v[a_], v[b_]))
pairs = np.array(pairs)
psnr = np.sqrt(snr[pairs[:, 1]] * snr[pairs[:, 2]])                   # mean S/N of the pair
pb = np.digitize(psnr, bins)
print(f'{len(pairs)} matched pairs of visits from {len(np.unique(pairs[:, 0]))} stars')


def pair_precision(lv, idx):
    """sqrt(mean of (l_i - l_j)^2 / 2) per S/N bin, for the pairs with indices idx; NaN for bins with < 15 pairs."""
    d2 = (lv[pairs[idx, 1]] - lv[pairs[idx, 2]])**2 / 2
    return np.array([[np.sqrt(d2[pb[idx] == i, k].mean()) if (pb[idx] == i).sum() >= 15 else np.nan
                      for i in range(1, len(bins))] for k in range(3)])


rng = np.random.default_rng(6)
stars_p = np.unique(pairs[:, 0])
prec, prec_lo, prec_hi = {}, {}, {}
for nm, lv in lab_v.items():
    prec[nm] = pair_precision(lv, np.arange(len(pairs)))
    boot = []
    for _ in range(200):                                              # bootstrap over stars
        pick = rng.choice(stars_p, len(stars_p))
        idx = np.concatenate([np.flatnonzero(pairs[:, 0] == st_) for st_ in pick])
        boot.append(pair_precision(lv, idx))
    prec_lo[nm], prec_hi[nm] = np.nanpercentile(boot, [16, 84], axis=0)
sig = sig_v['continuum pixels']
formal = np.array([[np.median(np.sqrt((sig[pairs[pb == i, 1], k]**2 + sig[pairs[pb == i, 2], k]**2) / 2))
                    if (pb == i).sum() >= 15 else np.nan for i in range(1, len(bins))] for k in range(3)])
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    for j, (nm, mk) in enumerate(zip(prec, 'os')):
        p_ = prec[nm][k]
        ax[k].errorbar(mid * (1 + 0.03 * j), p_, [p_ - prec_lo[nm][k], prec_hi[nm][k] - p_], fmt=mk + '-', capsize=2,
                       label=f'repeat visits ({nm})')
    ax[k].plot(mid, formal[k], 'k^:', label='median formal error')
    ax[k].set(xscale='log', yscale='log', xlabel='S/N of the visits', ylabel=f'single-visit precision, {TEX[k]} [{UNITS[k]}]')
    ax[k].legend(fontsize=7)
plt.tight_layout()
pc = prec['continuum pixels']
for k in range(3):
    print(f'{NAMES[k]:5s} repeat precision: {pc[k].round(3)} | 16-84%: {prec_lo["continuum pixels"][k].round(3)} - '
          f'{prec_hi["continuum pixels"][k].round(3)} | formal: {formal[k].round(3)}')
ok = np.isfinite(pc[0])
top = np.flatnonzero(ok)[-1]                                          # the highest-S/N bin with enough pairs
# Tests: in every populated S/N bin the repeat precision is larger than the formal error, and in the highest-S/N
# bin it exceeds the formal error by more than the bootstrap uncertainty (its 16th percentile is above the formal
# error); and the precision improves with S/N (highest bin better than lowest, beyond the bootstrap ranges).
assert np.all(pc[:, ok] > formal[:, ok])
assert all(prec_lo['continuum pixels'][k][top] > formal[k][top] for k in range(3))
assert all(prec_hi['continuum pixels'][k][top] < prec_lo['continuum pixels'][k][0] for k in range(3))

# %% [markdown]
# **Figure 6.2.** Single-visit precision of each label from pairs of visits of the same star with similar S/N, in
# bins of their mean S/N, for the two normalisations (circles, squares; error bars: 16–84% range of a bootstrap over
# stars), and the median formal error of the test step (triangles). Both fall with S/N, more slowly than
# $1/({\rm S/N})$ (log-log slopes ≈ −0.5 to −0.7), because the noise model includes the scatter $s_\lambda$, which
# does not shrink with S/N. From S/N ≈ 25 to 120 the precision improves from ≈ 45 K, 0.07 dex and 0.04 dex to
# ≈ 20 K, 0.03 dex and 0.014 dex, and it is 2–4 times the formal error at every S/N. The formal error treats the
# deviations of the pixels from the model as independent; real deviations are correlated over whole lines and bands,
# so they do not average down. Changes from visit to visit in the line-spread function (fibre), sky subtraction and
# telluric correction may add to this; we have not tested them. (An earlier version of this chapter subtracted each
# star's mean over all its visits and binned by the S/N of each visit; with visits of unequal S/N that estimator is
# biased — noisy companions inflate the residuals of good visits — and it produced a spurious "noise floor".)
#
# **Comparison with the literature.** N15 (§5.6) compared single visits with ASPCAP and found ≈ 100 K, 0.2 dex and
# 0.1 dex at S/N 30–50 (including ASPCAP's own errors). Leung & Bovy (2019, Table 2) compared single visits with
# the combined spectrum (an optimistic estimator, since they share noise) for stars with combined S/N 100–200 and
# found robust scatters of 29 K, 0.047 dex and 0.020 dex for their neural network. The ASPCAP pipeline itself
# reproduces labels from independent high-S/N spectra of the same star to ≈ 14 K, 0.04 dex and 0.02 dex in the
# difference of a pair (Jönsson et al. 2020, Table 10, DR16). Our single-visit numbers are of the same order.
#
# ## 6.3 Are low-S/N labels biased?
#
# Precision is not accuracy. If the normalisation changes with S/N, the labels of noisy spectra are *shifted*,
# not just noisier. Compare the mean visit label with the label of the star's high-S/N combined spectrum, in bins of
# visit S/N. (Here the shared noise does not matter: we look at the mean, not at the scatter.)

# %%
bias = {}
for nm, lv in lab_v.items():
    comb = lv[visit == 0][star[sel]]                                   # combined-spectrum label of each visit's star
    d = lv[sel] - comb
    b = np.digitize(snr[sel], bins)
    bias[nm] = np.array([[d[b == i, k].mean() if (b == i).sum() > 20 else np.nan for i in range(1, len(bins))] for k in range(3)])
    bias[nm + ' err'] = np.array([[d[b == i, k].std() / np.sqrt((b == i).sum()) if (b == i).sum() > 20 else np.nan
                                   for i in range(1, len(bins))] for k in range(3)])
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    for j, nm in enumerate(['continuum pixels', 'running quantile']):
        ax[k].errorbar(mid * (1 + 0.03 * j), bias[nm][k], bias[nm + ' err'][k], fmt='os'[j] + '-', label=nm, capsize=2)
    ax[k].axhline(0, color='k', lw=0.8)
    ax[k].set(xscale='log', xlabel='S/N of the visit', ylabel=f'mean (visit − combined) {TEX[k]} [{UNITS[k]}]')
    ax[k].legend(fontsize=8)
plt.tight_layout()
low = mid < 60
for k in range(3):
    print(f"{NAMES[k]:5s} mean bias at visit S/N < 60: continuum pixels {np.nanmean(bias['continuum pixels'][k][low]):+.3f}, "
          f"quantile {np.nanmean(bias['running quantile'][k][low]):+.3f} {UNITS[k]}")
# Test: at visit S/N < 60 the quantile normalisation biases [Fe/H] more than the continuum-pixel normalisation.
assert abs(np.nanmean(bias['running quantile'][2][low])) > abs(np.nanmean(bias['continuum pixels'][2][low]))

# %% [markdown]
# **Figure 6.3.** Mean difference between the labels of single visits and of the combined spectrum of the same
# star, in bins of visit S/N, with standard errors, for the two normalisations. With the running quantile the
# continuum of a noisy spectrum is placed higher (Chapter 3), lines look deeper, and the labels drift with S/N: at
# S/N ≈ 25 the visits come out ≈ 180 K cooler, ≈ 0.1 dex lower in $\log g$ and ≈ 0.09 dex more metal-rich than
# the combined spectrum. With the continuum pixels the drift is consistent with zero (≲ 10 K, 0.015 dex,
# 0.006 dex). This is the reason for the care taken
# in Chapter 3.
#
# ## Summary
#
# - Repeat visits measure precision without reference labels; use the scatter among independent visits, not the
#   difference between a visit and a combination that contains it.
# - Pairs of visits of similar S/N measure the single-visit precision without reference labels: ≈ 20 K, 0.03 dex
#   and 0.014 dex at S/N ≈ 120, 2–4 times the formal errors at every S/N, because model errors are correlated across
#   pixels.
# - An S/N-dependent normalisation produces S/N-dependent labels; a fixed set of continuum pixels largely avoids it.
#
# **Exercises.**
# 1. Plot the visit-to-visit precision against the formal errors of the test step. Are formal errors closer to the
#    truth at low S/N than at high S/N (Chapter 5)? Why?
# 2. Combine two visits of each star (inverse-variance weighted) and check that the precision improves by √2.

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Jönsson et al. 2020, *APOGEE Data and Spectral Analysis from SDSS Data Release 16: Seven Years of Observations Including First Results from APOGEE-South*, AJ 160, 120. [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J)
# - Leung & Bovy 2019, *Deep learning of multi-element abundances from high-resolution spectroscopic data*, MNRAS 483, 3255. [ADS](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
