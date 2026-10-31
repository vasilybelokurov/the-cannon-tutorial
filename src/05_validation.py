# %% [markdown]
# # Chapter 5 — Validation: how good are the labels?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/05_validation.ipynb)
#
# **Goal.** Measure how well The Cannon reproduces the reference labels for stars it was not trained on, decide
# whether the quadratic model is needed, compare the formal errors with the real ones, and find out what happens
# outside the training set. This follows the validation of Ness et al. 2015 (N15, §5.1–5.5 and Figs. 4, 11, 12).
#
# **You will learn**
# 1. $k$-fold cross-validation: every star is labelled by a model that never saw it;
# 2. to compare two models on the *same* stars (paired differences, bootstrap confidence intervals);
# 3. that formal errors are lower bounds;
# 4. that a data-driven model cannot extrapolate — and that χ² and an extrapolation flag reveal it only in part.
#
# Remember (Principle 4 of the tutorial): agreement with ASPCAP measures precision **on the ASPCAP scale**. If
# ASPCAP is biased, The Cannon inherits the bias.

# %%
#@@LOADER
#@@NORM
#@@CANNON
import time

labels = load('labels')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES, TEX, UNITS = ['teff', 'logg', 'fe_h'], [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]'], ['K', 'dex', 'dex']
data = {}
for nm in ['train', 'metalpoor', 'dwarfs']:
    lab = labels[labels['sample'] == nm].reset_index(drop=True)
    f, e = cheb_normalise(*prepare(load(f'spectra_{nm}')), cont)
    data[nm] = (lab, f, e)
tr, flux, err = data['train']
L = tr[NAMES].to_numpy()
print({k: len(v[0]) for k, v in data.items()})


def scaling(lab):
    """Pivot (median) and scale (half the 2.5-97.5 percentile range) of a set of training labels."""
    return np.median(lab, axis=0), (np.percentile(lab, 97.5, axis=0) - np.percentile(lab, 2.5, axis=0)) / 2


def label_all(f, e, theta, s, pivot, scale):
    """Test step for every spectrum: labels, formal errors (physical units), chi^2 per degree of freedom."""
    out = [test(f[n], e[n], theta, s) for n in range(len(f))]
    assert all(o[4] for o in out), 'the test-step optimiser did not converge for some spectra'
    x = np.array([o[0] for o in out])
    sig = np.array([np.sqrt(np.diag(o[1])) for o in out])
    K = x.shape[1]
    return x * scale + pivot, sig * scale, np.array([o[2] / (o[3] - K) for o in out])

# %% [markdown]
# ## 5.1 Ten-fold cross-validation
#
# Split the training set at random into 10 parts. For each part, train on the other nine (with their own pivot and
# scale) and label the part that was left out. After ten rounds every star has labels from a model that did not
# see it (N18 used the same "leave 10% out" scheme). Each star appears once in our training set, so no star can
# leak from the training part into the test part. In each round the model is trained with `train_clean` (Chapter 4).

# %%
rng = np.random.default_rng(5)
fold = rng.permutation(len(tr)) % 10
cv_lab, cv_sig, cv_chi = np.zeros_like(L), np.zeros_like(L), np.zeros(len(tr))
kept = {}                                                     # training stars kept by train_clean, per fold
t0 = time.time()
for k in range(10):
    t = fold != k
    pv, sc = scaling(L[t])
    th, _, s, kept[k] = train_clean((L[t] - pv) / sc, flux[t], err[t])
    cv_lab[~t], cv_sig[~t], cv_chi[~t] = label_all(flux[~t], err[~t], th, s, pv, sc)
print(f'10-fold cross-validation: {time.time() - t0:.0f} s')
dcv = cv_lab - L
stats = pd.DataFrame({NAMES[k]: scatter_stats(dcv[:, k]) for k in range(3)}).T
stats['median formal error'] = np.median(cv_sig, axis=0)
stats['median ASPCAP catalogue error'] = [tr[n + '_err'].median() for n in NAMES]
print('Cannon - ASPCAP, cross-validated (K, dex, dex); tail = fraction beyond 3 robust std:')
print(stats.round(4).to_string())

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for k in range(3):
    sc_ = ax[0, k].scatter(L[:, k], cv_lab[:, k], c=tr.fe_h if k < 2 else tr.teff, s=3, cmap='viridis')
    lim = [L[:, k].min(), L[:, k].max()]
    ax[0, k].plot(lim, lim, 'k-', lw=0.8)
    ax[0, k].set(xlabel='ASPCAP ' + TEX[k] + f' [{UNITS[k]}]', ylabel='Cannon (cross-validated) ' + TEX[k])
    plt.colorbar(sc_, ax=ax[0, k], label='[Fe/H]' if k < 2 else r'$T_{\rm eff}$')
    ax[1, k].hist(dcv[:, k], bins=60, histtype='step')
    ax[1, k].set(xlabel=f'Cannon − ASPCAP {TEX[k]} [{UNITS[k]}]', ylabel='stars',
                 title=f'bias {dcv[:, k].mean():+.3g}, rms {dcv[:, k].std():.3g}, formal {np.median(cv_sig[:, k]):.2g}')
plt.tight_layout()
# Test: at least as precise as N15's cross-validation (their Fig. 4: 96 K, 0.24 dex, 0.08 dex), with biases below
# a fifth of those values.
N15_RMS = np.array([96, 0.24, 0.08])
assert np.all(dcv.std(0) < N15_RMS) and np.all(np.abs(dcv.mean(0)) < N15_RMS / 5)

# %% [markdown]
# **Figure 5.1.** Ten-fold cross-validation of the three-label model on the training set. Top: cross-validated
# Cannon labels against ASPCAP's (colour: [Fe/H] for the first two panels, $T_{\rm eff}$ for the third); the line
# is the identity. Bottom: histograms of the differences, with their mean (bias), standard deviation ("rms") and the
# median formal error of the test step. The distributions have a narrow core and broad wings, so the robust standard
# deviation (1.4826 × median absolute deviation, printed above: 51 K, 0.115 dex, 0.034 dex) is 30–45% smaller than
# the standard deviation; a few
# per cent of the stars lie beyond three robust standard deviations (in the Cannon or in the ASPCAP label; we cannot
# tell which).
#
# **A small leak.** The continuum pixels of Chapter 3 were chosen with a model trained on *all* training stars, so
# the held-out stars of each fold helped to choose them. Re-selecting the continuum pixels inside each training fold
# (5 folds; the pixel lists overlap by 87–93%) changes the cross-validated scatter by less than 2%, in either
# direction (72.4 instead of 71.7 K, 0.188 instead of 0.191 dex, 0.062 instead of 0.063 dex): negligible here, but the
# right procedure (exercise 4).
#
# **Comparison with the literature.** These numbers measure agreement with ASPCAP, not accuracy: if The Cannon and
# ASPCAP share an error (they read the same spectra, and the model is trained on ASPCAP labels), the difference
# hides it, since ${\rm Var}(C - A) = {\rm Var}(C) + {\rm Var}(A) - 2\,{\rm Cov}(C, A)$. With that caveat:
#
# | | $T_{\rm eff}$ [K] | $\log g$ | [Fe/H] | training set, labels, statistic |
# |---|---|---|---|---|
# | this chapter (3 labels) | 72 (robust 51) | 0.19 (0.11) | 0.061 (0.034) | 1700 giants, S/N > 200, 3 labels; std (robust) |
# | Ness et al. 2015, Fig. 4 | 96 | 0.24 | 0.08 | 542 cluster stars, DR10, 3 labels; rms |
# | Ness et al. 2018, §3.0.1 | ≲ 45 | ≲ 0.10 | ≲ 0.02 | 5000 giants, S/N > 200, 23 labels incl. LSF width |
# | Casey et al. 2016, Fig. 6 | 38 | 0.07 | 0.03 | 12 681 giants, 17 labels, L1-regularised |
# | Leung & Bovy 2019, Table 1 (astroNN) | 30 | 0.051 | 0.020 | neural network trained on APOGEE DR14; robust std |
#
# We are ≈ 25% better than the original Cannon, but 2–3 times worse than the later APOGEE models in standard
# deviation, and ≈ 1.7–2.3 times worse than astroNN when both are measured robustly. The likely
# reasons, none of which we have isolated: they used 3–8 times more training stars (or a neural network) and many more labels (Chapter 8
# shows that adding labels alone takes us to 53 K, 0.15 dex, 0.052 dex), Ness et al. 2018 modelled the line-spread
# function, Leung & Bovy quote a robust statistic, and our label-balanced training set deliberately over-represents
# the hard edges of label space (cool, luminous and metal-poor giants), which an even cross-validation then weighs
# heavily.
#
# **Formal errors are lower bounds.** The formal error of the test step is 15–20 times smaller than the scatter. It
# propagates the noise model $\sigma_{n\lambda}^2 + s_\lambda^2$ through a model assumed to be correct, treating the
# pixels as independent; real model errors are correlated over whole lines and bands and do not average down (and
# the noise of the reference labels is not in it at all). N15 found the same: a "precision" of 6 K against an rms of
# 96 K. (The ASPCAP catalogue errors printed above are internal uncertainties of the same kind, also far smaller
# than any external comparison.)

# %%
ratio = np.median(cv_sig, axis=0) / dcv.std(0)
print('median formal error / cross-validation rms:', ratio.round(3))
assert np.all(ratio < 0.5)

# %% [markdown]
# ## 5.2 Where in label space?
#
# The training set was built to cover label space evenly, but some regions are still sparse, notably
# [Fe/H] < −1. How does the precision vary with metallicity, and with the number of training stars nearby?

# %%
fb = np.arange(-1.5, 0.41, 0.25)
c = np.digitize(L[:, 2], fb) - 1
rms_bin = np.array([[dcv[c == b, k].std() for b in range(len(fb) - 1)] for k in range(3)])
n_bin = np.array([(c == b).sum() for b in range(len(fb) - 1)])
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    ax[k].plot(0.5 * (fb[1:] + fb[:-1]), rms_bin[k], 'o-')
    ax[k].set(xlabel='ASPCAP [Fe/H]', ylabel=f'cross-validation rms of {TEX[k]} [{UNITS[k]}]')
    a2 = ax[k].twinx()
    a2.bar(0.5 * (fb[1:] + fb[:-1]), n_bin, width=0.2, alpha=0.15, color='k')
    a2.set(ylabel='training stars per bin')
plt.tight_layout()
print('rms of [Fe/H] per bin:', dict(zip(np.round(0.5 * (fb[1:] + fb[:-1]), 3), rms_bin[2].round(3))))

# %% [markdown]
# **Figure 5.2.** Cross-validation rms of each label in bins of ASPCAP [Fe/H] (0.25 dex; dots) and the number of
# training stars per bin (grey bars, right axis). The $T_{\rm eff}$ and $\log g$ errors grow towards low
# metallicity (from ≈ 67 to 89 K, and from ≈ 0.16 to 0.24 dex), where lines are weaker and training stars fewer —
# two effects that cannot be separated with these data alone. The [Fe/H] error does not: it stays at
# 0.05–0.07 dex, and is largest at the metal-rich end, where many lines are saturated and blended.
#
# ## 5.3 Linear or quadratic?
#
# N15 (§5.1) found a linear model "too inflexible". Test it on the same folds: the linear label vector is
# $[1, x_1, x_2, x_3]$ (4 coefficients), and its test step is a *linear* least-squares problem. For a fair
# comparison we look at the difference of the absolute errors, star by star, and bootstrap it.

# %%
def train_lin(x, f, e):
    """Linear model: theta (P, 4) at the scatter of the quadratic model would be unfair; fit its own scatter."""
    V = np.hstack([np.ones((len(x), 1)), x])
    s_ = np.zeros(f.shape[1])
    for _ in range(4):
        th_, _ = fit_theta(V, f, e**2 + s_**2)
        s_ = fit_scatter(f - V @ th_.T, e**2)
    return fit_theta(V, f, e**2 + s_**2)[0], s_


def test_lin(f, e, th_, s_):
    """Linear test step: weighted linear least squares for x, since f = th_0 + th_k x_k."""
    ok = np.isfinite(e) & (th_[:, 0] != 0)
    w = 1 / np.sqrt(e[ok]**2 + s_[ok]**2)
    return np.linalg.lstsq(th_[ok, 1:] * w[:, None], (f[ok] - th_[ok, 0]) * w, rcond=None)[0]


lin_lab = np.zeros_like(L)
for k in range(10):
    t = np.flatnonzero(fold != k)[kept[k]]                   # the same cleaned training stars as the quadratic model
    pv, sc = scaling(L[fold != k])
    th_l, s_l = train_lin((L[t] - pv) / sc, flux[t], err[t])
    lin_lab[fold == k] = np.array([test_lin(flux[n], err[n], th_l, s_l) for n in np.flatnonzero(fold == k)]) * sc + pv
dlin = lin_lab - L
gain = np.abs(dlin) - np.abs(dcv)                                  # > 0: the quadratic model is closer to ASPCAP
boot = np.array([gain[rng.integers(0, len(gain), len(gain))].mean(0) for _ in range(1000)])
lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)
for k in range(3):
    print(f'{NAMES[k]:5s}: rms linear {dlin[:, k].std():.3f}, quadratic {dcv[:, k].std():.3f}; '
          f'mean |err_lin| - |err_quad| = {gain[:, k].mean():.4f} (95% CI {lo95[k]:.4f} to {hi95[k]:.4f})')
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    ax[k].plot(L[:, k], dlin[:, k], '.', ms=2, color='C1', alpha=0.5, label='linear')
    ax[k].plot(L[:, k], dcv[:, k], '.', ms=2, color='C0', alpha=0.5, label='quadratic')
    ax[k].axhline(0, color='k', lw=0.8)
    ax[k].set(xlabel='ASPCAP ' + TEX[k], ylabel=f'Cannon − ASPCAP [{UNITS[k]}]')
    ax[k].legend(fontsize=8, markerscale=4)
plt.tight_layout()
# Test: the quadratic model is closer to ASPCAP for T_eff and log g, with 95% confidence.
assert lo95[0] > 0 and lo95[1] > 0

# %% [markdown]
# **Figure 5.3.** Cross-validated label errors (Cannon − ASPCAP) of the linear (orange) and quadratic (blue)
# models, against the ASPCAP label. The linear model has systematic, curved trends at the extremes of each label —
# it cannot follow lines that saturate or molecules that appear only in the coolest stars — which is why N15
# adopted the quadratic model.
#
# ## 5.4 Outside the training set
#
# Now train on the whole training set and label two samples it does not cover: giants with
# $-2.5 <$ [Fe/H] $< -1.5$ (below the training range) and main-sequence dwarfs ($\log g > 4$; no dwarfs in training).
# Two warnings are available without knowing the true labels:
#
# - the **χ²** of the fit: we flag a star if its χ²/ν exceeds the 95th percentile of the cross-validated giants;
# - N15's **extrapolation flag** EFLAG (their Table 1): set if any inferred label lies outside the range of the
#   training labels.

# %%
pv, sc = scaling(L)
th, _, s, _ = train_clean((L - pv) / sc, flux, err)
lo_tr, hi_tr = L.min(axis=0), L.max(axis=0)
chi_cut = np.percentile(cv_chi, 95)
res = {'giants (cross-validated)': (L, cv_lab, cv_chi)}
for nm in ['metalpoor', 'dwarfs']:
    lab, f, e = data[nm]
    lab_c, _, chi = label_all(f, e, th, s, pv, sc)
    res[nm] = (lab[NAMES].to_numpy(), lab_c, chi)
flags = {}
for nm, (la, lc, chi) in res.items():
    eflag = np.any((lc < lo_tr) | (lc > hi_tr), axis=1)
    flags[nm] = {'χ² flag': np.mean(chi > chi_cut), 'EFLAG': np.mean(eflag), 'either': np.mean((chi > chi_cut) | eflag)}
    print(f'{nm:25s}', {k: round(v, 2) for k, v in flags[nm].items()})

fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for j, (nm, (la, lc, chi)) in enumerate(res.items()):
    ax[0].scatter(la[:, 2], lc[:, 2], s=3, label=nm, color=f'C{j}')
    ax[1].scatter(la[:, 1], lc[:, 1], s=3, color=f'C{j}')
    ax[2].hist(chi, bins=np.logspace(-0.6, 1.2, 50), histtype='step', density=True, label=nm, color=f'C{j}')
    ax[3].bar(np.arange(3) + 0.27 * (j - 1), list(flags[nm].values()), width=0.27, color=f'C{j}', label=nm)
for a, k in [(ax[0], 2), (ax[1], 1)]:
    lim = [min(r[0][:, k].min() for r in res.values()), max(r[0][:, k].max() for r in res.values())]
    a.plot(lim, lim, 'k-', lw=0.8)
    a.axvspan(lo_tr[k], hi_tr[k], color='0.9', zorder=-1)
    a.axhline(lo_tr[k], color='0.5', lw=0.5, ls='--')
    a.axhline(hi_tr[k], color='0.5', lw=0.5, ls='--')
    a.set(xlabel='ASPCAP ' + TEX[k], ylabel='Cannon ' + TEX[k])
ax[0].legend(fontsize=7, markerscale=4)
ax[2].axvline(chi_cut, color='k', ls=':', lw=0.8)
ax[2].set(xscale='log', xlabel=r'$\chi^2/\nu$', ylabel='density')
ax[3].set(xticks=range(3), xticklabels=list(flags['dwarfs']), ylabel='fraction of stars flagged', ylim=(0, 1))
plt.tight_layout()
mp = res['metalpoor']
for lo_, hi_ in [(-2.5, -2.0), (-2.0, -1.5)]:
    b = (mp[0][:, 2] >= lo_) & (mp[0][:, 2] < hi_)
    dd = mp[1][b, 2] - mp[0][b, 2]
    print(f'metal-poor giants, ASPCAP [Fe/H] in [{lo_}, {hi_}): {b.sum()} stars, bias {dd.mean():+.3f} +- {dd.std() / np.sqrt(b.sum()):.3f} dex')
dw = res['dwarfs']
print(f"dwarfs: median ASPCAP log g {np.median(dw[0][:, 1]):.2f}, median Cannon log g {np.median(dw[1][:, 1]):.2f}")
b = mp[0][:, 2] < -2.0
bias_mp = mp[1][b, 2] - mp[0][b, 2]
# Tests: (1) the labels outside the training range are biased: metal-poor giants below [Fe/H] = -2 by more than
# 3 standard errors, dwarfs by more than 0.3 dex in log g (median); (2) EFLAG catches at least 5 times more dwarfs
# than it falsely flags among the cross-validated giants; (3) neither flag catches all extrapolated stars:
# fewer than 75% of the dwarfs and 50% of the metal-poor giants are flagged by either.
assert abs(bias_mp.mean()) > 3 * bias_mp.std() / np.sqrt(b.sum())
assert np.median(np.abs(dw[1][:, 1] - dw[0][:, 1])) > 0.3
assert flags['dwarfs']['EFLAG'] > 5 * flags['giants (cross-validated)']['EFLAG']
assert flags['dwarfs']['either'] < 0.75 and flags['metalpoor']['either'] < 0.5

# %% [markdown]
# **Figure 5.4.** The model trained on all training giants applied outside its training set. First two panels:
# Cannon against ASPCAP [Fe/H] and $\log g$ for the cross-validated training giants (blue), metal-poor giants
# (orange) and dwarfs (green); the grey band and dashed lines mark the range of the training labels, the solid line
# the identity. Third: distributions of χ²/ν, with the flagging threshold (95th percentile of the giants, dotted).
# Fourth: fraction of each sample flagged by χ², by N15's EFLAG, and by either. Below the training range the
# inferred [Fe/H] flattens at the edge of the training set (−1.5), and their $\log g$ comes out too high: the
# polynomial, fitted where there are data, does not continue the trend correctly. (N15 also found a flattening at
# [Fe/H] < −2.2, but inside their training range, where it reflects cluster labels; ours is pure extrapolation, and
# larger: +0.26 dex at −2 < [Fe/H] < −1.5 and +0.6 dex below −2, against their +0.05 to +0.3 dex.) The dwarfs are placed at $\log g \approx 3.8$, just above the training
# range, instead of 4.5, and their [Fe/H] is too low. The warnings help but do
# not suffice: EFLAG catches about 60% of the dwarfs and a third of the metal-poor giants; χ² is only mildly higher
# for them, because the model finds a combination of giant labels that fits a dwarf spectrum almost as well as a
# giant's. A fit can be acceptable and still wrong. The only safe rule: train on the kind of stars you want to
# label.
#
# ## Summary
#
# - Ten-fold cross-validation gives the honest precision of the three-label model on the ASPCAP scale.
# - The quadratic model is significantly better than the linear one for $T_{\rm eff}$ and $\log g$.
# - Formal errors are an order of magnitude smaller than the real errors; do not use them as uncertainties
#   without calibration.
# - Extrapolation fails, and is only partly detected by χ² or by an extrapolation flag.
#
# **Exercises.**
# 1. Repeat the cross-validation with only 300 training stars (random). How does the rms change? (N15 used 542.)
# 2. Train with the dwarfs included. Do the giants' labels get worse?
# 3. Implement N15's bootstrap (their Fig. 11): train 20 models on bootstrap samples of the training set and
#    compare the spread of the labels of the metal-poor giants with their formal errors.
# 4. Move the selection of continuum pixels (Chapter 3, §3.3) inside the cross-validation loop and confirm that the
#    scatter changes by less than 2%.

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Jönsson et al. 2020, *APOGEE Data and Spectral Analysis from SDSS Data Release 16: Seven Years of Observations Including First Results from APOGEE-South*, AJ 160, 120. [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J)
# - Leung & Bovy 2019, *Deep learning of multi-element abundances from high-resolution spectroscopic data*, MNRAS 483, 3255. [ADS](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
