# %% [markdown]
# # Chapter 10 — Many labels: flexibility, regularisation, censoring
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/10_many_labels.ipynb)
#
# **Goal.** Build a model with twelve labels — $T_{\rm eff}$, $\log g$, [Fe/H] and nine abundances [X/Fe] — and add
# to The Cannon of Chapter 2 the tools that published codes use for many labels (Casey et al. 2016, C16): models of
# higher polynomial order, L1 regularisation, censoring, several starting points for the test step, and a test of
# whether a star lies inside the training set. For each tool, measure on APOGEE spectra what it buys.
#
# **You will learn**
# 1. that more labels and a more flexible model both make the stellar parameters more precise, and to tell the two
#    apart;
# 2. what L1 regularisation does (sparse coefficients, better predicted spectra) and when it improves the labels:
#    only when the training set is not much larger than the number of coefficients;
# 3. that censoring — letting each abundance act only near its own lines — changes the precision element by
#    element (Holtzman et al. 2018);
# 4. why a convex-hull test of "inside the training set" flags almost every star in many dimensions.
#
# **The new code** is in the block `more.py` below (tested in `tests/test_cannon.py`):
# - `vectorize_n(x, order)` and `dvectorize_n`: label vectors of any polynomial order (for order 2 identical to
#   `vectorize`);
# - `fit_theta_l1`: at every pixel, minimise $\chi^2 + \Lambda \sum_{d \ge 1} |\theta_d|$ (C16, eq. 2 and 5), with
#   some coefficients forced to zero by a censoring mask. For $\Lambda = 0$ it is the weighted least squares of
#   Chapter 2; for $\Lambda > 0$ it uses ADMM (Boyd et al. 2011), which handles the strongly correlated quadratic
#   terms much better than coordinate descent;
# - `train_more`: the training step of Chapter 2 (coefficients and maximum-likelihood scatter, iterated) with these
#   options; `test_more`: the test step from several starting points; `censor_mask`; `in_hull`.

# %%
#@@LOADER
#@@NORM
#@@CANNON
#@@MORE
import time

labels = load('labels')
lines = load('linelist')
cont = load('continuum_pixels')['continuum'].to_numpy()
EL = ['c', 'n', 'o', 'mg', 'al', 'si', 'ca', 'mn', 'ni']
NAMES = ['teff', 'logg', 'fe_h'] + [e + '_fe' for e in EL]
TEX = [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]'] + [f'[{e.capitalize()}/Fe]' for e in EL]
data = {}
for nm in ['train', 'test']:
    lab = labels[labels['sample'] == nm].reset_index(drop=True)
    f, e = cheb_normalise(*prepare(load(f'spectra_{nm}')), cont)
    good = np.isfinite(lab[NAMES]).all(axis=1).to_numpy()   # a few stars lack Al or Mn
    data[nm] = (lab[NAMES].to_numpy()[good], f[good], e[good])
L, flux, err = data['train']
Lt, ft, et = (a[::2] for a in data['test'])                  # every second test star (S/N 50-100), to save time
val = np.random.default_rng(10).permutation(len(L)) % 5 == 0  # 20% of the training set held out for validation
tr = ~val
print(f'training {tr.sum()}, validation {val.sum()}, test {len(Lt)} stars; {len(NAMES)} labels')


def scaling(lab):
    """Pivot (median) and scale (half the 2.5-97.5 percentile range) of a set of training labels."""
    return np.median(lab, axis=0), (np.percentile(lab, 97.5, axis=0) - np.percentile(lab, 2.5, axis=0)) / 2


def label_set(f, e, theta, s, pivot, scale, order=2, starts=None):
    """Test step for every spectrum: labels in physical units and chi^2."""
    out = [test_more(f[n], e[n], theta, s, order, starts) for n in range(len(f))]
    assert all(o[4] for o in out), 'the test-step optimiser did not converge for some spectra'
    return np.array([o[0] for o in out]) * scale + pivot, np.array([o[2] for o in out])


rob = lambda d: np.array([scatter_stats(d[:, k])['robust'] for k in range(d.shape[1])])   # robust std per label
pv, sc = scaling(L[tr])
x = (L - pv) / sc

# %% [markdown]
# ## 10.1 More labels, or more flexibility?
#
# Chapter 8 found that a model with six labels gives better $T_{\rm eff}$ and $\log g$ than one with three: part of
# the spectral effect of the abundances was being absorbed by the stellar parameters. Twelve labels push this
# further — but they also make the model more flexible: 91 coefficients per pixel instead of 10. A three-label model
# of *third* order has 20 coefficients and no abundances. Compare the three on the validation stars (high S/N, not
# used to train the models; they did help to choose the continuum pixels in Chapter 3, which Chapter 5 found to
# change the precision by less than 2%) and the test stars (S/N 50–100), with labels measured against ASPCAP's.

# %%
t0 = time.time()
pv3, sc3 = scaling(L[tr, :3])
models = {'3 labels, quadratic': (3, 2), '3 labels, cubic': (3, 3), '12 labels, quadratic': (12, 2)}
fits, res = {}, {}
for name, (K, order) in models.items():
    p, c = (pv3, sc3) if K == 3 else (pv, sc)
    th, s = train_more((L[tr, :K] - p) / c, flux[tr], err[tr], order=order)
    fits[name] = (th, s)
    lab_val = label_set(flux[val], err[val], th, s, p, c, order)[0]
    lab_test = label_set(ft, et, th, s, p, c, order)[0]
    res[name] = {'validation': rob(lab_val - L[val, :K]), 'test': rob(lab_test - Lt[:, :K])}
    print(f"{name:22s} D = {th.shape[1]:3d}   validation: {res[name]['validation'][:3].round(3)}   "
          f"test: {res[name]['test'][:3].round(3)}")
th12, s12 = fits['12 labels, quadratic']
lab12_test = label_set(ft, et, th12, s12, pv, sc)[0]
print('12 labels, abundances [X/Fe], robust std (validation / test):')
print(pd.DataFrame({'validation': res['12 labels, quadratic']['validation'][3:],
                    'test': res['12 labels, quadratic']['test'][3:]}, index=NAMES[3:]).T.round(3).to_string())
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, a in enumerate(ax):
    for j, split in enumerate(['validation', 'test']):
        a.bar(np.arange(3) + 0.38 * j - 0.19, [res[m][split][k] for m in models], 0.36,
              color=['C0', 'C1'][j], label=split + (' (high S/N)' if j == 0 else ' (S/N 50–100)'))
    a.set_xticks(range(3), [m.replace(', ', '\n') for m in models])
    a.set(ylabel=f'robust std of Cannon − ASPCAP [{["K", "dex", "dex"][k]}]', title=TEX[k])
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10.1.** Precision of $T_{\rm eff}$, $\log g$ and [Fe/H] (robust standard deviation of the differences from
# ASPCAP) for three models trained on the same 1308 stars: three labels with the quadratic model of the earlier
# chapters (10 coefficients per pixel), three labels with a cubic model (20), and twelve labels with a quadratic model
# (91). Blue: the 328 validation stars, of the same high S/N as the training set; orange: 741 test stars at
# S/N 50–100. On the validation stars both changes improve $T_{\rm eff}$ by a third (50 → 33–34 K): the gain from
# adding labels can be had by making the three-label model more flexible, without measuring a single abundance. At
# low S/N the abundances help more than the flexibility ($T_{\rm eff}$ 47 → 34 K with twelve labels, → 41 K with the
# cubic model; likewise $\log g$).
#
# **Checks.**

# %%
q3, c3, q12 = (res[m]['test'] for m in models)
v3, vc3, v12 = (res[m]['validation'] for m in models)
assert vc3[0] < 0.8 * v3[0] and v12[0] < 0.8 * v3[0], 'neither more flexibility nor more labels improves T_eff'
assert q12[0] < c3[0] < q3[0] and q12[1] < c3[1] < q3[1]

# %% [markdown]
# ## 10.2 L1 regularisation
#
# With 91 coefficients per pixel, many of them describe effects that are not there — a dependence on [Mn/Fe] at a
# pixel with no Mn line — and fitting them adds noise. C16 penalised the sum of their absolute values, which sets
# many of them exactly to zero (the "lasso", Tibshirani 1996). The strength $\Lambda$ is in units of $\chi^2$ and is
# chosen by validation. Following C16 (their Figs. 2–4), we scan $\Lambda$ and measure, on the validation stars:
# the fraction of coefficients that are exactly zero, how well the model predicts their spectra at their ASPCAP
# labels (the mean of $r^2/(\sigma^2+s^2)$ per pixel, with the scatter $s$ of the unregularised model for all), and
# the precision of the labels.

# %%
t0 = time.time()
LAMS = [0, 1e2, 3e2, 1e3, 3e3, 1e4]
Vval = vectorize_n(x[val])
use = th12[:, 0] != 0                                          # pixels with data
path = {}
for lam in LAMS:
    th, s = (th12, s12) if lam == 0 else train_more(x[tr], flux[tr], err[tr], lam)
    ok = np.isfinite(err[val][:, use])                        # pixels with zero weight are not counted
    z2 = np.where(ok, (flux[val][:, use] - Vval @ th[use].T)**2 / (err[val][:, use]**2 + s12[use]**2), np.nan)
    lab = label_set(flux[val], err[val], th, s, pv, sc)[0]
    path[lam] = {'zero': np.mean(th[use, 1:] == 0), 'chi2': np.nanmean(z2),
                 'rob': rob(lab - L[val])}
    print(f"Lambda = {lam:7.0f}: zero coefficients {path[lam]['zero']:.2f}, validation chi^2 per pixel "
          f"{path[lam]['chi2']:.3f}, T_eff {path[lam]['rob'][0]:.1f} K, median [X/Fe] {np.median(path[lam]['rob'][3:]):.4f}")
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
lx = np.array(LAMS[1:])
ax[0].semilogx(lx, [path[l]['zero'] for l in lx], 'ko-')
ax[0].set(xlabel=r'$\Lambda$', ylabel='fraction of coefficients exactly zero', title='(a) sparsity')
ax[1].semilogx(lx, [path[l]['chi2'] for l in lx], 'ko-')
ax[1].axhline(path[0]['chi2'], color='r', ls='--', label=r'$\Lambda = 0$')
ax[1].set(xlabel=r'$\Lambda$', ylabel=r'validation $\chi^2$ per pixel at ASPCAP labels', title='(b) predicted spectra')
ax[1].legend()
ratio = np.array([path[l]['rob'] / path[0]['rob'] for l in lx])
for k in range(len(NAMES)):
    ax[2].semilogx(lx, ratio[:, k], color='C0' if k < 3 else '0.6', lw=1.5 if k < 3 else 0.8,
                   label=r'$T_{\rm eff}, \log g$, [Fe/H]' if k == 0 else ('[X/Fe]' if k == 3 else None))
ax[2].axhline(1, color='r', ls='--')
ax[2].set(xlabel=r'$\Lambda$', ylabel=r'robust std / robust std at $\Lambda = 0$', title='(c) labels')
ax[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10.2.** L1 regularisation of the twelve-label model trained on 1308 stars, measured on the 328 validation
# stars. **(a)** The fraction of the coefficients (all but the mean spectrum, at pixels with data) that are exactly
# zero rises from 12% to most of them. **(b)** Moderate regularisation predicts the spectra of stars the model has not
# seen better, with a minimum near $\Lambda \approx 300$; strong regularisation predicts them worse. C16 (their
# Fig. 4) found the same shape.
# **(c)** But the labels do not get better: the robust standard deviation of every label, relative to the
# unregularised model, stays near 1 for weak regularisation and grows for strong regularisation (blue: stellar
# parameters; grey: the nine abundances). Predicting spectra and measuring labels are different tasks; with 1308
# training stars for 91 coefficients, the unregularised model is not overfitting enough for the penalty to help the
# labels.
#
# ### When does regularisation pay?
#
# Overfitting grows as the number of training stars approaches the number of coefficients. Repeat the comparison
# with random training sets of 120 and 300 stars (two of each, to see the scatter between random draws). The penalty
# is a $\chi^2$, which grows with the number of stars, so $\Lambda$ is scaled with it: $\Lambda = 300\,N/1308$ (for
# the full training set this is the $\Lambda = 300$ model above).

# %%
t0 = time.time()
row = lambda n, seed, reg, r: {'N': n, 'seed': seed, 'reg': reg, 'logg': r[1], 'teff': r[0], 'x_fe': np.median(r[3:])}
size = [row(tr.sum(), 0, reg, path[300 if reg else 0]['rob']) for reg in [False, True]]   # full set: from the scan
for n in [120, 300]:
    for seed in [11, 12]:
        sub = np.flatnonzero(tr)[np.random.default_rng(seed).permutation(tr.sum())[:n]]
        p, c = scaling(L[sub])
        for lam in [0, 300 * n / tr.sum()]:
            th, s = train_more((L[sub] - p) / c, flux[sub], err[sub], lam)
            size.append(row(n, seed, lam > 0, rob(label_set(flux[val], err[val], th, s, p, c)[0] - L[val])))
size = pd.DataFrame(size).sort_values(['N', 'seed', 'reg'])
print(size.round(4).to_string(index=False))
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (col, lab) in zip(ax, [('teff', r'$T_{\rm eff}$ [K]'), ('logg', r'$\log g$'), ('x_fe', 'median of the nine [X/Fe]')]):
    for reg, c, nm in [(False, 'k', r'$\Lambda = 0$'), (True, 'C3', r'$\Lambda = 300\,N/1308$')]:
        d = size[size.reg == reg]
        a.plot(d.N * (1.04 if reg else 0.96), d[col], 'o', color=c, label=nm)
        a.plot(d.groupby('N').N.first(), d.groupby('N')[col].mean(), '-', color=c, lw=1)
    a.set(xscale='log', xlabel='number of training stars', ylabel=f'robust std, {lab}')
ax[0].axhline(0, lw=0)
ax[0].legend()
for a in ax:
    a.axvline(91, color='0.6', ls=':')
ax[1].text(95, ax[1].get_ylim()[1] * 0.95, '91 coefficients', color='0.4', fontsize=8, va='top')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10.3.** Precision on the validation stars of twelve-label models trained on random subsets of 120 and 300
# stars (two subsets each) and on all 1308, without (black) and with (red) L1 regularisation; lines join the means.
# The dotted line marks the number of coefficients per pixel. With 120 training stars the unregularised model
# overfits — its $\log g$ is twice as bad as with 300 — and regularisation recovers much of the loss, in $\log g$ and
# in the abundances. With 300 or 1308 stars this penalty no longer helps. Two caveats: there are only two random
# subsets per size, and $\Lambda$ was not re-tuned for each size (a $\Lambda$ chosen separately for 300 stars might still
# help a little). C16 used regularisation for a 17-label model (171 coefficients per pixel); it matters when the
# training set is small for the number of labels.
#
# The abundances of the validation stars come out *better* from 300 training stars than from all 1308, for both
# random subsets and without regularisation. Appendix C investigates this with three splits, training-only
# continuum selection, exchanged coefficients/scatter, quality cuts and several test-step starting points.
# It finds that the original split's narrower central distribution does not mean uniformly smaller errors:
# the 300-star model has worse RMSE for five of nine abundances. The effect follows the learned coefficients
# more than the test-time scatter weights, and the other splits have smaller, less decisive differences.
#
# **Checks.**

# %%
g = size.groupby(['N', 'reg'])[['logg', 'x_fe']].mean()
small, full = 120, tr.sum()
assert g.loc[(small, True), 'logg'] < 0.8 * g.loc[(small, False), 'logg'], 'regularisation does not help at N = 120'
assert g.loc[(small, True), 'x_fe'] < g.loc[(small, False), 'x_fe']
assert g.loc[(full, True), 'x_fe'] > 0.97 * g.loc[(full, False), 'x_fe'], 'regularisation helps at full N after all'
assert min(path[l]['chi2'] for l in LAMS[1:]) < path[0]['chi2'] and path[1e4]['chi2'] > path[0]['chi2']
assert np.median(path[1e4]['rob'] / path[0]['rob']) > 1

# %% [markdown]
# ### Does this depend on how the scatter is fitted?
#
# Our `train_more` fits the coefficients with weights $1/(\sigma^2 + s^2)$ and $s$ by maximum likelihood, iterated.
# AnniesLasso fits the coefficients once with $s = 0$ and then sets $s$ so that the median $\chi^2$ of the training
# stars is 1 (`scatter='median'`, the appendix explains the difference). Its $\chi^2$ ignores $s$ and is therefore
# larger, so the same $\Lambda$ regularises less: it takes $\Lambda \approx 3000$ to zero the 27% of the
# coefficients that $\Lambda = 300$ zeroes in our recipe. Repeat the comparison with AnniesLasso's recipe at
# $\Lambda = 0$ and $\Lambda = 3000$.

# %%
t0 = time.time()
rec = {}
for lam in [0, 3000]:
    th, s = train_more(x[tr], flux[tr], err[tr], lam, scatter='median')
    rec[lam] = {'zero': np.mean(th[use, 1:] == 0),
                'validation': rob(label_set(flux[val], err[val], th, s, pv, sc)[0] - L[val]),
                'test': rob(label_set(ft, et, th, s, pv, sc)[0] - Lt)}
    print(f"AnniesLasso recipe, Lambda = {lam:4d}: zero coefficients {rec[lam]['zero']:.2f}; robust std "
          f"T_eff, log g, median [X/Fe]: validation {rec[lam]['validation'][0]:.1f} K, {rec[lam]['validation'][1]:.3f}, "
          f"{np.median(rec[lam]['validation'][3:]):.4f}; test {rec[lam]['test'][0]:.1f} K, {rec[lam]['test'][1]:.3f}, "
          f"{np.median(rec[lam]['test'][3:]):.4f}")
print(f'{time.time() - t0:.0f} s')

# %% [markdown]
# With AnniesLasso's recipe, too, regularising a quarter of the coefficients to zero does not improve the labels of
# either the validation or the test stars; the conclusion above does not depend on how the scatter is fitted. (The
# two recipes differ in other ways: without regularisation AnniesLasso's is slightly better on the high-S/N
# validation stars and worse in $T_{\rm eff}$ on the low-S/N test stars, and its labels degrade more slowly as
# $\Lambda$ grows.)
#
# **Checks.**

# %%
assert abs(rec[3000]['zero'] - path[300]['zero']) < 0.05, 'Lambda = 3000 no longer matches the sparsity of Lambda = 300'
for split in ['validation', 'test']:
    assert np.median(rec[3000][split][3:]) > 0.97 * np.median(rec[0][split][3:]), f'regularisation helps ({split})'

# %% [markdown]
# ## 10.3 Censoring: each abundance from its own lines
#
# Chapter 8 asked whether [Mg/Fe] is *measured* from Mg lines or *predicted* from the rest of the spectrum through
# its correlations with other labels. Censoring answers by construction: a coefficient that involves [X/Fe] may be
# non-zero only at pixels where element X has lines. APOGEE did this for the Cannon abundances of DR14, using the
# ASPCAP abundance windows (Holtzman et al. 2018, §8), because "correlations of abundances of different elements in
# the training set" would otherwise be imposed on every star labelled. We use our line list: the pixels within 1 Å
# of the 40 strongest lines of the element's neutral atom, plus the CO and CN bands for carbon, CN for nitrogen, and
# OH and CO for oxygen. $T_{\rm eff}$, $\log g$ and [Fe/H] act everywhere. A term like $T_{\rm eff}\times$[Mg/Fe] is
# allowed only where [Mg/Fe] is.

# %%
SPECIES = {'c': ['C  I', 'CO', 'CN'], 'n': ['N  I', 'CN'], 'o': ['O  I', 'OH', 'CO'], 'mg': ['MG I'], 'al': ['AL I'],
           'si': ['SI I'], 'ca': ['CA I'], 'mn': ['MN I'], 'ni': ['NI I']}


def near(species, half_width=1.0):
    """Pixels within half_width [A] of a line of the given species (vacuum wavelengths)."""
    w = lines[lines.species.isin(species)].wave_vac.to_numpy()
    return np.min(np.abs(wave[:, None] - w[None, :]), axis=1) < half_width


allowed = np.column_stack([np.ones((NPIX, 3), bool)] + [near(SPECIES[e]) for e in EL])
mask = censor_mask(allowed)
print('pixels allowed for each abundance:', dict(zip(EL, allowed[:, 3:].sum(axis=0))))
t0 = time.time()
thc, sc_c = train_more(x[tr], flux[tr], err[tr], mask=mask)
res_c = {'validation': rob(label_set(flux[val], err[val], thc, sc_c, pv, sc)[0] - L[val]),
         'test': rob(label_set(ft, et, thc, sc_c, pv, sc)[0] - Lt)}
print(pd.DataFrame({'uncensored': res['12 labels, quadratic']['test'], 'censored': res_c['test']}, index=NAMES).T.round(3).to_string())
print(f'{time.time() - t0:.0f} s')
dmg = th12[:, 1 + NAMES.index('mg_fe')] / sc[NAMES.index('mg_fe')]   # df/d[Mg/Fe] per dex, uncensored
on = th12[:, 0] != 0
out_frac = np.sum(np.abs(dmg[on & ~allowed[:, 6]])) / np.sum(np.abs(dmg[on]))
print(f'uncensored model: {out_frac:.0%} of the total |df/d[Mg/Fe]| lies at pixels more than 1 A from a Mg I line')

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1.4, 1]})
lo, hi = 15730, 15800
z = on & (wave > lo) & (wave < hi)
ax[0].fill_between(wave[z], -0.25, 0.1, where=allowed[z, 6], color='C2', alpha=0.15, lw=0, label='Mg I windows')
ax[0].plot(wave[z], th12[z, 0] - 1, 'k', lw=0.8, label=r'$\theta_0 - 1$')
ax[0].plot(wave[z], dmg[z], 'C0', lw=1.2, label=r'$\partial f/\partial$[Mg/Fe], uncensored')
ax[0].plot(wave[z], thc[z, 1 + NAMES.index('mg_fe')] / sc[NAMES.index('mg_fe')], 'C3', lw=1.2,
           label=r'$\partial f/\partial$[Mg/Fe], censored')
ax[0].set(xlim=(lo, hi), ylim=(-0.25, 0.1), xlabel='vacuum wavelength [Å]', ylabel='normalised flux (per dex)',
          title='(a) where [Mg/Fe] acts')
ax[0].legend(fontsize=8, ncol=2, loc='lower left')
for j, split in enumerate(['validation', 'test']):
    ax[1].bar(np.arange(12) + 0.38 * j - 0.19, res_c[split] / res['12 labels, quadratic'][split], 0.36,
              color=['C0', 'C1'][j], label=split)
ax[1].axhline(1, color='k', lw=0.8)
ax[1].set_xticks(range(12), TEX, rotation=60, fontsize=8)
ax[1].set(ylabel='robust std: censored / uncensored', title='(b) precision with censoring')
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10.4.** **(a)** The derivative of the flux with respect to [Mg/Fe] at the median labels (the linear
# coefficient; away from the median the cross terms change it) in a 70 Å window, in the uncensored
# (blue) and censored (red) twelve-label models; the green bands are the pixels within 1 Å of Mg I lines, the only
# ones where the censored model may use [Mg/Fe]. The uncensored model also gives [Mg/Fe] small derivatives
# everywhere else (the fraction printed above). **(b)** The robust standard deviation of each label with censoring,
# relative to the uncensored model, for the validation (high S/N) and test (S/N 50–100) stars. At low S/N censoring
# *improves* Mg, Al, Si, Ca, Mn and Ni (by 12–21%). A likely reason is that the small derivatives spread over
# thousands of pixels fit noise more than signal at low S/N; we have not proved it. At high S/N censoring improves
# only Mg, Al and Ca, and costs 6–17% in Si, Mn and Ni, which may draw on real information outside our windows — or on
# correlations with other labels in the training set; this experiment cannot tell which. It degrades O and C at
# both S/N — their information is spread over many molecular lines, which our 1 Å windows around 60 lines per molecule
# capture only in part — and $T_{\rm eff}$, which loses the cross terms with the abundances away from their lines.
# Holtzman et al. 2018 saw the same kind of element-by-element change for DR14 (worse Ca, still good Ni, with the
# ASPCAP windows). Censoring guarantees only that an abundance acts directly on its own pixels; it still couples to
# the other labels in the joint fit. All precisions here are measured against ASPCAP, which is not the truth.
#
# **Checks.**

# %%
rc = res_c['test'] / res['12 labels, quadratic']['test']
for el in ['mg_fe', 'al_fe', 'ca_fe', 'ni_fe']:
    assert rc[NAMES.index(el)] < 0.95, f'censoring does not improve {el} at low S/N'
assert rc[NAMES.index('o_fe')] > 1.1 and rc[0] > 1.1
assert out_frac > 0.5

# %% [markdown]
# ## 10.4 Three small tools
#
# **Several starting points.** The test step of Chapter 2 starts at the median labels. With twelve labels the
# $\chi^2$ surface has more room for local minima. Start also from four other points (low and high [Fe/H] and
# $T_{\rm eff}$, in scaled units) and keep, of the fits that converged, the one with the lowest $\chi^2$.
#
# **Inside the training set?** Chapter 5 flagged extrapolation with N15's EFLAG. AnniesLasso tests whether the labels
# lie inside the convex hull of the training labels, with a Delaunay triangulation; that does not work beyond a few
# dimensions, but a small linear program does (`in_hull`).
#
# **Saving a model** needs nothing but `np.savez`.

# %%
t0 = time.time()
starts = np.zeros((5, len(NAMES)))
starts[1, 2], starts[2, 2], starts[3, 0], starts[4, 0] = -1.5, 1.0, -1.5, 1.5
lab_many, chi_many = label_set(ft, et, th12, s12, pv, sc, starts=starts)
lab_one, chi_one = label_set(ft, et, th12, s12, pv, sc)
dchi = chi_one - chi_many
better = dchi > 1
print(f'several starting points: chi^2 lower by > 1 for {better.sum()} of {len(ft)} test stars '
      f'(largest improvement {dchi.max():.0f}); their T_eff changes by up to {np.abs(lab_many - lab_one)[better, 0].max():.0f} K')
closer = np.abs(lab_many - Lt)[better] < np.abs(lab_one - Lt)[better]     # (stars, labels): closer to ASPCAP?
print(f'for these stars the new labels are closer to ASPCAP in {closer.mean():.0%} of the (star, label) pairs; '
      f'T_eff closer for {closer[:, 0].sum()} of {better.sum()}')
mp = labels[labels['sample'] == 'metalpoor'][NAMES[:3]].to_numpy()
inside = {'test stars, 3 labels': in_hull(Lt[:, :3], L[tr, :3]), 'test stars, 12 labels': in_hull(Lt, L[tr]),
          'metal-poor stars, 3 labels': in_hull(mp, L[tr, :3])}
for k, v in inside.items():
    print(f'inside the training-set hull: {v.mean():.2f} of {k}')
import tempfile
path = Path(tempfile.mkdtemp()) / 'model_12labels.npz'
np.savez(path, theta=th12, s=s12, pivot=pv, scale=sc, names=NAMES)
with np.load(path) as m:
    assert np.array_equal(m['theta'], th12) and list(m['names']) == NAMES
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(lab_one[~better, 0], lab_many[~better, 0] - lab_one[~better, 0], '.', color='0.6', ms=3, label='same fit')
ax[0].plot(lab_one[better, 0], lab_many[better, 0] - lab_one[better, 0], 'o', color='C3', ms=5,
           label=r'lower $\chi^2$ from another start')
ax[0].set(xlabel=r'$T_{\rm eff}$ from the median start [K]', ylabel=r'$\Delta T_{\rm eff}$, best of 5 starts [K]',
          title='(a) several starting points')
ax[0].legend(fontsize=8)
ax[1].plot(L[tr, 0], L[tr, 1], '.', color='0.7', ms=2, label='training set')
ax[1].plot(mp[:, 0], mp[:, 1], '.', color='C3', ms=3, label='metal-poor sample (outside)')
ax[1].plot(Lt[inside['test stars, 12 labels'], 0], Lt[inside['test stars, 12 labels'], 1], '.', color='C0', ms=4,
           label='test stars inside the 12-label hull')
ax[1].invert_xaxis()
ax[1].invert_yaxis()
ax[1].set(xlabel=r'$T_{\rm eff}$ [K]', ylabel=r'$\log g$', title='(b) inside the training set?')
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10.5.** **(a)** Test stars labelled with the twelve-label model from the median labels only, and from the
# best of five starting points: for most stars the two fits are the same (grey); for a few (red) another start finds
# a lower $\chi^2$ and a different $T_{\rm eff}$. A lower $\chi^2$ is a better fit of the model, not necessarily
# better labels: the printout above says how often the new labels are closer to ASPCAP's. **(b)** Kiel diagram of the training set (grey) and of the
# metal-poor sample of Chapter 5 (red), all outside the convex hull of the training labels in ($T_{\rm eff}$,
# $\log g$, [Fe/H]). In three dimensions almost every test star is inside the hull (fraction printed above). In
# twelve dimensions only the blue ones are, fewer than one in five, although the test stars are disc giants drawn from
# the same survey as the training stars: in many dimensions almost every new point lies outside the hull of a finite
# sample. Being outside the twelve-dimensional hull therefore says little by itself (we have not measured whether
# those stars have worse labels). Testing a few labels is more useful, at the price of missing extrapolation in the
# others.
#
# **Checks.**

# %%
assert np.all(dchi > -1e-6) and better.sum() >= 1, 'more starting points must never give a worse fit'
assert inside['test stars, 3 labels'].mean() > 0.95 and inside['metal-poor stars, 3 labels'].mean() < 0.2
assert inside['test stars, 12 labels'].mean() < 0.5

# %% [markdown]
# ## Summary
#
# - Twelve labels make $T_{\rm eff}$ about a third more precise than three — but so does a cubic three-label model.
#   Before attributing a gain to the extra labels, try a more flexible model with the same labels.
# - L1 regularisation (C16) makes the model sparse and, when moderate, its predicted spectra better. In our tests it
#   improved the labels only when the training set was not much larger than the number of coefficients (at 120
#   stars, not at 300 or 1308; two random subsets per size, $\Lambda$ not re-tuned per size), whether the scatter is
#   fitted our way or AnniesLasso's.
# - Censoring makes each abundance a measurement from its own lines. At S/N 50–100 it improves Mg, Al, Si, Ca, Mn and
#   Ni, and degrades O, C and $T_{\rm eff}$; the outcome depends on the element and on the windows (Holtzman et al.
#   2018).
# - Several starting points find a better fit for a few test stars in a hundred, some hundreds of kelvin away —
#   but the new labels are closer to ASPCAP's for only about half of them. A convex-hull test of extrapolation is
#   informative in three dimensions; in twelve it flags most ordinary stars.
# - The appendix compares our regularised training step with AnniesLasso's.
#
# ## References
#
# - Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. 2011, *Distributed optimization and statistical learning
#   via the alternating direction method of multipliers*, Foundations and Trends in Machine Learning 3, 1,
#   [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016) — ADMM for the lasso, §6.4.
# - Casey, A. R., Hogg, D. W., Ness, M., et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for
#   detailed chemical abundance analyses*, [arXiv:1603.03040](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
#   — L1-regularised 17-label model; choice of $\Lambda$ by validation.
# - Casey, A. R., Hawkins, K., Hogg, D. W., et al. 2017, *The RAVE-on catalog of stellar atmospheric parameters and
#   chemical abundances for chemo-dynamic studies in the Gaia era*,
#   [ApJ 840, 59](https://ui.adsabs.harvard.edu/abs/2017ApJ...840...59C) — regularisation chosen by cross-validation.
# - Holtzman, J. A., Hasselquist, S., Shetrone, M., et al. 2018, *APOGEE Data Releases 13 and 14: Data and
#   Analysis*, [AJ 156, 125](https://ui.adsabs.harvard.edu/abs/2018AJ....156..125H) — the censored Cannon of DR14.
# - Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, *The Cannon: a data-driven approach to
#   stellar label determination*, [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N).
# - Tibshirani, R. 1996, *Regression shrinkage and selection via the lasso*, JRSS B 58, 267,
#   [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x).
