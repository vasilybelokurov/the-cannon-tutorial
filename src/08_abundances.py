# %% [markdown]
# # Chapter 8 — More labels: element abundances
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/08_abundances.ipynb)
#
# **Goal.** Extend the model from three to six labels — adding [Mg/Fe], [C/Fe] and [N/Fe] — as Casey et al. 2016
# (C16, [arXiv:1603.03040](https://arxiv.org/abs/1603.03040)) and Ness et al. 2018 (N18) did with up to 20 labels.
# Then ask the question every data-driven abundance must answer: is the label **measured** from the lines of the
# element, or **predicted** from other labels it correlates with?
#
# **You will learn**
# 1. to train and cross-validate a six-label model (28 coefficients per pixel);
# 2. to check where in the spectrum each abundance label acts (Mg I lines; CO and CN bands);
# 3. to map the two α sequences of the Galactic disc with S/N = 50–100 spectra;
# 4. **censoring tests**: hide the Mg lines from the model, and compare with a prediction from the other labels.
#
# Why C and N? In red giants the surface [C/N] changes in the first dredge-up by an amount that depends on the
# star's mass, so [C/N] carries age information (Martig et al. 2016, [MNRAS 456, 3655](https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3655M);
# Ness et al. 2016, [ApJ 823, 114](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..114N)). In the $H$ band C and N
# are measured from molecular bands: CO for carbon, CN for nitrogen (given carbon), and OH for oxygen.

# %%
#@@LOADER
#@@NORM
#@@CANNON
import time

labels = load('labels')
lines = load('linelist')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES = ['teff', 'logg', 'fe_h', 'mg_fe', 'c_fe', 'n_fe']
TEX = [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]', '[Mg/Fe]', '[C/Fe]', '[N/Fe]']
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
L = tr[NAMES].to_numpy()
flux, err = cheb_normalise(*prepare(load('spectra_train')), cont)
print(f'{len(tr)} training stars; label ranges:')
print(tr[NAMES].describe().loc[['min', '50%', 'max']].round(2).to_string())


def scaling(lab):
    return np.median(lab, axis=0), (np.percentile(lab, 97.5, axis=0) - np.percentile(lab, 2.5, axis=0)) / 2


def cross_validate(f, e, L, nfold=5, seed=8):
    """nfold cross-validation: labels of every star from a model trained on the other folds."""
    fold = np.random.default_rng(seed).permutation(len(L)) % nfold
    out = np.zeros_like(L)
    for k in range(nfold):
        t = fold != k
        pv, sc = scaling(L[t])
        th, _, s, _ = train_clean((L[t] - pv) / sc, f[t], e[t])
        out[~t] = np.array([test(f[n], e[n], th, s)[0] for n in np.flatnonzero(~t)]) * sc + pv
    return out

# %% [markdown]
# ## 8.1 A six-label model, cross-validated
#
# The label vector now has $1 + 6 + 21 = 28$ entries. The training step is the same weighted least squares, with
# 28 instead of 10 coefficients per pixel; the training set (~1700 stars) is still much larger than that. We
# cross-validate with 5 folds.

# %%
t0 = time.time()
cv = cross_validate(flux, err, L)
print(f'5-fold cross-validation of the 6-label model: {time.time() - t0:.0f} s')
d = cv - L
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for k, a in enumerate(ax.flat):
    a.plot(L[:, k], cv[:, k], '.', ms=2, alpha=0.6)
    lim = np.percentile(L[:, k], [0.5, 99.5])
    a.plot(lim, lim, 'k-', lw=0.8)
    a.set(xlabel='ASPCAP ' + TEX[k], ylabel='Cannon ' + TEX[k], title=f'rms {d[:, k].std():.3g}, bias {d[:, k].mean():+.2g}')
plt.tight_layout()
ratio = d.std(0) / L.std(0)                                       # rms relative to the spread of each label
print('cross-validation rms:', dict(zip(NAMES, d.std(0).round(3))))
print('rms / spread of the label in the training set:', dict(zip(NAMES, ratio.round(2))))
# the same folds with the three-label model of Chapters 4-5
cv3 = cross_validate(flux, err, L[:, :3])
d3 = cv3 - L[:, :3]
print('three-label model on the same folds, rms:', dict(zip(NAMES[:3], d3.std(0).round(3))))
# Tests: (1) each abundance is recovered with an rms well below its spread (budgets frozen after a development run:
# measured ratios 0.49, 0.57, 0.80 for Mg, C, N); (2) adding the abundance labels improves T_eff and log g.
assert ratio[3] < 0.6 and ratio[4] < 0.7 and ratio[5] < 0.9
assert d[:, 0].std() < d3[:, 0].std() and d[:, 1].std() < d3[:, 1].std()

# %% [markdown]
# **Figure 8.1.** Five-fold cross-validation of the six-label model: Cannon against ASPCAP labels for the training
# stars (each labelled by a model that did not see it), with the standard deviation and bias of the differences.
# Two results stand out. First, the stellar parameters are *better* than with three labels on the same folds
# (≈ 53 K and 0.15 dex instead of ≈ 72 K and 0.19 dex): a three-label model must absorb the spectral effects of
# [Mg/Fe], [C/Fe] and [N/Fe] somewhere, and part of them leaks into $T_{\rm eff}$ and $\log g$ — N15 suspected this
# when they noted that the lack of [α/Fe] may inflate their scatter. Second, the abundances are recovered with a
# standard deviation of 0.06 ([Mg/Fe]), 0.10 ([C/Fe]) and 0.11 dex ([N/Fe]): about half the spread of [Mg/Fe] and
# [C/Fe] in the training set, but only 20% better than no information for [N/Fe], which is measured from CN lines
# whose strength also depends on carbon.
#
# **Comparison with the literature.** Casey et al. 2016 (Fig. 6; 12 681 training stars, 17 labels, L1-regularised)
# reach 0.04, 0.06 and 0.06 dex for Mg, C and N, and Leung & Bovy 2019 (astroNN, robust statistic) 0.027, 0.040 and
# 0.041 dex; Ness et al. 2018 quote 0.02–0.12 dex for 20 elements. Those numbers are for [X/H] (ours are [X/Fe],
# which also carries the error of [Fe/H]), and come from much larger training sets and more flexible or regularised
# models; ours are 1.5–3 times larger.
#
# ## 8.2 Where does each abundance act?
#
# Train on all stars and look at the derivatives of the flux with respect to the abundance labels, next to the
# lines of the element. A label that is measured from its own lines should have large derivatives there.

# %%
pv, sc = scaling(L)
th, _, s, _ = train_clean((L - pv) / sc, flux, err)
dfdl = th[:, 1:7] / sc                                            # per unit label (per K, per dex)
on = th[:, 0] != 0
strong = lambda sp, dex: lines[(lines.species == sp) & (lines.strength > lines[lines.species == sp].strength.max() - dex)].wave_vac
fig, ax = plt.subplots(3, 1, figsize=(14, 9))
for a, (k, sp, (lo, hi)) in zip(ax, [(3, 'MG I', (15730, 15800)), (4, 'CO', (16180, 16250)), (5, 'CN', (15240, 15310))]):
    z = on & (wave > lo) & (wave < hi)
    a.plot(wave[z], th[z, 0] - 1, 'k', lw=0.8, label=r'$\theta_0 - 1$')
    for j, c in zip(range(3, 6), ['C0', 'C1', 'C2']):
        a.plot(wave[z], dfdl[z, j], color=c, lw=1, label=r'$\partial f/\partial$' + TEX[j] + ' (per dex)')
    w = strong(sp, 1.5)
    a.vlines(w[(w > lo) & (w < hi)], 0.12, 0.17, color='k', lw=1.2, label=f'{sp} lines')
    a.set(xlim=(lo, hi), ylim=(-0.6, 0.2), ylabel='normalised flux')
    a.legend(fontsize=7, ncol=5, loc='lower left')
ax[-1].set(xlabel='vacuum wavelength [Å]')
plt.tight_layout()
dist = lambda sp, dex: np.min(np.abs(wave[:, None] - strong(sp, dex).to_numpy()[None, :]), axis=1)
ratio = {}
for k, sp in [(3, 'MG I'), (4, 'CO'), (5, 'CN')]:
    near = on & (dist(sp, 1.5) < 0.3)
    ratio[NAMES[k]] = np.mean(np.abs(dfdl[near, k])) / np.mean(np.abs(dfdl[on, k]))
    print(f'{NAMES[k]}: mean |df/dlabel| at {near.sum()} pixels within 0.3 A of strong {sp} lines / at all pixels = {ratio[NAMES[k]]:.1f}')
# Test: each abundance label acts most strongly at the lines of its own element/molecule (> 2x the average pixel).
assert all(v > 2 for v in ratio.values())

# %% [markdown]
# **Figure 8.2.** Derivatives of the normalised flux with respect to [Mg/Fe] (blue), [C/Fe] (orange) and [N/Fe]
# (green), per dex, in three 70 Å windows, with the baseline spectrum (black, minus 1) and the strongest lines of
# one species per window (ticks): the Mg I lines (top), CO lines (middle) and CN lines (bottom). Each abundance
# derivative is largest at the lines of its own element: the Mg I cores deepen with [Mg/Fe], the CO lines with
# [C/Fe] only, the CN lines with both [C/Fe] and [N/Fe] (a CN molecule needs both atoms). The model found these lines
# by itself, from the labels alone — C16 used this to identify previously unknown lines. Note also the *positive*
# [Mg/Fe] derivative at many CN lines (bottom): no physics makes CN weaker when Mg increases; the model has learned a
# correlation between [Mg/Fe] and the C and N labels of the training stars. Section 8.4 follows this up.
#
# ## 8.3 The two α sequences at S/N = 50–100
#
# Apply the six-label model to the 1500 giants of the `test` sample, with S/N between 50 and 100 (3–6 times noisier
# than the training spectra), and plot [Mg/Fe] against [Fe/H].

# %%
te = labels[labels['sample'] == 'test'].reset_index(drop=True)
ft, et = cheb_normalise(*prepare(load('spectra_test')), cont)
t0 = time.time()
lab_t = np.array([test(ft[n], et[n], th, s)[0] for n in range(len(te))]) * sc + pv
print(f'labelled {len(te)} test stars in {time.time() - t0:.0f} s')
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(te.fe_h, te.mg_fe, '.', ms=3, color='0.4')
ax[0].set(title=f'ASPCAP ({len(te)} stars, S/N 50-100)')
ax[1].plot(lab_t[:, 2], lab_t[:, 3], '.', ms=3, color='C3')
ax[1].set(title='The Cannon, six labels')
for a in ax[:2]:
    a.set(xlabel='[Fe/H]', ylabel='[Mg/Fe]', xlim=(-1.6, 0.5), ylim=(-0.2, 0.5))
band = (lab_t[:, 2] > -0.7) & (lab_t[:, 2] < -0.3)
hb = np.arange(-0.15, 0.45, 0.025)
ax[2].hist(lab_t[band, 3], bins=hb, histtype='step', color='C3', label='Cannon')
ax[2].hist(te.mg_fe[(te.fe_h > -0.7) & (te.fe_h < -0.3)], bins=hb, histtype='step', color='0.4', label='ASPCAP')
ax[2].set(xlabel='[Mg/Fe]', ylabel='stars', title='−0.7 < [Fe/H] < −0.3')
ax[2].legend(fontsize=8)
plt.tight_layout()
dm = lab_t[:, 3] - te.mg_fe.to_numpy()
ok = np.isfinite(dm)
print(f'[Mg/Fe] Cannon - ASPCAP on the test set: bias {dm[ok].mean():+.3f}, rms {dm[ok].std():.3f} dex')
# Which sequence is each star on? Split at [Mg/Fe] = 0.2 (the trough between the sequences in ASPCAP) and compare.
hi_a, hi_c = te.mg_fe.to_numpy() > 0.2, lab_t[:, 3] > 0.2
agree = np.mean(hi_a[band] == hi_c[band])
print(f'-0.7 < [Fe/H] < -0.3: same sequence as ASPCAP for {agree:.0%} of {band.sum()} stars; mean [Mg/Fe] of the '
      f'high-alpha stars: Cannon {lab_t[band & hi_a, 3].mean():.3f}, ASPCAP {te.mg_fe[band & hi_a].mean():.3f}; '
      f'low-alpha: Cannon {lab_t[band & ~hi_a, 3].mean():.3f}, ASPCAP {te.mg_fe[band & ~hi_a].mean():.3f}')
# Test (frozen after a development run, measured 88%): at least 80% of the stars are put on the same sequence as by
# ASPCAP, and the mean [Mg/Fe] of each sequence agrees within 0.03 dex.
assert agree > 0.8
assert abs(lab_t[band & hi_a, 3].mean() - te.mg_fe[band & hi_a].mean()) < 0.03
assert abs(lab_t[band & ~hi_a, 3].mean() - te.mg_fe[band & ~hi_a].mean()) < 0.03

# %% [markdown]
# **Figure 8.3.** [Mg/Fe] against [Fe/H] for 1500 giants with S/N = 50–100, from ASPCAP (left) and from the
# six-label Cannon model trained at S/N > 200 (middle), and the [Mg/Fe] distributions at $-0.7 <$ [Fe/H] $< -0.3$
# (right; bins of 0.025 dex). The Cannon recovers the high-α (thick-disc) sequence — its mean [Mg/Fe] is the same
# as ASPCAP's — and puts 88% of the stars in this [Fe/H] range on the same sequence as ASPCAP. But its sequences are
# broader (rms difference from ASPCAP 0.07 dex), the low-α sequence is smeared down to [Mg/Fe] ≈ −0.1 and up into the
# gap, and the gap that is so clean in ASPCAP is partly filled. At this S/N, ASPCAP's [Mg/Fe] (fitted to the Mg
# lines with physical models) separates the populations better than our six-label Cannon. (The jagged histogram
# reflects ~600 stars in 0.025-dex bins.)
#
# ## 8.4 Measured or predicted?
#
# [Mg/Fe] correlates with other labels: in the disc, high-α stars are metal-poor, and C and N also differ between the
# two sequences. A model can therefore "predict" [Mg/Fe] from the rest of the spectrum without reading a single Mg
# line (C16; Ho et al. 2017 warn about this for label transfer). Experiments, all cross-validated on the same five
# folds:
#
# 1. **Hide the Mg lines**: zero weight, in training and testing, for every pixel within 1.5 Å of any of the 40 Mg I
#    lines of our line list (the strongest ones; weaker Mg lines and blends remain).
# 2. **A control**: hide the same number of randomly chosen pixels instead.
# 3. **Correlations only**: predict [Mg/Fe] with a quadratic function of the other five *ASPCAP* labels, fitted on the
#    training folds — no spectrum at all.
# 4. **Correlations via The Cannon, without Mg lines**: train a five-label model (no [Mg/Fe]) with the Mg lines
#    hidden, label the stars with it, and apply the regression of experiment 3 to those labels.

# %%
mg_pix = dist('MG I', 99) < 1.5                                  # within 1.5 A of any listed Mg I line
rand_pix = np.zeros(NPIX, bool)
rand_pix[np.random.default_rng(1).choice(np.flatnonzero(np.isfinite(err).mean(0) > 0.5), mg_pix.sum(), replace=False)] = True
err_nomg = np.where(mg_pix[None, :], np.inf, err)
print(f'hiding {mg_pix.sum()} pixels around {(lines.species == "MG I").sum()} Mg I lines (or as many random pixels)')
t0 = time.time()
cv_mg = cross_validate(flux, err_nomg, L)
cv_rand = cross_validate(flux, np.where(rand_pix[None, :], np.inf, err), L)
others = [0, 1, 2, 4, 5]                                          # T_eff, log g, [Fe/H], [C/Fe], [N/Fe]
cv5 = cross_validate(flux, err_nomg, L[:, others])                # five labels, Mg lines hidden
fold = np.random.default_rng(8).permutation(len(L)) % 5           # the folds used by cross_validate
pred_aspcap, pred_cannon = np.zeros(len(L)), np.zeros(len(L))
for k in range(5):
    t = fold != k
    pv5, sc5 = scaling(L[t][:, others])
    coef = np.linalg.lstsq(vectorize((L[t][:, others] - pv5) / sc5), L[t, 3], rcond=None)[0]     # fit on reference labels
    pred_aspcap[~t] = vectorize((L[~t][:, others] - pv5) / sc5) @ coef                          # 3: ASPCAP labels in
    pred_cannon[~t] = vectorize((cv5[~t] - pv5) / sc5) @ coef                                   # 4: Cannon labels in
print(f'four experiments: {time.time() - t0:.0f} s')
preds = {'six-label Cannon': cv[:, 3], '1. Mg lines hidden': cv_mg[:, 3], '2. random pixels hidden': cv_rand[:, 3],
         '3. other 5 ASPCAP labels': pred_aspcap, '4. other 5 Cannon labels, no Mg lines': pred_cannon}
st = pd.DataFrame({nm: scatter_stats(y - L[:, 3]) for nm, y in preds.items()}).T
st.loc['no information', ['std', 'robust']] = L[:, 3].std(), 1.4826 * np.median(np.abs(L[:, 3] - np.median(L[:, 3])))
print(st.round(4).to_string())
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
for a, (nm, y) in zip(ax, preds.items()):
    a.plot(L[:, 3], y, '.', ms=2, alpha=0.6)
    a.plot([-0.2, 0.45], [-0.2, 0.45], 'k-', lw=0.8)
    a.set(xlabel='ASPCAP [Mg/Fe]', ylabel='[Mg/Fe]', title=f'{nm}\nstd {st.loc[nm, "std"]:.3f}, robust {st.loc[nm, "robust"]:.3f}')
    a.title.set_fontsize(9)
plt.tight_layout()
six, nomg, rnd = st.loc['six-label Cannon', 'std'], st.loc['1. Mg lines hidden', 'std'], st.loc['2. random pixels hidden', 'std']
# Tests: (1) hiding the Mg lines degrades [Mg/Fe] by > 10%, hiding as many random pixels by < 3%: the model reads the
# Mg lines; (2) yet the six-label [Mg/Fe] is at most 10% better than correlations with the other labels, whether these
# are ASPCAP's or The Cannon's measured without Mg lines.
assert nomg > 1.1 * six and rnd < 1.03 * six
assert st.loc['3. other 5 ASPCAP labels', 'std'] < 1.1 * six and st.loc['4. other 5 Cannon labels, no Mg lines', 'std'] < 1.1 * six

# %% [markdown]
# **Figure 8.4.** Cross-validated [Mg/Fe] against ASPCAP's for five predictors (titles: standard deviation and
# robust standard deviation of the difference): the six-label model; the same with all pixels near the 40 Mg I lines
# hidden in training and testing; the same with as many random pixels hidden; a quadratic function of the other five
# ASPCAP labels, without any spectrum; and the same function applied to the labels of a five-label Cannon model that
# never saw [Mg/Fe] nor the Mg lines. Hiding the Mg lines degrades [Mg/Fe] by ≈ 16%, hiding random pixels does
# nothing: the model does read the Mg lines. (Hiding only the five strongest Mg lines has no effect: the weaker lines
# carry the same information.) But the last two panels are nearly as good as the first: with these training data,
# the six-label model's [Mg/Fe] is at most 5–10% more precise (in standard and in robust deviation) than what follows
# from $T_{\rm eff}$, $\log g$, [Fe/H], [C/Fe] and [N/Fe] through the correlations among disc stars — even when those
# labels are measured without the Mg lines. A cross-validation rms alone therefore cannot show that an abundance is
# *measured*. For stars whose Mg does not follow the
# training-set correlations — a dwarf galaxy, a globular cluster, an accreted halo star — the inferred [Mg/Fe] may be
# pulled towards the disc relation with a deceptively small error. This is why C16 regularised the model (an L1
# penalty, so that each label acts on few pixels), and why every data-driven abundance needs tests like these before
# it is trusted for stars unlike the training set. (The test in reverse — stars whose [Mg/Fe] departs from the disc
# relation — needs a different sample; exercise 3.)
#
# ## 8.5 How redundant is the chemistry of the disc?
#
# Section 8.4 is not a failure of The Cannon. It reflects a property of the Milky Way disc that Ness and
# collaborators have measured directly from abundances: once a few labels are known, most others follow. Given
# [Fe/H] and age, the other [X/Fe] of low-α disc stars are predicted to ≈ 0.02 dex (Ness et al. 2019); given [Fe/H]
# and [Mg/H], elements such as Si, O, Ca, Ni, Al and Mn are predicted to about twice their measurement errors, leaving
# an intrinsic scatter of only ≈ 0.015–0.04 dex (Ness et al. 2022); a handful of elements predict most α and iron-peak
# elements to 0.03 dex (Mead et al. 2025), and four nucleosynthetic patterns describe 16 elements of 70 000 giants
# (Ness et al. 2026). Griffith et al. (2021) predict the abundances of bulge stars from [Mg/Fe] and [Mg/H] alone.
#
# We can see the same in our training set, with ASPCAP's labels and no spectra: predict each [X/H] from
# ([Fe/H], [Mg/H]) with a quadratic function, cross-validated with 5 folds, and compare the robust scatter of the
# residuals with the spread of the element and with ASPCAP's quoted error.

# %%
rstd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def cv_regression(X, y, nfold=5):
    """Cross-validated prediction of y from a quadratic function of the columns of X (standardised)."""
    Xs = (X - X.mean(0)) / X.std(0)
    V = vectorize(Xs)                                             # 1, linear and quadratic terms (the Cannon vector)
    f = np.random.default_rng(0).permutation(len(y)) % nfold
    p = np.zeros(len(y))
    for k in range(nfold):
        p[f == k] = V[f == k] @ np.linalg.lstsq(V[f != k], y[f != k], rcond=None)[0]
    return y - p


ELEMS = ['si', 'ni', 'ca', 'o', 'mn', 'al', 'c', 'n']
rows = []
for e in ELEMS:
    ok = (tr[f'{e}_fe_flag'] == 0).to_numpy()
    y = (tr[f'{e}_fe'] + tr.fe_h).to_numpy()[ok]                  # [X/H]
    X = np.c_[tr.fe_h, tr.mg_fe + tr.fe_h][ok]                    # [Fe/H], [Mg/H]
    rows.append((f'[{e.capitalize()}/H]', rstd(cv_regression(X, y)), y.std(), tr[f'{e}_fe_err'].median()))
red = pd.DataFrame(rows, columns=['element', 'robust residual', 'spread', 'ASPCAP error']).set_index('element')
print(red.round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(red))
ax.bar(x - 0.2, red['robust residual'], 0.4, label='robust scatter of the prediction from [Fe/H], [Mg/H]')
ax.bar(x + 0.2, red['ASPCAP error'], 0.4, label='median ASPCAP error')
ax.set(xticks=x, xticklabels=red.index, ylabel='dex')
ax.legend(fontsize=8)
plt.tight_layout()
# Test: Si, Ni and Ca follow from Fe and Mg to < 0.05 dex, less than a tenth of their spread in the training set.
for el in ['[Si/H]', '[Ni/H]', '[Ca/H]']:
    assert red.loc[el, 'robust residual'] < 0.05 and red.loc[el, 'robust residual'] < 0.1 * red.loc[el, 'spread']

# %% [markdown]
# **Figure 8.5.** For eight elements, the robust scatter of the cross-validated prediction of [X/H] from [Fe/H] and
# [Mg/H] (ASPCAP labels of the 1703 training giants, quadratic regression, no spectra; blue), and the median error
# that ASPCAP quotes for that element (orange). Si, Ni and Ca follow from Fe and Mg to ≈ 0.03–0.04 dex, 2–2.5 times
# their quoted errors and less than a tenth of their range — as Ness et al. (2022) found; O, Mn and Al are less
# tightly tied, and C and N, which change inside red giants with mass, least of all.
#
# **What this means for a data-driven abundance.** For disc stars, an abundance is largely predictable from a few
# labels; the star-specific part is at the 0.015–0.04 dex level (Ness et al. 2022), below the ≈ 0.05 dex precision of
# our six-label [Mg/Fe]. A model trained on disc stars learns these relations and uses them, so cross-validation on
# disc stars cannot tell a measured abundance from a predicted one (§8.4). That is harmless for typical disc stars and
# misleading for stars that do not follow the disc relations — accreted halo stars, globular-cluster stars, stars of
# dwarf galaxies — whose abundances are pulled towards the disc pattern. Ness et al. (2026) turn this around: the
# stars that their four disc patterns fail to describe are the interesting ones.
#
# ## Summary
#
# - Six labels, 28 coefficients per pixel, trained on ~1700 stars: [Mg/Fe], [C/Fe] and [N/Fe] are recovered with
#   rms of a few hundredths of a dex.
# - The abundance derivatives are largest at the lines of each element (Mg I) or molecule (CO, CN).
# - The disc's chemistry is largely redundant (Ness et al. 2019, 2022): Si, Ni and Ca follow from Fe and Mg to ≈ 0.03
#   dex. A data-driven abundance of a disc star is therefore partly a prediction, which matters for unusual stars.
# - At S/N 50–100 the model separates the high- and low-α sequences of the disc for most stars, but less cleanly
#   than ASPCAP.
# - The model reads the Mg lines (hiding them costs 16% in precision), yet its [Mg/Fe] is at most 5–10% better than a
#   prediction from the other labels: a cross-validation rms does not prove that an abundance is measured.
#
# **Exercises.**
# 1. Censor the CN lines and repeat for [N/Fe]. Is [N/Fe] more or less "measured" than [Mg/Fe]?
# 2. Add [Al/Fe] as a seventh label (column `al_fe` in `labels`; keep only stars with `al_fe_flag == 0`). Where does
#    the model place its derivative? (The H band has two strong Al I lines near 16 723 and 16 755 Å vacuum.)
# 3. Find training or test stars whose ASPCAP [Mg/Fe] is far (> 0.1 dex) from the prediction of experiment 3. Does
#    the six-label model follow ASPCAP for them, or the correlation?

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Griffith et al. 2021, *The Similarity of Abundance Ratio Trends and Nucleosynthetic Patterns in the Milky Way Disk and Bulge*, ApJ 909, 77. [ADS](https://ui.adsabs.harvard.edu/abs/2021ApJ...909...77G)
# - Ho et al. 2017, *Label Transfer from APOGEE to LAMOST: Precise Stellar Parameters for 450,000 LAMOST Giants*, ApJ 836, 5. [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H)
# - Hogg et al. 2016, *Chemical Tagging Can Work: Identification of Stellar Phase-space Structures Purely by Chemical-abundance Similarity*, ApJ 833, 262. [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...833..262H)
# - Leung & Bovy 2019, *Deep learning of multi-element abundances from high-resolution spectroscopic data*, MNRAS 483, 3255. [ADS](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L)
# - Martig et al. 2016, *Red giant masses and ages derived from carbon and nitrogen abundances*, MNRAS 456, 3655. [ADS](https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3655M)
# - Mead et al. 2025, *Four Elements to Rule Them All: Abundances Are Rigidly Coupled in the Milky Way Disk*, ApJ 994, 52. [ADS](https://ui.adsabs.harvard.edu/abs/2025ApJ...994...52M)
# - Ness et al. 2016, *Spectroscopic Determination of Masses (and Implied Ages) for Red Giants*, ApJ 823, 114. [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..114N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
# - Ness et al. 2019, *In the Galactic Disk, Stellar [Fe/H] and Age Predict Orbits and Precise [X/Fe]*, ApJ 883, 177. [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...883..177N)
# - Ness et al. 2022, *The Homogeneity of the Star-forming Environment of the Milky Way Disk over Time*, ApJ 926, 144. [ADS](https://ui.adsabs.harvard.edu/abs/2022ApJ...926..144N)
# - Ness et al. 2026, *Milky Way Mapper decoded abundances - I. Shared disc enrichment patterns*, MNRAS 548, stag637. [ADS](https://ui.adsabs.harvard.edu/abs/2026MNRAS.548ag637N)
# - Smith et al. 2021, *The APOGEE Data Release 16 Spectral Line List*, AJ 161, 254. [ADS](https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S)
# - Ting et al. 2019, *The Payne: Self-consistent ab initio Fitting of Stellar Spectra*, ApJ 879, 69. [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T)
