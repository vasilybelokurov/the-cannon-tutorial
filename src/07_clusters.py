# %% [markdown]
# # Chapter 7 — Star clusters as a test
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/07_clusters.ipynb)
#
# **Goal.** Use star clusters as an external check of the labels. The stars of a cluster formed together: they
# share an age, a distance and — to a good approximation for iron in open clusters — a chemical composition. Their
# giants must therefore lie on one sequence in the Kiel diagram and have the same [Fe/H]. N15 *trained* on
# cluster stars; we never did (the training set excludes all members), so the clusters are an independent test.
#
# **You will learn**
# 1. to compare cluster mean metallicities between two methods, across clusters (a constant predictor would fail);
# 2. to estimate the spread of a label within a cluster and compare it with the measurement errors;
# 3. where the test stops being a test: globular clusters below the training range.
#
# Caveats: APOGEE's `member` flag is a preliminary membership (position, radial velocity, proper motion), not a
# vetted probability; and globular clusters are *not* homogeneous in light elements (C, N, O, Na, Mg, Al vary from
# star to star), so we only use [Fe/H] for them.

# %%
#@@LOADER
#@@NORM
#@@CANNON

labels = load('labels')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES, TEX = ['teff', 'logg', 'fe_h'], [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]']
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
L = tr[NAMES].to_numpy()
pv = np.median(L, axis=0)
sc = (np.percentile(L, 97.5, axis=0) - np.percentile(L, 2.5, axis=0)) / 2
th, _, s, _ = train_clean((L - pv) / sc, *cheb_normalise(*prepare(load('spectra_train')), cont))
cl = labels[labels['sample'] == 'clusters'].reset_index(drop=True)
fc, ec = cheb_normalise(*prepare(load('spectra_clusters')), cont)
out = [test(fc[n], ec[n], th, s) for n in range(len(cl))]
can = np.array([o[0] for o in out]) * sc + pv
cl['teff_c'], cl['logg_c'], cl['fe_h_c'] = can.T
cl['chi2nu'] = [o[2] / (o[3] - 3) for o in out]
OPEN = ['M67', 'N6819', 'N188', 'N7789', 'N2420', 'N6791', 'N2158']
cl['type'] = np.where(cl.member.isin(OPEN), 'open', 'globular')
print(cl.groupby('member').size().sort_values(ascending=False).to_string())

# %% [markdown]
# ## 7.1 Kiel diagrams

# %%
order = cl.groupby('member').fe_h.median().sort_values(ascending=False).index
fig, ax = plt.subplots(3, 6, figsize=(17, 9), sharex=True, sharey=True)
for a, m in zip(ax.flat, order):
    d = cl[cl.member == m]
    a.plot(d.teff, d.logg, 'o', mfc='none', color='0.5', ms=4, label='ASPCAP')
    a.plot(d.teff_c, d.logg_c, '.', color='C3', ms=5, label='Cannon')
    a.set_title(f'{m} ({d.type.iloc[0]}), [Fe/H] ≈ {d.fe_h.median():.2f}', fontsize=9)
for a in ax.flat[len(order):]:
    a.axis('off')
ax[0, 0].set(xlim=(5400, 3700), ylim=(4, 0))
ax[0, 0].legend(fontsize=7)
for a in ax[:, 0]:
    a.set_ylabel(r'$\log g$')
for a in ax[-1]:
    a.set_xlabel(r'$T_{\rm eff}$ [K]')
plt.tight_layout()

# %% [markdown]
# **Figure 7.1.** Kiel diagrams of the 17 clusters, ordered by metallicity (title: cluster, type, median ASPCAP
# [Fe/H]): ASPCAP labels (open grey circles) and The Cannon's (red dots) for the same stars. For the open
# clusters and the metal-rich globulars the two agree and trace narrow giant branches (and red clumps). For the
# most metal-poor globulars (bottom row: M53, M92, M15, [Fe/H] < −1.5, outside the training range) The Cannon's
# giants are displaced to higher gravity, by ≈ 0.7 dex in $\log g$ at the same $T_{\rm eff}$ — Chapter 5's
# extrapolation problem (the metal-poor field giants showed the same). In M13 the shift has started already.
#
# ## 7.2 Mean metallicities across clusters
#
# The mean [Fe/H] of each cluster from The Cannon against that from ASPCAP. Agreement for one cluster is easy (even
# a model that returned the training mean for every star would agree at [Fe/H] ≈ −0.3); agreement over a range of
# clusters, i.e. a slope of 1, is a real test.

# %%
g = cl.groupby('member')
tab = pd.DataFrame({'n': g.size(), 'type': g.type.first(),
                    'feh_aspcap': g.fe_h.mean(), 'feh_cannon': g.fe_h_c.mean(),
                    'std_aspcap': g.fe_h.std(), 'std_cannon': g.fe_h_c.std()})
tab['err_cannon'] = tab.std_cannon / np.sqrt(tab.n)
inside = tab.feh_aspcap > L[:, 2].min()                     # cluster mean inside the training range of [Fe/H]
print(tab.round(3).sort_values('feh_aspcap', ascending=False).to_string())
p = np.polyfit(tab.feh_aspcap[inside], tab.feh_cannon[inside], 1, w=1 / tab.err_cannon[inside].clip(0.01))
fig, ax = plt.subplots(figsize=(6.5, 5))
for t, mk in [('open', 'o'), ('globular', 's')]:
    d = tab[(tab.type == t)]
    ax.errorbar(d.feh_aspcap, d.feh_cannon, d.err_cannon, fmt=mk, mfc='none', label=f'{t} clusters', capsize=2)
xx = np.array([-2.4, 0.5])
ax.plot(xx, xx, 'k-', lw=0.8, label='identity')
ax.plot(xx, np.polyval(p, xx), 'C3--', lw=1, label=f'fit inside training range: slope {p[0]:.2f}')
ax.axvspan(-2.6, L[:, 2].min(), color='0.9', zorder=-1)
ax.set(xlabel='mean ASPCAP [Fe/H]', ylabel='mean Cannon [Fe/H]', xlim=(-2.5, 0.5), ylim=(-2.5, 0.5))
ax.legend(fontsize=8)
plt.tight_layout()
print(f'slope {p[0]:.3f}, offset at [Fe/H] = 0: {p[1]:+.3f}')
# Test: inside the training range The Cannon reproduces the differences between clusters (slope 0.95-1.05).
assert 0.95 < p[0] < 1.05

# %% [markdown]
# **Figure 7.2.** Mean [Fe/H] of each cluster from The Cannon against ASPCAP, with the standard error of the
# Cannon mean; circles: open clusters, squares: globular clusters. The dashed line is a weighted fit to the
# clusters inside the training range of [Fe/H] (white); the grey band is below it. Inside the range the points
# follow the identity (slope 0.98); below it the three most metal-poor globular clusters (M53, M92, M15, ASPCAP
# [Fe/H] ≈ −1.9 to −2.3) all come out at ≈ −1.5, the edge of the training set, as the metal-poor field giants of
# Chapter 5 did.
#
# ## 7.3 Spread within clusters
#
# If the stars of an open cluster have the same [Fe/H], the spread of their measured [Fe/H] is the measurement
# error. Compare the standard deviation within each cluster for the two methods, and look for a trend with
# temperature, which would betray a systematic error that depends on the stellar type.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for t, mk in [('open', 'o'), ('globular', 's')]:
    d = tab[(tab.type == t) & inside]
    ax[0].plot(d.std_aspcap, d.std_cannon, mk, mfc='none', label=f'{t} (inside training range)')
ax[0].plot([0, 0.15], [0, 0.15], 'k-', lw=0.8)
ax[0].set(xlabel='std of ASPCAP [Fe/H] within cluster', ylabel='std of Cannon [Fe/H] within cluster')
ax[0].legend(fontsize=8)
oc = cl[cl.type == 'open'].copy()
oc['dfe'] = oc.fe_h_c - oc.groupby('member').fe_h_c.transform('mean')   # deviation from the cluster mean
oc['dfe_a'] = oc.fe_h - oc.groupby('member').fe_h.transform('mean')
ax[1].plot(oc.teff_c, oc.dfe, '.', color='C3', label='Cannon')
ax[1].plot(oc.teff, oc.dfe_a, 'o', mfc='none', color='0.5', ms=4, label='ASPCAP')
# temperatures are centred within each cluster too, so that only within-cluster trends enter the slope
slope_c = np.polyfit(oc.teff_c - oc.groupby('member').teff_c.transform('mean'), oc.dfe, 1)[0] * 1000
slope_a = np.polyfit(oc.teff - oc.groupby('member').teff.transform('mean'), oc.dfe_a, 1)[0] * 1000
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set(xlabel=r'$T_{\rm eff}$ [K]', ylabel='[Fe/H] − cluster mean [dex]',
          title=f'open clusters; slopes per 1000 K: Cannon {slope_c:+.3f}, ASPCAP {slope_a:+.3f}')
ax[1].legend(fontsize=8)
plt.tight_layout()
oc_std = tab[tab.type == 'open'].std_cannon
n_wider = (tab[tab.type == 'open'].std_cannon > tab[tab.type == 'open'].std_aspcap).sum()
print(f'open clusters: median within-cluster std of [Fe/H]: Cannon {oc_std.median():.3f}, ASPCAP {tab[tab.type == "open"].std_aspcap.median():.3f} dex; '
      f'Cannon wider than ASPCAP in {n_wider} of {(tab.type == "open").sum()}')
# Test: open clusters are homogeneous in Cannon [Fe/H] to better than 0.05 dex (median std), comparable to ASPCAP.
assert oc_std.median() < 0.05 and oc_std.median() < 1.5 * tab[tab.type == 'open'].std_aspcap.median()

# %% [markdown]
# **Figure 7.3.** Left: standard deviation of [Fe/H] within each cluster, The Cannon against ASPCAP (clusters
# inside the training range; the line is equality). Right: deviation of each open-cluster star from its cluster's
# mean [Fe/H] against $T_{\rm eff}$, for The Cannon (red) and ASPCAP (grey), with the slopes of straight-line fits.
# Typical open-cluster spreads are similar in the two methods (median 0.036 dex for The Cannon, 0.034 dex for
# ASPCAP), but cluster by cluster The Cannon is *wider* in five of the seven (M67 and NGC 7789 by a factor 1.6–1.8;
# the star at −0.34 dex is probably a non-member or a bad spectrum). These spreads are upper limits on the
# precision: the clusters may have a small intrinsic spread and the membership is not perfect. For comparison, Ness
# et al. 2018 (Table 2) found [Fe/H] spreads of 0.01–0.08 dex in the same clusters with a 23-label Cannon model, and
# Bovy (2016) and Poovelil et al. (2020) found APOGEE open clusters homogeneous in [Fe/H] to a few hundredths of a dex.
# In the globular clusters The Cannon's spread is mostly *smaller* than ASPCAP's — which is not good news for
# M92 and M15: their stars are tightly grouped at a mean that is ≈ 0.7 dex too high. A narrow distribution
# says nothing about accuracy, and near the edge of the training range (M3, M2, M13) the flattening of Chapter 5
# compresses the labels. Neither method shows a strong trend of [Fe/H] with $T_{\rm eff}$ within the open
# clusters (slopes per 1000 K in the title, with temperatures centred within each cluster).
#
# ## Summary
#
# - Clusters test labels independently of the training set: the cluster means must follow ASPCAP across clusters,
#   and the members of each open cluster must agree with each other.
# - Inside the training range both tests pass; below it (metal-poor globulars) The Cannon is biased.
#
# **Exercises.**
# 1. Add the metal-poor globular clusters to the training set (with the cluster mean [Fe/H] as label for all their
#    members, as N15 did). Do the field metal-poor giants of Chapter 5 improve?
# 2. Compute the intrinsic spread of [Fe/H] in M67 by maximum likelihood, using the single-visit precision of
#    Chapter 6 (scaled to each star's S/N) as the measurement error.

# %% [markdown]
# ## References
#
# - Bovy 2016, *The Chemical Homogeneity of Open Clusters*, ApJ 817, 49. [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...817...49B)
# - Donor et al. 2020, *The Open Cluster Chemical Abundances and Mapping Survey. IV. Abundances for 128 Open Clusters Using SDSS/APOGEE DR16*, AJ 159, 199. [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....159..199D)
# - Myers et al. 2022, *The Open Cluster Chemical Abundances and Mapping Survey. VI. Galactic Chemical Gradient Analysis from APOGEE DR17*, AJ 164, 85. [ADS](https://ui.adsabs.harvard.edu/abs/2022AJ....164...85M)
# - Mészáros et al. 2013, *Calibrations of Atmospheric Parameters Obtained from the First Year of SDSS-III APOGEE Observations*, AJ 146, 133. [ADS](https://ui.adsabs.harvard.edu/abs/2013AJ....146..133M)
# - Mészáros et al. 2020, *Homogeneous analysis of globular clusters from the APOGEE survey with the BACCHUS code - II. The Southern clusters and overview*, MNRAS 492, 1641. [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1641M)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
# - Poovelil et al. 2020, *Open Cluster Chemical Homogeneity throughout the Milky Way*, ApJ 903, 55. [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...903...55P)
# - Ting et al. 2019, *The Payne: Self-consistent ab initio Fitting of Stellar Spectra*, ApJ 879, 69. [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T)
