# %% [markdown]
# # Chapter 9 — Label transfer to Gaia RVS spectra
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/09_gaia_rvs.ipynb)
#
# **Goal.** The Cannon does not care which instrument took the spectra: it only needs reference stars whose labels
# are known and whose spectra are taken with the *same* instrument as the stars to be labelled. Ho et al. 2017
# ([ApJ 836, 5](https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H)) used this to put 450 000 low-resolution
# LAMOST spectra on the APOGEE label scale, training on stars observed by both surveys — **label transfer**. Here we
# do the same with the Gaia DR3 RVS spectra of APOGEE giants (as Das et al. 2025,
# [MNRAS 538, 605](https://ui.adsabs.harvard.edu/abs/2025MNRAS.538..605D), did with GALAH labels), and compare the
# result with Gaia's own spectroscopic parameters (GSP-Spec; Recio-Blanco et al. 2023,
# [A&A 674, A29](https://ui.adsabs.harvard.edu/abs/2023A&A...674A..29R)).
#
# **You will learn**
# 1. what a Gaia RVS spectrum is (846–870 nm, $R \approx 11\,500$, the Ca II triplet);
# 2. to train a model for a new instrument with labels from another survey, and cross-validate it;
# 3. how precision depends on the S/N of the new spectra;
# 4. why two pipelines on two instruments disagree, and how label transfer puts them on one scale.

# %%
#@@LOADER
#@@CANNON
import time

rvs = load('rvs_apogee')
lab = load('labels_rvs')
assert (rvs['apogee_id'] == lab.apogee_id.to_numpy()).all()
wl = rvs['wavelength_nm']                                          # 846-870 nm, 0.01 nm per pixel
flux = rvs['flux'].astype(float)
err = np.where(np.isfinite(flux) & np.isfinite(rvs['err']) & (rvs['err'] > 0), rvs['err'], np.inf).astype(float)
flux = np.nan_to_num(flux)
CAT = [850.04, 854.44, 866.45]                                     # Ca II triplet, vacuum wavelengths [nm]
NAMES = ['teff', 'logg', 'fe_h', 'mg_fe']
TEX = [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]', '[Mg/Fe]']
UNITS = ['K', 'dex', 'dex', 'dex']
L = lab[NAMES].to_numpy()
good = np.all(np.isfinite(L), axis=1)
print(f'{len(lab)} APOGEE giants with a Gaia RVS spectrum; {good.sum()} with all four labels; '
      f'RVS S/N: median {lab.rvs_spec_sig_to_noise.median():.0f}, 10-90%: '
      f'{lab.rvs_spec_sig_to_noise.quantile(0.1):.0f}-{lab.rvs_spec_sig_to_noise.quantile(0.9):.0f}')

# %% [markdown]
# ## 9.1 Gaia RVS spectra
#
# The Radial Velocity Spectrometer on Gaia records 846–870 nm at a resolving power $R \approx 11\,500$, half that of
# APOGEE, with 2401 pixels of 0.01 nm (Cropper et al. 2018, [A&A 616, A5](https://ui.adsabs.harvard.edu/abs/2018A&A...616A...5C)).
# Gaia DR3 published ~1 million mean spectra, already normalised to a pseudo-continuum by the Gaia RVS processing
# (Gaia DR3 documentation, chapter on spectroscopic processing; GSP-Spec re-normalises them internally, Recio-Blanco
# et al. 2023 §3). At first we use that normalisation as it is: if it is the same for training and test stars, that
# is all The Cannon needs (§9.3 tests whether it is). The window is dominated
# by the near-infrared **Ca II triplet** (850.04, 854.44 and 866.45 nm in vacuum; 849.80, 854.21, 866.21 nm in air), whose broad wings are sensitive to
# gravity and metallicity, plus weaker lines of Fe I, Si I, Ti I, Mg I and CN.

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
for a, (nm, k, sel) in zip(ax, [('[Fe/H]', 2, (np.abs(L[:, 0] - 4700) < 100) & (np.abs(L[:, 1] - 2.5) < 0.3)),
                              (r'$T_{\rm eff}$', 0, (np.abs(L[:, 2] + 0.2) < 0.1) & (np.abs(L[:, 1] - 2.5) < 0.4))]):
    idx = np.flatnonzero(sel & good & (lab.rvs_spec_sig_to_noise > 80))
    idx = idx[np.argsort(L[idx, k])][::max(1, len(idx) // 8)]
    cm = plt.get_cmap('viridis' if k == 2 else 'plasma')
    for j in idx:
        a.plot(wl, rvs['flux'][j], lw=0.7, color=cm((L[j, k] - L[idx, k].min()) / np.ptp(L[idx, k])))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(L[idx, k].min(), L[idx, k].max()))
    plt.colorbar(sm, ax=a, label=nm + ' (APOGEE)')
    for ca in CAT:
        a.axvline(ca, color='k', ls=':', lw=0.6)
    a.set(xlabel='wavelength [nm]', ylabel='normalised flux', ylim=(0.2, 1.15), title=f'{len(idx)} giants ordered by {nm}')
plt.tight_layout()

# %% [markdown]
# **Figure 9.1.** Gaia RVS spectra of APOGEE giants with RVS S/N > 80. Left: $T_{\rm eff} = 4700 \pm 100$ K,
# $\log g = 2.5 \pm 0.3$, coloured by APOGEE [Fe/H]; right: [Fe/H] $= -0.2 \pm 0.1$, $\log g = 2.5 \pm 0.4$,
# coloured by APOGEE $T_{\rm eff}$. Dotted lines: the Ca II triplet. With metallicity, the weak metal lines between
# the Ca II lines and the Ca II wings deepen; the temperature changes are subtler, at the 1–2% level.
#
# ## 9.2 Training on RVS spectra with APOGEE labels
#
# Same code as before, new data: four labels ($T_{\rm eff}$, $\log g$, [Fe/H], [Mg/Fe]; 15 coefficients per
# pixel), 2401 pixels. We cross-validate with five folds on all stars with the four labels. (Ho et al. 2017 also used
# the K-band extinction as a label for LAMOST, whose spectra are not normalised; the RVS spectra are, so we do not.)

# %%
def scaling(lab_):
    return np.median(lab_, axis=0), (np.percentile(lab_, 97.5, axis=0) - np.percentile(lab_, 2.5, axis=0)) / 2


idx = np.flatnonzero(good)
fold = np.random.default_rng(9).permutation(len(idx)) % 5
cv = np.full_like(L, np.nan)
chi = np.full(len(L), np.nan)
t0 = time.time()
for k in range(5):
    t, v = idx[fold != k], idx[fold == k]
    pv, sc = scaling(L[t])
    th, _, s, _ = train_clean((L[t] - pv) / sc, flux[t], err[t])
    out = [test(flux[n], err[n], th, s) for n in v]
    cv[v] = np.array([o[0] for o in out]) * sc + pv
    chi[v] = [o[2] / (o[3] - 4) for o in out]
print(f'5-fold cross-validation on {len(idx)} stars: {time.time() - t0:.0f} s')
d = cv - L
snr = lab.rvs_spec_sig_to_noise.to_numpy()
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for k in range(4):
    sc_ = ax[k].scatter(L[good, k], cv[good, k], c=np.log10(snr[good]), s=2, cmap='viridis', vmin=1.3, vmax=2.5)
    lim = np.nanpercentile(L[good, k], [0.5, 99.5])
    ax[k].plot(lim, lim, 'k-', lw=0.8)
    ax[k].set(xlabel='APOGEE ' + TEX[k], ylabel='Cannon on RVS ' + TEX[k],
              title=f'rms {np.nanstd(d[:, k]):.3g}, bias {np.nanmean(d[:, k]):+.2g}')
plt.colorbar(sc_, ax=ax[-1], label='log$_{10}$ RVS S/N')
plt.tight_layout()

# %% [markdown]
# **Figure 9.2.** Five-fold cross-validation of a four-label Cannon model trained on Gaia RVS spectra (as
# normalised by the Gaia pipeline) with APOGEE labels: labels inferred from the RVS spectrum alone against the
# APOGEE labels of the same stars, coloured by RVS S/N (log scale); titles give the rms and the bias. $T_{\rm eff}$
# transfers well. In $\log g$ and [Fe/H] there is a second, displaced branch of low-S/N stars (dark points) whose
# [Fe/H] is too low by 0.5–1 dex. They are not failures of the optimiser: restarting the test step at the APOGEE
# labels converges to the same wrong answer, so these spectra really look metal-poor to the model.
#
# ## 9.3 A normalisation problem again
#
# A spectrum looks metal-poor when its lines look shallow — which also happens when its continuum is placed too
# *low*. The Gaia pipeline normalises each RVS spectrum itself; if that normalisation depends on S/N (Chapter 3),
# the low-S/N spectra are normalised differently from the high-S/N ones the model mostly learned from. Test it:
# train on stars with RVS S/N > 60, predict every spectrum at its APOGEE labels, and measure, star by star, the mean
# offset (data − model) at pixels where the model spectrum is ≈ 1 (continuum).

# %%
snr = lab.rvs_spec_sig_to_noise.to_numpy()
hi = good & (snr > 60)
pv, sc = scaling(L[hi])
th_hi, _, s_hi, _ = train_clean((L[hi] - pv) / sc, flux[hi], err[hi])
model = vectorize((L - pv) / sc) @ th_hi.T                           # every star's spectrum predicted at its APOGEE labels
d1 = th_hi[:, 1:5] / sc                                             # derivatives per K and per dex
cont = (np.abs(th_hi[:, 0] - 1) < 0.01) & (np.abs(d1[:, 0]) < 1e-5) & np.all(np.abs(d1[:, 1:3]) < 0.01, axis=1)
offset = np.array([np.mean((flux[n] - model[n])[cont & np.isfinite(err[n])]) for n in range(len(L))])
print(f'{cont.sum()} continuum pixels (|theta_0 - 1| < 0.01, |df/dT| < 1e-5 per K, |df/dlog g|, |df/d[Fe/H]| < 0.01 per dex)')
sbins = np.array([10, 20, 30, 45, 70, 1000])
sb = np.digitize(snr, sbins)
spread = [offset[good & (sb == i)].std() for i in range(1, len(sbins))]
print('std of the per-star continuum offset in RVS S/N bins', dict(zip([f'{a}-{b}' for a, b in zip(sbins[:-1], sbins[1:])], np.round(spread, 4))))
# expected from noise alone: the error of an (unweighted) mean of m independent pixels is sqrt(sum sigma_i^2) / m
noise = np.array([np.sqrt(np.sum(err[n][cont & np.isfinite(err[n])]**2)) / np.sum(cont & np.isfinite(err[n]))
                  for n in range(len(L))])
nb_ = np.digitize(snr, sbins)
env = np.array([np.median(noise[good & (nb_ == i)]) for i in range(1, len(sbins))])      # median noise per S/N bin
xmid = np.sqrt(sbins[:-1] * np.r_[sbins[1:-1], 300])         # geometric bin centres (last bin capped at 300)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].scatter(snr[good], offset[good] * 100, s=2, c='0.4')
ax[0].step(xmid, 3 * env * 100, 'C3-', where='mid', lw=1, label='±3× noise of the mean (median per S/N bin)')
ax[0].step(xmid, -3 * env * 100, 'C3-', where='mid', lw=1)
ax[0].set(xscale='log', xlabel='RVS S/N', ylabel='continuum offset, data − model [%]', ylim=(-12, 8))
ax[0].legend(fontsize=8)
dfe = cv[:, 2] - L[:, 2]
lowsn = good & (snr < 30)
ax[1].scatter(offset[lowsn] * 100, dfe[lowsn], s=3)
ax[1].set(xlabel='continuum offset [%]', ylabel='Cannon − APOGEE [Fe/H] (cross-validated)', title='RVS S/N < 30')
plt.tight_layout()
from scipy.stats import spearmanr
rho = spearmanr(offset[lowsn], dfe[lowsn])[0]
G = lab.gaiaedr3_phot_g_mean_mag.to_numpy()
for g0 in [11.5, 12, 12.5, 13, 13.5]:
    bb = good & (G >= g0) & (G < g0 + 0.5)
    print(f'G {g0}-{g0 + 0.5}: {bb.sum():4d} stars, fraction with continuum offset > +1%: {np.mean(offset[bb] > 0.01):.2f}')
print(f'S/N < 30: Spearman correlation of continuum offset and [Fe/H] error = {rho:.2f}')
# Tests: at S/N < 30 the continuum offsets scatter far more than at S/N > 70 (5x), and they drive the [Fe/H] errors:
# a positive offset (continuum placed too low by the pipeline, lines look shallow) goes with too low an [Fe/H].
assert spread[1] > 5 * spread[-1] and rho < -0.5

# %% [markdown]
# **Figure 9.3.** Left: mean offset between each star's Gaia-normalised RVS spectrum and the model spectrum at its
# APOGEE labels, at the continuum pixels of the model (grey dots), against RVS S/N; the red lines are ±3 times the
# offset expected from the noise of the continuum pixels alone. Above S/N ≈ 60 the offsets are within the noise.
# Below, they are not: most spectra drift to an offset of ≈ −1.5% (continuum placed too high, lines look too deep),
# and a separate group sits at ≈ +2.5% (continuum placed too low, lines look too shallow). The second group consists
# of faint stars only (printed above: none at G < 12.5, about half at G > 13); we have not traced its origin in the
# Gaia processing. Right: at S/N < 30, the cross-validated [Fe/H] error against this offset. The two groups are
# biased in opposite directions — too metal-rich by ≈ 0.2 dex and too metal-poor by ≈ 0.5 dex. The Gaia
# normalisation is not stable at low S/N — the problem of Chapter 3, in another survey.
#
# The remedy is that of Chapter 3: normalise all spectra ourselves, in the same way, using fixed continuum pixels.
# Fit a second-order Chebyshev polynomial to the continuum pixels of each Gaia-normalised spectrum
# (inverse-variance weights) and divide by it. Then cross-validate again, with the same folds. To keep the test
# honest, the continuum pixels are chosen *inside each fold*, from a model trained on the S/N > 60 stars of the
# training part only: the held-out stars play no role in choosing them.

# %%
B = np.polynomial.chebyshev.chebvander((wl - 858) / 12, 2)           # (2401, 3) basis; wavelength mapped to [-1, 1]


def continuum_pixels(rows):
    """RVS continuum pixels from a model trained on the stars `rows` (as for Figure 9.3)."""
    pv_, sc_ = scaling(L[rows])
    th_, _, _, _ = train_clean((L[rows] - pv_) / sc_, flux[rows], err[rows])
    d_ = th_[:, 1:5] / sc_
    return (np.abs(th_[:, 0] - 1) < 0.01) & (np.abs(d_[:, 0]) < 1e-5) & np.all(np.abs(d_[:, 1:3]) < 0.01, axis=1)


def renormalise(cp):
    """Divide every spectrum by a 2nd-order Chebyshev polynomial fitted to its continuum pixels cp."""
    w_ = 1 / err[:, cp]**2
    coef = np.linalg.solve(np.einsum('pi,np,pj->nij', B[cp], w_, B[cp]),
                           np.einsum('pi,np->ni', B[cp], w_ * flux[:, cp])[:, :, None])[:, :, 0]
    renorm = coef @ B.T                                               # the correction to Gaia's continuum, per star
    return flux / renorm, err / renorm


cv2 = np.full_like(L, np.nan)
for k in range(5):
    t, v = idx[fold != k], idx[fold == k]
    flux2, err2 = renormalise(continuum_pixels(t[snr[t] > 60]))      # pixels chosen from training stars only
    pv, sc = scaling(L[t])
    th, _, s, _ = train_clean((L[t] - pv) / sc, flux2[t], err2[t])
    cv2[v] = np.array([test(flux2[n], err2[n], th, s)[0] for n in v]) * sc + pv
flux2, err2 = renormalise(cont)                                       # all stars, for Figure 9.6
d2 = cv2 - L
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for k in range(4):
    sc_ = ax[k].scatter(L[good, k], cv2[good, k], c=np.log10(snr[good]), s=2, cmap='viridis', vmin=1.3, vmax=2.5)
    lim = np.nanpercentile(L[good, k], [0.5, 99.5])
    ax[k].plot(lim, lim, 'k-', lw=0.8)
    ax[k].set(xlabel='APOGEE ' + TEX[k], ylabel='Cannon on renormalised RVS ' + TEX[k],
              title=f'rms {np.nanstd(d2[:, k]):.3g} (was {np.nanstd(d[:, k]):.3g})')
plt.colorbar(sc_, ax=ax[-1], label='log$_{10}$ RVS S/N')
plt.tight_layout()
n_out = [np.sum(good & (x[:, 2] - L[:, 2] < -0.4)) for x in (cv, cv2)]
for k in range(4):
    print(f'{NAMES[k]:5s}: rms Gaia normalisation {np.nanstd(d[good, k]):.3f}, renormalised {np.nanstd(d2[good, k]):.3f}; '
          f'at S/N < 30: {np.nanstd(d[lowsn, k]):.3f} -> {np.nanstd(d2[lowsn, k]):.3f}')
print(f'stars with [Fe/H] error < -0.4 dex: {n_out[0]} -> {n_out[1]}')
# Test: renormalising on fixed continuum pixels improves log g and [Fe/H] at S/N < 30 by > 30% and removes most
# of the metal-poor branch.
assert all(np.nanstd(d2[lowsn, k]) < 0.7 * np.nanstd(d[lowsn, k]) for k in (1, 2)) and n_out[1] < 0.3 * n_out[0]

# %% [markdown]
# **Figure 9.4.** As Figure 9.2, after renormalising every RVS spectrum on the model's continuum pixels (titles:
# new rms, and in brackets the rms with Gaia's normalisation). The metal-poor branch has almost disappeared, and
# the $\log g$ and [Fe/H] errors drop by a large factor at low S/N. (The continuum pixels of each fold were chosen
# from that fold's training stars only.)
#
# ## 9.4 Comparison with Gaia's own parameters, and with the literature
#
# How do the transferred labels compare with the parameters that the Gaia pipeline derived from the *same* RVS
# spectra by fitting synthetic spectra (GSP-Spec; Recio-Blanco et al. 2023, RB23)? A fair comparison needs three
# things RB23 recommend: their quality flags (we keep the "best quality" stars whose first 13 characters of
# `flags_gspspec` are all 0), their calibration of $\log g$ and [M/H] as polynomials in $\log g$ (RB23 eqs. 1–2 and
# Table 3), and a statistic that is not dominated by a few outliers. We report both the standard deviation and the
# **robust** standard deviation $1.4826 \times {\rm MAD}$ (median absolute deviation), which equals the standard
# deviation for a Gaussian. GSP-Spec's [M/H] is a global metallicity; we compare it with APOGEE [Fe/H].

# %%
rstd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))       # robust standard deviation
gsp = lab[['teff_gspspec', 'logg_gspspec', 'mh_gspspec']].to_numpy()
lg = gsp[:, 1]
gsp_cal = gsp.copy()                                                 # RB23 Table 3 calibrations, polynomials in raw log g
gsp_cal[:, 1] = lg + 0.4496 - 0.0036 * lg - 0.0224 * lg**2
gsp_cal[:, 2] = gsp[:, 2] + 0.274 - 0.1373 * lg - 0.0050 * lg**2 + 0.0048 * lg**3
best = lab.flags_gspspec.fillna('9' * 41).astype(str).str[:13] == '0' * 13
has_gsp = np.all(np.isfinite(gsp), axis=1) & good
rows = []
for nm, sel, g_ in [('all stars with GSP-Spec, raw', has_gsp, gsp), ('best quality, calibrated', has_gsp & best, gsp_cal)]:
    for k in range(3):
        dc, dg = d2[sel, k], g_[sel, k] - L[sel, k]
        rows.append((nm, NAMES[k], sel.sum(), np.median(snr[sel]), dc.std(), rstd(dc), dg.std(), rstd(dg), np.median(dg)))
tab = pd.DataFrame(rows, columns=['sample', 'label', 'stars', 'median S/N', 'Cannon std', 'Cannon robust',
                                  'GSP-Spec std', 'GSP-Spec robust', 'GSP-Spec median offset'])
print(tab.round(3).to_string(index=False))
sel = has_gsp & best
bins = np.array([25, 40, 60, 100, 160, 300])
mid = np.sqrt(bins[1:] * bins[:-1])
b = np.digitize(snr, bins)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    ax[k].plot(mid, [rstd(d2[sel & (b == i), k]) for i in range(1, len(bins))], 'o-', label='The Cannon (APOGEE labels transferred)')
    ax[k].plot(mid, [rstd(gsp_cal[sel & (b == i), k] - L[sel & (b == i), k]) for i in range(1, len(bins))], 's-',
               label='GSP-Spec, calibrated')
    ax[k].set(xscale='log', xlabel='RVS S/N', ylabel=f'robust std of difference from APOGEE {TEX[k]} [{UNITS[k]}]')
    ax[k].legend(fontsize=8)
plt.tight_layout()
t = tab[tab['sample'] == 'best quality, calibrated']
# Test: on the same best-quality stars, with GSP-Spec calibrated and a robust statistic, the transferred labels are
# closer to APOGEE than GSP-Spec for all three labels (measured: 48 vs 70 K, 0.13 vs 0.17 dex, 0.067 vs 0.081 dex).
assert np.all(t['Cannon robust'].to_numpy() < t['GSP-Spec robust'].to_numpy())

# %% [markdown]
# **Figure 9.5.** Robust standard deviation of the difference from APOGEE, in bins of RVS S/N, for the stars with
# GSP-Spec best-quality flags: labels transferred by The Cannon (circles; cross-validated) and calibrated GSP-Spec
# parameters from the same spectra (squares). The table above gives both statistics for all stars and for this
# sample. Three lessons:
#
# - **How you compare matters.** Raw GSP-Spec values for all stars look twice as bad as The Cannon's; with RB23's
#   flags, calibration and a robust statistic the factor shrinks to 1.2–1.5 overall. The advantage of The Cannon is
#   at low S/N and in $T_{\rm eff}$; above S/N ≈ 100, calibrated GSP-Spec matches or beats it in $\log g$ and [Fe/H]
#   (with only a few hundred stars there). Our calibrated GSP-Spec numbers
#   (≈ 70 K, 0.17 dex, 0.08 dex) are consistent with what RB23 found for their best-quality sample against the
#   literature (robust σ of 90 K, 0.19 dex and 0.13 dex, with a raw $\log g$ offset of −0.3 dex; RB23 §9.1).
# - **Part of the advantage is the label scale.** The Cannon learns the APOGEE scale, including APOGEE's own
#   calibrations; GSP-Spec has its own. Putting surveys on one scale is what label transfer is for (Ho et al. 2017);
#   which scale is *right* is a different question, answered by clusters, asteroseismology and benchmark stars.
# - **Comparison with published data-driven RVS labels.** Das et al. (2025) trained The Cannon on Gaia RVS spectra
#   with GALAH DR4 labels (14 484 training stars, S/N > 50; 8 labels) and report, for giants, cross-validated
#   scatters of 70 K, 0.14 dex and 0.07 dex. Our best-quality sample (median S/N 64) reaches 55 K, 0.16 dex and
#   0.08 dex in standard deviation: consistent, with 4000 instead of ~14 500 training stars.
#
# ## 9.5 Where is the information in an RVS spectrum?

# %%
pv, sc = scaling(L[good])
th, _, s, _ = train_clean((L[good] - pv) / sc, flux2[good], err2[good])
dfdl = th[:, 1:5] / sc
fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
ax[0].plot(wl, th[:, 0], 'k', lw=0.8, label=r'$\theta_0$')
ax[0].fill_between(wl, th[:, 0] - s, th[:, 0] + s, color='C3', alpha=0.4, label=r'$\pm s_\lambda$')
ax[0].set(ylabel='normalised flux')
ax[0].legend(fontsize=8)
for k, fac in zip(range(4), [1000, 1, 1, 1]):
    ax[1].plot(wl, dfdl[:, k] * fac, lw=0.8, label=r'$\partial f/\partial$' + TEX[k] + (' (per 1000 K)' if k == 0 else ' (per dex)'))
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set(xlabel='wavelength [nm]', ylabel='derivative', ylim=(-0.3, 0.3))
ax[1].legend(fontsize=8, ncol=4)
plt.tight_layout()
wing = (np.abs(wl - CAT[1]) > 0.1) & (np.abs(wl - CAT[1]) < 0.6)          # wings of the strongest Ca II line
print(f'mean d flux / d log g in the 854.44 nm wings: {dfdl[wing, 1].mean():+.3f} per dex')

# %% [markdown]
# **Figure 9.6.** The four-label model trained on all renormalised RVS spectra. Top: baseline spectrum with
# $\pm s_\lambda$. Bottom: derivatives per 1000 K and per dex. The Ca II wings get *shallower* with increasing
# $\log g$ (positive derivative): the Ca II triplet is a classical luminosity indicator, stronger in low-gravity
# giants and supergiants, whose extended, low-density atmospheres put more Ca II above the continuum-forming layers.
# The wings also deepen with [Fe/H], and the many weak lines between them carry most of the [Fe/H] and
# $T_{\rm eff}$ information. The [Mg/Fe] derivative is weak everywhere, consistent with its poor transfer.
#
# ## Summary
#
# - Label transfer needs only stars observed by both surveys: The Cannon trained on Gaia RVS spectra with APOGEE
#   labels puts RVS spectra on the APOGEE label scale.
# - Another pipeline's normalisation is not to be trusted blindly: Gaia's RVS normalisation varies at low S/N and
#   produced a branch of spurious metal-poor stars; renormalising on fixed continuum pixels (Chapter 3) removed it.
# - $T_{\rm eff}$, $\log g$ and [Fe/H] transfer with a precision that degrades with RVS S/N; [Mg/Fe] barely
#   transfers from this window.
# - On the same stars, with GSP-Spec's own quality flags and calibration and a robust statistic, the transferred
#   labels agree with APOGEE better than GSP-Spec by a factor 1.2–1.5 — partly a matter of label scale, which label
#   transfer removes by construction. The precision is consistent with Das et al. (2025).
#
# **Exercises.**
# 1. Train only on stars with RVS S/N > 100 and test on S/N < 40. Does the precision at low S/N change? (Chapter 2:
#    the noise model lets a high-S/N model label low-S/N spectra.)
# 2. Use the transferred model on Gaia RVS spectra of stars *without* APOGEE spectra (query
#    `gaia_dr3.rvs_mean_spectrum` in the Gaia archive). Which stars would be outside the training set?

# %% [markdown]
# ## References
#
# - Cropper et al. 2018, *Gaia Data Release 2. Gaia Radial Velocity Spectrometer*, A&A 616, A5. [ADS](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A...5C)
# - Das et al. 2025, *The GALAH Survey: Stellar parameters and abundances for 800 000 Gaia RVS spectra using GALAH DR4 and The Cannon*, MNRAS 538, 605. [ADS](https://ui.adsabs.harvard.edu/abs/2025MNRAS.538..605D)
# - Gaia Collaboration et al. 2023, *Gaia Data Release 3. Summary of the content and survey properties*, A&A 674, A1. [ADS](https://ui.adsabs.harvard.edu/abs/2023A%26A...674A...1G)
# - Ho et al. 2017, *Label Transfer from APOGEE to LAMOST: Precise Stellar Parameters for 450,000 LAMOST Giants*, ApJ 836, 5. [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H)
# - Ho et al. 2017, *Masses and Ages for 230,000 LAMOST Giants, via Their Carbon and Nitrogen Abundances*, ApJ 841, 40. [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...841...40H)
# - Katz et al. 2023, *Gaia Data Release 3. Properties and validation of the radial velocities*, A&A 674, A5. [ADS](https://ui.adsabs.harvard.edu/abs/2023A%26A...674A...5K)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Recio-Blanco et al. 2023, *Gaia Data Release 3. Analysis of RVS spectra using the General Stellar Parametriser from spectroscopy*, A&A 674, A29. [ADS](https://ui.adsabs.harvard.edu/abs/2023A%26A...674A..29R)
