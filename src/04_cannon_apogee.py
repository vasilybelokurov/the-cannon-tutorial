# %% [markdown]
# # Chapter 4 — Training The Cannon on APOGEE spectra
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/04_cannon_apogee.ipynb)
#
# **Goal.** Train the model of Chapter 2 on real spectra, normalised as in Chapter 3, with three labels
# ($T_{\rm eff}$, $\log g$, [Fe/H]) as in Ness et al. 2015 (N15), and look inside it: what the coefficients and the
# scatter say about the spectrum, how well the model fits stars it was not trained on, and what it does *not* fit.
#
# **You will learn**
# 1. to train on ~1700 real spectra × 8575 pixels in seconds, and to read the coefficients against a line list
#    (N15, Fig. 5);
# 2. that the intrinsic scatter $s_\lambda$ is not small on real data, and where it is large;
# 3. to check the fit of held-out stars with χ², with and without $s_\lambda$;
# 4. two diagnostics of things the labels do not describe: flagged pixels and the fibre (line-spread function).

# %%
#@@LOADER
#@@NORM
#@@CANNON
import time

labels = load('labels')
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
spec = load('spectra_train')
lines = load('linelist')
cont = load('continuum_pixels')['continuum'].to_numpy()       # continuum pixels of Chapter 3
flux, err = cheb_normalise(*prepare(spec), cont)               # normalised flux and error, err = inf where unusable
NAMES = ['teff', 'logg', 'fe_h']
TEX = [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]']
L = tr[NAMES].to_numpy()

# Hold out 20% of the stars (chosen at random) to test the model on spectra it has not seen.
rng = np.random.default_rng(4)
held = np.zeros(len(tr), bool)
held[rng.choice(len(tr), len(tr) // 5, replace=False)] = True
pivot = np.median(L[~held], axis=0)
scale = (np.percentile(L[~held], 97.5, axis=0) - np.percentile(L[~held], 2.5, axis=0)) / 2
to_x, to_lab = (lambda lab: (lab - pivot) / scale), (lambda x: x * scale + pivot)
t0 = time.time()
theta, cov, s = train(to_x(L[~held]), flux[~held], err[~held])
print(f'trained on {(~held).sum()} stars x {NPIX} pixels in {time.time() - t0:.1f} s; '
      f'pivot {pivot.round(2)}, scale {scale.round(2)}')

# %% [markdown]
# ## 4.1 Cleaning the training set
#
# The model has one scatter $s_\lambda$ per pixel, shared by all stars. If a few training stars are badly described
# — a fast rotator, an unresolved binary, a spectrum with an artefact, a star with a wrong reference label — their
# large residuals inflate $s_\lambda$ for everybody. A simple check is the mean squared residual of each training
# star in units of the model's variance, $\chi^2/\nu = \langle (f - \theta^{\sf T} v)^2/(\sigma^2 + s^2) \rangle$,
# which should be ≈ 1.

# %%
tr_idx = np.flatnonzero(~held)
z2 = (flux[tr_idx] - vectorize(to_x(L[tr_idx])) @ theta.T)**2 / (err[tr_idx]**2 + s**2)
chi_tr = np.nanmean(np.where(np.isfinite(err[tr_idx]) & (theta[:, 0] != 0), z2, np.nan), axis=1)
bad_tr = chi_tr >= 3
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(chi_tr, bins=np.logspace(-0.6, 1.8, 60))
ax[0].axvline(3, color='C3', ls='--', label=r'cut: $\chi^2/\nu = 3$')
ax[0].set(xscale='log', xlabel=r'$\chi^2/\nu$ of each training star (in-sample)', ylabel='stars')
ax[0].legend(fontsize=8)
ax[1].scatter(tr.aspcap_chi2[tr_idx], chi_tr, s=3, alpha=0.5)
ax[1].set(xscale='log', yscale='log', xlabel=r'ASPCAP $\chi^2$ of the same star', ylabel=r'Cannon $\chi^2/\nu$')
plt.tight_layout()
print(f'{bad_tr.sum()} of {len(tr_idx)} training stars have chi2/nu >= 3; the worst:')
print(tr.iloc[tr_idx[np.argsort(chi_tr)[-6:]]][['apogee_id', 'teff', 'logg', 'fe_h', 'snr', 'vsini', 'aspcap_chi2']]
      .assign(chi2nu=np.sort(chi_tr)[-6:]).round(2).to_string(index=False))
t_use = tr_idx[~bad_tr]                                             # the cleaned training set
theta, cov, s = train(to_x(L[t_use]), flux[t_use], err[t_use])
print(f'retrained on {len(t_use)} stars')

# %% [markdown]
# **Figure 4.1.** Left: $\chi^2/\nu$ of each training star under the model trained on all of them (log scale); most
# stars are below 1 — the shared scatter is larger than they need — and a tail of ≈ 2% extends to 10–50. Right: the
# same against the $\chi^2$ of ASPCAP's fit to that star: stars that ASPCAP also fits badly tend to be outliers here
# too. Among the worst are stars with measured rotation ($v \sin i \approx 25$ km/s, broadened lines) and stars with
# high ASPCAP $\chi^2$. We drop the training stars with $\chi^2/\nu \geq 3$ and retrain; from now on, the function
# `train_clean` does exactly this (train, cut, retrain). One pass is enough here: the tail is not fully removed, and
# residuals stay heavier-tailed than a Gaussian — a fact to keep in mind when we read $\chi^2$ values.
#
# ## 4.2 What the model learned
#
# As in Chapter 1, $\theta_{\lambda 0}$ is the spectrum of a star with the pivot labels and the first-order
# coefficients are derivatives. Dividing them by the label scales $q_k$ turns them into derivatives per K and per
# dex, which can be compared between labels.

# %%
on = np.isfinite(err).any(axis=0) & (theta[:, 0] != 0)          # pixels on the detectors
dfdl = theta[:, 1:4] / scale                                    # d flux / d label, per K, per dex, per dex
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
gap = lambda v: np.where(on, v, np.nan)                         # NaN: no line drawn across the detector gaps
ax[0].plot(wave, gap(theta[:, 0]), 'k', lw=0.3)
ax[0].set(ylabel=r'$\theta_0$', ylim=(0.4, 1.1))
for d, lab, fac in zip(range(3), TEX, [100, 1, 1]):
    ax[1].plot(wave, gap(dfdl[:, d]) * fac, lw=0.3, label=r'$\partial f/\partial$' + lab + (' (per 100 K)' if d == 0 else ' (per dex)'))
ax[1].set(ylabel='first-order coefficient', ylim=(-0.25, 0.25))
ax[1].legend(fontsize=8, ncol=3, loc='lower left')
ax[2].plot(wave, gap(s), 'C3', lw=0.3)
ax[2].set(xlabel='vacuum wavelength [Å]', ylabel=r'scatter $s_\lambda$', ylim=(0, 0.04))
plt.tight_layout()
print(f'median scatter {np.median(s[on]):.4f}; median error of the training spectra {np.median(err[t_use][:, on][np.isfinite(err[t_use][:, on])]):.4f}')

# %% [markdown]
# **Figure 4.2.** The trained three-label model over the whole APOGEE range. Top: the baseline spectrum
# $\theta_0$ (a giant with $T_{\rm eff}$, $\log g$, [Fe/H] at the pivot). Middle: the derivatives of the
# normalised flux with respect to $T_{\rm eff}$ (per 100 K), $\log g$ and [Fe/H] (per dex). Bottom: the intrinsic
# scatter $s_\lambda$. Every absorption feature has a non-zero derivative, usually with respect to more than one
# label; the [Fe/H] derivative (green) is the largest and negative almost everywhere. The scatter is comparable to
# or larger than the noise of the training spectra (S/N ≈ 300, i.e. ≈ 0.003) in most lines: three labels do not
# describe everything that changes a giant's spectrum.

# %%
fig, ax = plt.subplots(2, 2, figsize=(15, 7), sharey='row')
for j, (lo, hi) in enumerate([(15660, 15790), (16160, 16290)]):     # N15's regions A and B (their Fig. 3)
    z = on & (wave > lo) & (wave < hi)
    ax[0, j].plot(wave[z], theta[z, 0], 'k', lw=1)
    ax[0, j].fill_between(wave[z], theta[z, 0] - s[z], theta[z, 0] + s[z], color='C3', alpha=0.4, label=r'$\pm s_\lambda$')
    ax[0, j].plot(wave[z & cont], theta[z & cont, 0], 'o', color='C2', ms=3, label='continuum pixels')
    win = lines[(lines.wave_vac > lo) & (lines.wave_vac < hi)]
    win = win[win.strength > win.groupby('species').strength.transform('max') - 0.7]
    for k, (sp, d) in enumerate(win.groupby('species')):
        if len(d) and sp in ('FE I', 'MG I', 'SI I', 'TI I', 'K  I', 'MN I', 'NI I', 'AL I', 'CN', 'CO', 'OH'):
            ax[0, j].vlines(d.wave_vac, 1.06, 1.1, color=f'C{k % 10}', label=sp)
    for d, lab, fac in zip(range(3), TEX, [100, 1, 1]):
        ax[1, j].plot(wave[z], dfdl[z, d] * fac, lw=0.9, label=lab + (' (per 100 K)' if d == 0 else ' (per dex)'))
    ax[1, j].axhline(0, color='k', lw=0.4)
    ax[1, j].set(xlabel='vacuum wavelength [Å]')
    ax[0, j].legend(fontsize=7, ncol=4, loc='lower left')
ax[0, 0].set(ylabel=r'$\theta_0 \pm s$', ylim=(0.35, 1.12))
ax[1, 0].set(ylabel=r'$\partial f/\partial$ label')
ax[1, 0].legend(fontsize=8)
plt.tight_layout()

fe = lines[(lines.species == 'FE I')]
fe = fe[fe.strength > fe.strength.max() - 1.5]                      # the strong Fe I lines (within 1.5 dex of the strongest)
core = np.array([np.argmin(np.abs(wave - w)) for w in fe.wave_vac])
core = core[on[core] & (theta[core, 0] < 0.9)]                       # on a detector and visibly absorbed
frac_neg = np.mean(dfdl[core, 2] < 0)
print(f'{len(core)} strong Fe I line cores: fraction with d flux/d[Fe/H] < 0 = {frac_neg:.2f}')
# Test: the model has learned that iron lines deepen with [Fe/H], without being told where they are.
assert frac_neg >= 0.9

# %% [markdown]
# **Figure 4.3.** The model in the two 130 Å regions that N15 used in their Figures 3–6 (A, left; B, right).
# Top: baseline spectrum (black) with $\pm s_\lambda$ (red band), continuum pixels (green dots) and lines from the
# line list (ticks, by species). Bottom: derivatives per 100 K and per dex. The cores of strong lines carry most of
# the information: large derivatives — and the largest scatter. Lines of a given species do not respond to the
# labels in the same way: a high-excitation Fe I line weakens with temperature differently from an OH line,
# which is what lets the model separate $T_{\rm eff}$ from [Fe/H].
#
# ## 4.3 Stars the model has not seen
#
# Apply the test step (Chapter 2) to the 20% held-out stars. Their χ² per degree of freedom measures how well the
# model describes a new spectrum; computed **without** $s_\lambda$ (only the measurement errors) it measures how
# much the model misses.

# %%
t0 = time.time()
res = [test(flux[n], err[n], theta, s) for n in np.flatnonzero(held)]
print(f'test step: {time.time() - t0:.1f} s for {held.sum()} stars')
xh = np.array([r[0] for r in res])
chi2nu = np.array([r[2] / (r[3] - 3) for r in res])
mh = vectorize(xh) @ theta.T                                         # model spectra of the held-out stars
ok = np.isfinite(err[held]) & on
chi2nu_noise = np.array([np.sum(((flux[held][j] - mh[j]) / err[held][j])[ok[j]]**2) / (ok[j].sum() - 3) for j in range(held.sum())])
lab_h = to_lab(xh)
d = lab_h - L[held]
print('held-out stars, Cannon - ASPCAP: bias', d.mean(0).round(3), 'rms', d.std(0).round(3))
print(f'median chi2/nu: with s {np.median(chi2nu):.2f}, measurement errors only {np.median(chi2nu_noise):.2f}')

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for k in range(3):
    ax[k].plot(L[held, k], lab_h[:, k], '.', ms=3)
    lim = [L[:, k].min(), L[:, k].max()]
    ax[k].plot(lim, lim, 'k-', lw=0.8)
    ax[k].set(xlabel='ASPCAP ' + TEX[k], ylabel='Cannon ' + TEX[k], title=f'rms {d[:, k].std():.3g}, bias {d[:, k].mean():.2g}')
ax[3].hist(chi2nu, bins=np.linspace(0, 6, 61), histtype='step', label=r'with $s_\lambda$')
ax[3].hist(chi2nu_noise, bins=np.linspace(0, 6, 61), histtype='step', label='measurement errors only')
ax[3].set(xlabel=r'$\chi^2/\nu$ of held-out stars', ylabel='stars')
ax[3].legend(fontsize=8)
plt.tight_layout()
CHI2_BAND = (0.7, 1.1)                                                # frozen after a development run (measured 0.80)
assert CHI2_BAND[0] < np.median(chi2nu) < CHI2_BAND[1] and np.median(chi2nu_noise) > 1.5 * np.median(chi2nu)

# %% [markdown]
# **Figure 4.4.** Held-out stars (20% of the training set, not used in training). Left three panels: labels from
# The Cannon against ASPCAP's, with the identity line; titles give the rms and the mean difference. Right: their
# χ² per degree of freedom, with the model variance $\sigma^2 + s_\lambda^2$ (blue) and with the measurement
# variance $\sigma^2$ only (orange). With the scatter included the fits are statistically acceptable for most stars;
# without it, χ²/ν is several times larger: at S/N ≈ 300 the measurement noise is smaller than the imperfections
# of a three-label model. Chapter 5 turns the label comparison into a proper cross-validation.
#
# ## 4.4 Flagged pixels
#
# In Chapter 0 we gave zero weight to pixels with the bad-pixel and sky bits (0–7, 12). Were they really bad? The
# model now predicts every pixel, so we can look at the residuals of those pixels with their *original* errors.

# %%
# The continuum polynomial is the one fitted with the masked errors (as for the model); it is applied to the raw
# flux and raw errors, so that flagged pixels keep their values and their pipeline errors.
fm_raw, em_mask = np.nan_to_num(spec['flux'][held].astype(float)), prepare(spec, held)[1]
poly = fm_raw / cheb_normalise(fm_raw, em_mask, cont)[0]          # the fitted continuum at every pixel
with np.errstate(invalid='ignore', divide='ignore'):
    f_raw, e_raw = spec['flux'][held] / poly, spec['err'][held].astype(float) / poly
m = spec['mask'][held]
z = (f_raw - mh) / np.sqrt(e_raw**2 + s**2)                          # residual in units of the model's sigma
fin = np.isfinite(z) & on
groups = {'no bit set': m == 0, 'bad bits 0-7 (zero weight)': (m & 0xFF) > 0, 'sky line, bit 12 (zero weight)': ((m >> 12) & 1) > 0,
          'persistence, bits 9-11 (kept)': ((m >> 9) & 7) > 0, 'telluric, bit 13 (kept)': ((m >> 13) & 1) > 0}
fig, ax = plt.subplots(figsize=(7, 4.5))
out5 = {}
for nm, g in groups.items():
    v = z[g & fin]
    out5[nm] = np.mean(np.abs(v) > 5)
    ax.hist(v, bins=np.linspace(-10, 10, 101), density=True, histtype='step', label=f'{nm}: |r| > 5σ in {out5[nm]:.2%}')
ax.set(yscale='log', xlabel='(data − model) / σ_model', ylabel='density')
ax.legend(fontsize=7)
plt.tight_layout()
print({k: round(v, 4) for k, v in out5.items()})
print(f"sky-line pixels / clean pixels, fraction beyond 5 sigma: {out5['sky line, bit 12 (zero weight)'] / out5['no bit set']:.1f}")
# Test: sky-line pixels really are worse (outliers beyond 5 sigma at least twice as frequent as in clean pixels;
# measured ratio printed above, 3.4).
assert out5['sky line, bit 12 (zero weight)'] > 2 * out5['no bit set']

# %% [markdown]
# **Figure 4.5.** Residuals of the held-out stars, (data − model)/$\sqrt{\sigma^2 + s_\lambda^2}$, with the
# original pipeline errors, grouped by the mask bits of the pixel (log scale; the legend gives the fraction beyond
# ±5σ). Pixels near bright sky lines (bit 12) have heavy tails — the sky subtraction leaves residuals that the
# error does not describe — so giving them zero weight was right. Persistence and telluric pixels (kept) are close
# to the clean pixels. (Pixels with bits 0–7 mostly carry no usable flux; they contribute the spike at zero.)
#
# ## 4.5 The fibre: what the labels do not know
#
# The APOGEE spectrograph's line-spread function (LSF) — the profile into which an infinitely narrow line is
# blurred — varies across the detector, i.e. with the fibre on which a star was observed (Ness et al. 2018,
# [ApJ 853, 198](https://doi.org/10.3847/1538-4357/aa9d8e), §3; Casey et al. 2016, §6). A broader LSF makes line
# cores shallower at fixed equivalent width. Our model does not know the fibre, so this should show up in the
# residuals. The catalogue gives `meanfib`, the mean fibre number over a star's visits (1–300).

# %%
mt = vectorize(to_x(L[t_use])) @ theta.T                            # model at the ASPCAP labels, training stars
rt = np.where(np.isfinite(err[t_use]), flux[t_use] - mt, np.nan)
deep = on & (theta[:, 0] < 0.7)                                      # cores of strong lines
depth_res = np.nanmean(rt[:, deep], axis=1)                          # mean residual at line cores, per star
fib = tr.meanfib.to_numpy()[t_use]
from scipy.stats import spearmanr
rho, pval = spearmanr(fib, depth_res, nan_policy='omit')
bins = np.arange(0, 301, 30)
idx = np.digitize(fib, bins)
med = [np.nanmedian(depth_res[idx == b]) for b in range(1, len(bins))]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(fib, depth_res * 100, '.', ms=2, alpha=0.4)
ax.plot(0.5 * (bins[1:] + bins[:-1]), np.array(med) * 100, 'ko-', label='median in 30-fibre bins')
ax.axhline(0, color='k', lw=0.5)
ax.set(xlabel='mean fibre number', ylabel='mean residual at strong line cores [%]', ylim=(-1.5, 1.5))
ax.legend(fontsize=8)
plt.tight_layout()
print(f'{deep.sum()} line-core pixels; Spearman rho(fibre, residual) = {rho:.3f}, p = {pval:.1e}')
print('median residual [%] in 30-fibre bins:', np.round(np.array(med) * 100, 2))
# Test: line-core residuals depend on the fibre (a significant monotonic trend), i.e. on something not in the labels.
assert rho > 0.2 and pval < 1e-6

# %% [markdown]
# **Figure 4.6.** Mean residual (data − model) over the cores of strong lines ($\theta_0 < 0.7$) of each training
# star, against its mean fibre number (dots), with medians in bins of 30 fibres (black). A positive residual means
# shallower cores than the model predicts at the star's labels. The binned medians (printed above) rise from about
# −0.9% at the lowest fibre numbers to about +0.4–0.5% above 240: stars on high-numbered fibres have shallower line cores at the same labels, the signature of
# a broader LSF. N18 added the LSF width
# as an extra "label" of the model to absorb this; here we only note that part of $s_\lambda$ at line cores is
# instrumental.
#
# ## Summary
#
# - Trained on ~1300 high-S/N giants (after removing the 3% that the model fits badly), the three-label quadratic
#   model fits held-out spectra with median χ²/ν ≈ 0.8 once its per-pixel scatter is included; without it, χ²/ν is
#   ≈ 8. Residuals are heavier-tailed than Gaussian.
# - The coefficients are interpretable: they are large in known lines, and iron lines deepen with [Fe/H].
# - Bad-pixel and sky-line flags mark real outliers; their zero weight is justified.
# - Part of what the labels miss is instrumental: line depths depend on the fibre (LSF).
#
# **Exercises.**
# 1. Retrain without the scatter (set $s_\lambda = 0$). How do the labels of the held-out stars change?
# 2. Add `meanfib` (scaled) as a fourth label, as N18 did with the LSF width. Does the residual trend of Figure 4.6
#    disappear? What happens to the scatter at line cores?

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
# - Nidever et al. 2015, *The Data Reduction Pipeline for the Apache Point Observatory Galactic Evolution Experiment*, AJ 150, 173. [ADS](https://ui.adsabs.harvard.edu/abs/2015AJ....150..173N)
# - Smith et al. 2021, *The APOGEE Data Release 16 Spectral Line List*, AJ 161, 254. [ADS](https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S)
