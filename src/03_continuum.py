# %% [markdown]
# # Chapter 3 — Continuum normalisation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/03_continuum.ipynb)
#
# **Goal.** The Cannon models *continuum-normalised* spectra: the flux divided by an estimate of the stellar
# continuum, so that only the absorption lines remain. Here we build that normalisation following Ness et al. 2015
# (N15, §2.3 and §5.3), and test the one property The Cannon needs from it: that it does not change with the S/N of
# the spectrum. A model trained on S/N = 300 spectra and applied to S/N = 30 spectra is only right if both are
# normalised in the same way.
#
# **You will learn**
# 1. the simplest pseudo-continuum, an upper quantile of the flux in a running window, and why it depends on S/N;
# 2. how to find **continuum pixels** — pixels whose normalised flux is ≈ 1 and does not depend on the labels —
#    from a first Cannon model;
# 3. to fit a low-order polynomial to those pixels only, with inverse-variance weights, per detector;
# 4. to measure the S/N dependence of a normalisation by adding noise to high-S/N spectra.

# %%
#@@LOADER
#@@NORM
#@@CANNON

labels = load('labels')
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
spec = load('spectra_train')
lines = load('linelist')
flux, err = prepare(spec)                                  # err = inf at flagged pixels and outside the detectors
L = tr[['teff', 'logg', 'fe_h']].to_numpy()
print(f'{len(tr)} training spectra, {NPIX} pixels each')

# %% [markdown]
# ## 3.1 A running-quantile pseudo-continuum
#
# Absorption lines only remove flux, so the continuum lies near the *top* of the spectrum. N15 started from the
# 90th percentile of the flux in a window of 50 Å moved along the spectrum. We compute it on a coarse grid of
# window centres (every 10 pixels) within each detector and interpolate in between.

# %%
from numpy.lib.stride_tricks import sliding_window_view


def quantile_continuum(flux, err, q=90, width=50.0, step=10):
    """q-th percentile of the good pixels in a running window of `width` A, per detector (N15 sect. 2.3).
    Returns the continuum (N, NPIX); NaN outside the detectors."""
    cont = np.full(flux.shape, np.nan)
    for lo, hi in CHIPS:
        c = np.flatnonzero((wave > lo) & (wave < hi))
        npw = int(width / (np.median(wave[c]) * np.log(10) * 6e-6))      # window length in pixels (~230)
        f = np.where(np.isfinite(err[:, c]), flux[:, c], np.nan)          # flagged pixels do not vote
        win = sliding_window_view(f, npw, axis=1)[:, ::step]               # (N, centres, npw): a view, no copy
        qc = np.nanpercentile(win, q, axis=2)                              # (N, centres)
        centre = np.arange(win.shape[1]) * step + npw // 2                 # pixel index (within c) of each centre
        for n in range(len(f)):
            cont[n, c] = np.interp(np.arange(len(c)), centre, qc[n])       # constant beyond the first/last centre
    return cont


cq = quantile_continuum(flux, err)
fq, eq = flux / cq, err / cq                                  # quantile-normalised flux and error
i = np.argmin(np.abs(tr.teff - 4500) + 300 * np.abs(tr.logg - 2.0) + 1000 * np.abs(tr.fe_h))
fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
good = np.isfinite(err[i])
show = lambda f: np.where(good, f, np.nan)                     # NaN breaks the line at gaps and flagged pixels
ax[0].plot(wave, show(flux[i]), 'k', lw=0.4, label='flux / median')
ax[0].plot(wave, cq[i], 'C1', lw=1.5, label='running 90th percentile (50 Å)')
ax[0].set(ylabel='flux', ylim=(0.3, 1.5))
ax[0].legend(fontsize=8)
ax[1].plot(wave, show(fq[i]), 'k', lw=0.4)
ax[1].axhline(1, color='C1', lw=1)
ax[1].set(xlabel='vacuum wavelength [Å]', ylabel='normalised flux', ylim=(0.3, 1.2))
plt.tight_layout()

# %% [markdown]
# **Figure 3.1.** Top: the spectrum of a typical training giant (black; flagged pixels omitted) and its running
# 90th-percentile continuum (orange). Bottom: the ratio, a pseudo-continuum-normalised spectrum (the few upward
# spikes are sky-line residuals that the mask does not flag). The method follows
# the broad shape of the spectrum well, but by construction 10% of the pixels in every window lie above 1, and the
# level it assigns depends on how many lines are in the window — and on the noise, as the next test shows.
#
# ## 3.2 The S/N test
#
# To see how a normalisation responds to noise, take high-S/N spectra, add Gaussian noise to bring them to a lower
# S/N, normalise both versions, and compare. If a spectrum has errors $\sigma$ and we want errors
# $\sigma' = f/{\rm (S/N)}' > \sigma$, we add noise of variance $\sigma'^2 - \sigma^2$ (the noise already there
# counts). The **bias** of the normalisation is the mean difference between the normalised noisy spectrum and the
# normalised original, at pixels that are continuum in the original. We average over 200 stars, so the random part
# of the difference averages away.

# %%
rng = np.random.default_rng(3)
sub = rng.choice(len(tr), 200, replace=False)                 # 200 training stars for the test


def degrade(f, e, snr):
    """Add Gaussian noise so that the per-pixel S/N becomes snr (where it was higher)."""
    target = np.abs(f) / snr
    extra = np.sqrt(np.clip(target**2 - e**2, 0, None))
    e2 = np.where(np.isfinite(e), np.maximum(e, target), np.inf)
    return f + rng.normal(size=f.shape) * np.where(np.isfinite(e), extra, 0), e2


SNRS = [20, 30, 50, 100, 200]
f0, e0 = flux[sub], err[sub]
ref_q = f0 / quantile_continuum(f0, e0)
bias_q = []
for snr in SNRS:
    fd, ed = degrade(f0, e0, snr)
    nq = fd / quantile_continuum(fd, ed)
    sel = np.isfinite(e0) & (ref_q > 0.98)                    # pixels near the continuum in the original
    bias_q.append(np.mean((nq - ref_q)[sel]))
print('quantile normalisation, mean change of the continuum level vs S/N:', dict(zip(SNRS, np.round(bias_q, 4))))

# %% [markdown]
# At S/N = 30 the normalised continuum drops by ≈ 2% (≈ 4% at S/N = 20): noise scatters pixels upwards as well as
# downwards, so the 90th percentile of a noisy window lies higher than that of the noiseless one, and dividing by it
# pushes the whole spectrum down. The change of a continuum level is small, but it is
# systematic and S/N-dependent — exactly what makes a model trained at high S/N biased at low S/N (N15 §2.3).
#
# ## 3.3 Continuum pixels from a first model
#
# N15 (§2.3, §5.3) define the continuum *with the model itself*: train a Cannon model on the
# quantile-normalised spectra, and call continuum the pixels where the model's baseline spectrum $\theta_0$ is
# close to 1 **and** the flux hardly depends on any label. Those pixels are fixed once and for all; every spectrum —
# training or test, high or low S/N — is then normalised with the same pixels.
#
# We train a three-label model ($T_{\rm eff}$, $\log g$, [Fe/H]) on the quantile-normalised training spectra and
# apply N15's criteria: $|\theta_0 - 1| < 0.015$ and first-order coefficients (per unit of the *unscaled* labels)
# below $10^{-5}$ per K, 0.0045 per dex of $\log g$ and 0.0085 per dex of [Fe/H].

# %%
pivot = np.median(L, axis=0)
scale = (np.percentile(L, 97.5, axis=0) - np.percentile(L, 2.5, axis=0)) / 2
x = (L - pivot) / scale
eq_use = np.where(np.isfinite(fq), eq, np.inf)
theta_q, _, s_q = train(x, np.nan_to_num(fq), eq_use)
dphys = theta_q[:, 1:4] / scale                                # coefficients per K, per dex, per dex
LIMITS = np.array([1e-5, 0.0045, 0.0085])                      # N15 sect. 5.3
on_chip = np.isfinite(cq[0])
cont = on_chip & (np.abs(theta_q[:, 0] - 1) < 0.015) & np.all(np.abs(dphys) < LIMITS, axis=1)
print(f'continuum pixels: {cont.sum()} of {on_chip.sum()} pixels on the detectors ({cont.sum() / on_chip.sum():.1%}); '
      'per detector:', [int(cont[(wave > lo) & (wave < hi)].sum()) for lo, hi in CHIPS])

fig, ax = plt.subplots(3, 1, figsize=(14, 8))
for a, (lo, hi) in zip(ax, [(15200, 15330), (16150, 16280), (16700, 16830)]):
    z = (wave > lo) & (wave < hi)
    a.plot(wave[z], theta_q[z, 0], 'k', lw=1, label=r'baseline $\theta_0$')
    a.plot(wave[z & cont], theta_q[z & cont, 0], 'o', color='C2', ms=4, label='continuum pixels')
    for d, nm in zip(range(3), [r'$T_{\rm eff}$ (per 1000 K)', r'$\log g$ (per dex)', '[Fe/H] (per dex)']):
        a.plot(wave[z], 0.75 + dphys[z, d] * (1000 if d == 0 else 1), lw=0.7, label=r'0.75 + $\partial f/\partial$' + nm)
    a.set(ylabel='normalised flux', ylim=(0.55, 1.08))
ax[0].legend(fontsize=7, ncol=5, loc='lower left')
ax[-1].set(xlabel='vacuum wavelength [Å]')
plt.tight_layout()

# %% [markdown]
# **Figure 3.2.** The first model, trained on quantile-normalised spectra, in three 130 Å windows (one per
# detector). Black: the baseline spectrum $\theta_0$ (a giant with the median training labels). Coloured lines,
# offset to 0.75: the derivatives of the flux with respect to $T_{\rm eff}$ (per 1000 K), $\log g$ and [Fe/H] (per
# dex). Green dots: the continuum pixels — near 1 in $\theta_0$ and with all three derivatives small. They sit
# between the lines, a few per window.
#
# Are they really free of lines? Compare their distance to the nearest line of the line list with that of all
# pixels on the detectors.

# %%
strong = lines[lines.strength > lines.groupby('species').strength.transform('max') - 1.0]   # within 1 dex of each species' strongest
dist = np.min(np.abs(wave[:, None] - strong.wave_vac.to_numpy()[None, :]), axis=1)          # A to the nearest strong line
near_all, near_cont = np.mean(dist[on_chip] < 0.5), np.mean(dist[cont] < 0.5)
print(f'fraction within 0.5 A of a strong listed line: all detector pixels {near_all:.3f}, continuum pixels {near_cont:.3f}')
# Test: continuum pixels avoid strong lines (at most half as often within 0.5 A of one as an average pixel).
assert near_cont < 0.5 * near_all

# %% [markdown]
# ## 3.4 Normalising with the continuum pixels
#
# On each detector, fit a second-order Chebyshev polynomial in wavelength to the flux of the continuum pixels only,
# weighting each pixel by $1/\sigma^2$ (flagged pixels have zero weight), and divide the spectrum by it (N15 §2.3).
# This is a small weighted least-squares fit per star and detector; `cheb_normalise` (defined at the top) does it
# for all stars at once.
#
# The pixel list we derived is also shipped with the tutorial as the file `continuum_pixels` (computed by
# `scripts/build_data.py` with the same code), so that later chapters do not have to repeat this chapter. Check
# that the two agree:

# %%
shipped = load('continuum_pixels')['continuum'].to_numpy()
print(f'shipped list: {shipped.sum()} pixels; identical to ours: {np.array_equal(shipped, cont)}')
assert np.array_equal(shipped, cont)

fc, ec = cheb_normalise(flux, err, cont)
fig, ax = plt.subplots(figsize=(14, 4))
z = (wave > 15600) & (wave < 15800) & np.isfinite(err[i])
ax.plot(wave[z], fq[i, z], color='C1', lw=0.8, label='running 90th percentile')
ax.plot(wave[z], fc[i, z], 'k', lw=0.8, label='Chebyshev on continuum pixels')
ax.plot(wave[z & cont], fc[i, z & cont], 'o', color='C2', ms=4, label='continuum pixels')
ax.axhline(1, color='0.5', lw=0.5)
ax.set(xlabel='vacuum wavelength [Å]', ylabel='normalised flux')
ax.legend(fontsize=8)
plt.tight_layout()
with np.errstate(invalid='ignore', divide='ignore'):
    print(f'star {i}: median ratio of the two normalisations = {np.nanmedian(fc[i, on_chip] / fq[i, on_chip]):.4f}')

# %% [markdown]
# **Figure 3.3.** A 200 Å section of the typical giant normalised in the two ways: running 90th percentile
# (orange) and second-order Chebyshev polynomials fitted to the continuum pixels (black; green dots). The two
# differ by a slowly varying factor of a few tenths of a per cent (median ratio 1.004 for this star): the running
# quantile puts the "continuum" near the top of the noise and of the weak lines, the polynomial through the
# continuum pixels near their mean. Neither is the true
# continuum of the star; The Cannon only needs the *same* definition for every spectrum.
#
# ## 3.5 The S/N test again
#
# Now repeat the noise-injection test of §3.2 with the continuum-pixel normalisation. The division by the fitted
# continuum is not exactly linear in the noise (a ratio of noisy quantities), so a small bias can remain: measure
# it, do not assume it is zero.

# %%
ref_c = cheb_normalise(f0, e0, cont)[0]
bias_c = []
for snr in SNRS:
    fd, ed = degrade(f0, e0, snr)
    nc = cheb_normalise(fd, ed, cont)[0]
    sel = np.isfinite(e0) & (ref_c > 0.98) & np.isfinite(nc)
    bias_c.append(np.mean((nc - ref_c)[sel]))
print('continuum-pixel normalisation, mean change vs S/N:', dict(zip(SNRS, np.round(bias_c, 5))))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(SNRS, np.abs(bias_q) * 100, 'o-', label='running 90th percentile')
ax.plot(SNRS, np.abs(bias_c) * 100, 's-', label='Chebyshev on continuum pixels')
ax.set(xscale='log', yscale='log', xlabel='S/N after adding noise', ylabel='|change of continuum level| [%]')
ax.legend(fontsize=8)
plt.tight_layout()
k30 = SNRS.index(30)
# Test: at S/N = 30 the quantile normalisation shifts the continuum at least 3 times more than the fixed-pixel one.
assert abs(bias_q[k30]) > 3 * abs(bias_c[k30])

# %% [markdown]
# **Figure 3.4.** Mean change of the normalised flux at continuum pixels when 200 high-S/N training spectra are
# degraded to a lower S/N (horizontal axis) and normalised again, for the two methods (absolute values, log
# scales). The running quantile moves the continuum by ≈ 1% at S/N = 50, 2% at 30 and 4% at 20. The
# continuum-pixel fit changes it by less than $10^{-4}$ at every S/N — consistent with zero at the precision of this
# test — because the same pixels, with the same weights, define the continuum at any S/N.
#
# ## Summary
#
# - Quantile-based pseudo-continua depend on S/N; that would bias any model trained at high S/N and used at low S/N.
# - Continuum pixels are chosen from a first Cannon model: baseline flux ≈ 1 and no dependence on the labels
#   (N15's criteria select ≈ 9% of the pixels here; N15 had 6.5%).
# - A low-order polynomial fitted to those pixels, per detector, with inverse-variance weights, is a normalisation
#   whose S/N dependence we measured to be small.
# - From now on every spectrum is normalised this way, with the shipped list `continuum_pixels`.
#
# **Exercises.**
# 1. Use a 4th-order instead of a 2nd-order Chebyshev polynomial. Does the S/N bias change? Does the normalised
#    flux at the detector edges?
# 2. Choose the continuum pixels at random (same number). Casey et al. 2016 argue that any fixed set of pixels would
#    work for The Cannon; what do you lose?

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Ness et al. 2018, *Galactic Doppelgängers: The Chemical Similarity Among Field Stars and Among Stars with a Common Birth Origin*, ApJ 853, 198. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N)
# - Santos-Peral et al. 2020, *The AMBRE Project: Spectrum normalisation influence on Mg abundances in the metal-rich Galactic disc*, A&A 639, A140. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...639A.140S)
