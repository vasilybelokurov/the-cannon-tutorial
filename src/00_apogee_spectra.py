# %% [markdown]
# # Chapter 0 — APOGEE spectra and their labels
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/00_apogee_spectra.ipynb)
#
# **Goal.** Meet the data of this tutorial: near-infrared spectra of red giants from the APOGEE survey, and the
# stellar parameters and abundances ("labels") that the APOGEE pipeline derived from them. Everything that follows
# — the data-driven model called The Cannon — learns how these spectra depend on these labels.
#
# **You will learn**
# 1. what an APOGEE spectrum is: wavelength grid, three detectors, flux, flux error and a per-pixel bitmask;
# 2. how the signal-to-noise ratio (S/N) of a spectrum is defined, and how to check it;
# 3. what the labels are ($T_{\rm eff}$, $\log g$, [Fe/H], [X/Fe]), where they come from, and which stars are in
#    each data file of the tutorial;
# 4. how much precision the compact storage of the spectra costs (none that matters).
#
# **About APOGEE.** The Apache Point Observatory Galactic Evolution Experiment (Majewski et al. 2017,
# [AJ 154, 94](https://ui.adsabs.harvard.edu/abs/2017AJ....154...94M)) observed several hundred thousand stars,
# mostly red giants, with a 300-fibre spectrograph at a resolving power $R = \lambda/\Delta\lambda \approx 22\,500$
# in the $H$ band (1.51–1.70 µm), where interstellar dust absorbs ~6 times less light than in the $V$ band. We use
# Data Release 17 (Abdurro'uf et al. 2022, [ApJS 259, 35](https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A)),
# and only spectra taken with the 2.5 m telescope at Apache Point (APO), so that all spectra come from one
# instrument.

# %%
#@@LOADER

labels = load('labels')                                   # one row per star and sample (APOGEE DR17 allStar columns)
spec = load('spectra_train')                              # the training set of the tutorial
print(labels['sample'].value_counts().to_string())
print('arrays in spectra_train:', {k: (v.shape, v.dtype.name) for k, v in spec.items()})

# %% [markdown]
# ## 0.1 The stars and their labels
#
# The **labels** are the stellar parameters and abundances measured by ASPCAP, the APOGEE Stellar Parameter and
# Chemical Abundances Pipeline (García Pérez et al. 2016, [AJ 151, 144](https://ui.adsabs.harvard.edu/abs/2016AJ....151..144G)),
# which fits each spectrum with a grid of synthetic spectra computed from model atmospheres. DR17 gives
# *calibrated* values (`teff`, `logg`, `fe_h`, `mg_fe`, ...), shifted onto external scales (photometric
# temperatures, asteroseismic gravities), and the raw spectroscopic ones (`teff_spec`, `logg_spec`). We use the
# calibrated values as our reference labels throughout; whatever The Cannon learns will be on this scale.
#
# - $T_{\rm eff}$: effective temperature [K]; $\log g$: base-10 logarithm of the surface gravity in cgs units
#   [dex]; [Fe/H] $= \log_{10}(N_{\rm Fe}/N_{\rm H})_\star - \log_{10}(N_{\rm Fe}/N_{\rm H})_\odot$ [dex];
#   [X/Fe] = [X/H] − [Fe/H] for element X.
#
# The data files are samples of APOGEE stars chosen for specific tests (selection in `scripts/build_data.py`):
#
# | sample | content | used in |
# |---|---|---|
# | `train` | giants with S/N > 200 (> 100 at [Fe/H] < −1), at most 4 stars per cell of 100 K × 0.25 dex × 0.25 dex | training (Ch 3–5, 8) |
# | `test` | giants with 50 < S/N < 100, same label range | low-S/N application (Ch 8) |
# | `metalpoor` | giants with −2.5 < [Fe/H] < −1.5 | extrapolation (Ch 5) |
# | `dwarfs` | main-sequence stars, $\log g > 4$ | extrapolation (Ch 5) |
# | `visits` | giants with 4–12 single-visit spectra | repeat observations (Ch 6) |
# | `clusters` | members of 17 open and globular clusters | homogeneity (Ch 7) |
#
# All samples avoid stars that ASPCAP flagged as bad, likely binaries (radial-velocity scatter between visits
# above 1 km/s) and spectra with severe warnings; the field samples contain no cluster members.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
order = ['train', 'test', 'metalpoor', 'dwarfs', 'visits', 'clusters']
for a, group in zip(ax, [['train'], ['test', 'visits'], ['metalpoor', 'dwarfs', 'clusters']]):
    for nm in group:
        d = labels[labels['sample'] == nm]
        a.scatter(d.teff, d.logg, s=4, alpha=0.6, label=f'{nm} ({len(d)})')
    a.set(xlabel=r'$T_{\rm eff}$ [K]', xlim=(6200, 3600))
    a.legend(fontsize=8, markerscale=3, loc='upper left')
ax[0].set(ylabel=r'$\log g$ [dex]', ylim=(5, 0))
plt.tight_layout()

tr = labels[labels['sample'] == 'train']
print(f"train: {len(tr)} stars; T_eff {tr.teff.min():.0f}-{tr.teff.max():.0f} K, log g {tr.logg.min():.2f}-{tr.logg.max():.2f}, "
      f"[Fe/H] {tr.fe_h.min():.2f}-{tr.fe_h.max():.2f}; median S/N {tr.snr.median():.0f}")
# the spectra file and the label table list the same stars in the same order
assert (spec['apogee_id'] == tr.apogee_id.to_numpy()).all()

# %% [markdown]
# **Figure 0.1.** Kiel diagrams ($\log g$ against $T_{\rm eff}$, with the hot, high-gravity side to the left and
# bottom as in a Hertzsprung–Russell diagram) of the samples, with ASPCAP calibrated labels; the number of stars is
# in the legend. Left: the training set, red giants from the base of the giant branch ($\log g \approx 3.5$) to
# near its tip ($\log g \approx 0.8$), plus the red clump at $T_{\rm eff} \approx 4700$ K, $\log g \approx 2.4$.
# Middle: the low-S/N test set and the stars with repeat visits, over the same range. Right: the samples that test
# the model outside or beside the training range — metal-poor giants (hotter at fixed $\log g$, because metal-poor
# giant branches are bluer), main-sequence dwarfs ($\log g > 4$, absent from the training set), and cluster members.
#
# Because the training set was chosen to cover label space evenly (a cap on the number of stars per cell), it is
# *not* a random sample of APOGEE giants: rare labels are over-represented. Figure 0.2 shows the coverage in the
# three labels we will start with.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
pairs = [('teff', 'logg'), ('teff', 'fe_h'), ('fe_h', 'mg_fe')]
names = {'teff': r'$T_{\rm eff}$ [K]', 'logg': r'$\log g$ [dex]', 'fe_h': '[Fe/H] [dex]', 'mg_fe': '[Mg/Fe] [dex]'}
for a, (x, y) in zip(ax, pairs):
    h = a.hist2d(tr[x], tr[y], bins=30, cmap='Greys', cmin=1)
    plt.colorbar(h[3], ax=a, label='stars per bin')
    a.set(xlabel=names[x], ylabel=names[y])
plt.tight_layout()
n_mp = (tr.fe_h < -1).sum()
print(f'training stars with [Fe/H] < -1: {n_mp} ({n_mp / len(tr):.0%})')

# %% [markdown]
# **Figure 0.2.** Label coverage of the training set: numbers of stars in bins of $T_{\rm eff}$–$\log g$,
# $T_{\rm eff}$–[Fe/H] and [Fe/H]–[Mg/Fe] (grey scale; empty bins blank). The giants follow the narrow giant
# branch in the first panel, so $T_{\rm eff}$ and $\log g$ are strongly correlated: a spectrum must separate two
# labels that, among real stars, nearly move together. In the last panel, the two sequences of the Milky Way disc
# in [Mg/Fe] at fixed [Fe/H] (the high-α thick disc and the low-α thin disc) are already visible; we come back to
# them in Chapter 8.
#
# ## 0.2 One spectrum
#
# A combined APOGEE spectrum (`apStar` file) is sampled on a fixed grid of 8575 pixels, equally spaced in the
# logarithm of the vacuum wavelength,
#
# $$ \log_{10}(\lambda_i / {\rm Å}) = 4.179 + 6 \times 10^{-6}\, i, \qquad i = 0, \dots, 8574, $$
#
# so each pixel has the same velocity width $\Delta v = c \ln(10) \times 6\times10^{-6} = 4.14$ km/s, and the
# resolution element ($c/R \approx 13$ km/s) spans about three pixels. The spectra are already shifted to the
# rest frame of the star. Light falls on three detectors ("blue", "green", "red"), with gaps between them where
# there are no data (NaN in our files). Each spectrum comes with
#
# - `flux`: the flux, here divided by the median flux of the spectrum (so typical values are ≈ 1);
# - `err`: its 1σ uncertainty (same scaling);
# - `mask`: the `APOGEE_PIXMASK` bitmask, one integer per pixel whose bits flag problems (bad pixels, cosmic
#   rays, bright sky lines, strong telluric absorption, persistence in the detector, ...).

# %%
NPIX = 8575
wave = 10**(4.179 + 6e-6 * np.arange(NPIX))              # vacuum wavelength [Angstrom] of every pixel
flux, err, mask = spec['flux'].astype(float), spec['err'].astype(float), spec['mask']   # to float64 for arithmetic
has_data = np.isfinite(flux).mean(axis=0) > 0.5          # pixels with data in at least half of the spectra
edges = np.flatnonzero(np.diff(has_data.astype(int)))     # where data start and stop: the three detectors
chips = [(wave[a + 1], wave[b]) for a, b in zip(edges[::2], edges[1::2])]
print(f'grid: {wave[0]:.2f} - {wave[-1]:.2f} A; velocity step {2.998e5 * np.log(10) * 6e-6:.3f} km/s')
print('detectors [A]:', [(round(a), round(b)) for a, b in chips], f'; {has_data.sum()} pixels with data')
# Test: the grid is the documented apStar grid, and there are three separate detectors.
assert abs(wave[0] - 15100.80) < 0.01 and abs(wave[-1] - 16999.80) < 0.05 and len(chips) == 3

i = np.argmin(np.abs(tr.teff - 4500) + 300 * np.abs(tr.logg - 2.0) + 1000 * np.abs(tr.fe_h))   # a typical giant
star = tr.iloc[i]
fig, ax = plt.subplots(2, 1, figsize=(14, 7))
ax[0].plot(wave, flux[i], 'k', lw=0.4)
for a, b in chips:
    ax[0].axvspan(a, b, color='C0', alpha=0.07)
ax[0].set(xlabel='vacuum wavelength [Å]', ylabel='flux / median', ylim=(0.3, 1.5),
          title=f'{star.apogee_id}: $T_{{\\rm eff}}$ = {star.teff:.0f} K, log g = {star.logg:.2f}, [Fe/H] = {star.fe_h:.2f}, S/N = {star.snr:.0f}')
z = (wave > 15650) & (wave < 15800)                       # a zoom on 150 A of the blue detector
ax[1].plot(wave[z], flux[i, z], 'k', lw=0.8, drawstyle='steps-mid', label='flux')
ax[1].fill_between(wave[z], flux[i, z] - err[i, z], flux[i, z] + err[i, z], color='C1', alpha=0.5, step='mid', label='±1σ')
bad = z & (mask[i] > 0)
ax[1].plot(wave[bad], flux[i, bad], 'r|', ms=10, label='a mask bit is set')
ax[1].set(xlabel='vacuum wavelength [Å]', ylabel='flux / median')
ax[1].legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# **Figure 0.3.** The combined APOGEE spectrum of a typical red giant of the training set (identifier, labels and
# S/N in the title). Top: the full spectrum, divided by its median; shaded bands are the three detectors, with no
# data in between. The overall shape (here a slow decline to the red, with steps between detectors) is the
# spectral energy distribution of the star multiplied by the instrument's throughput, not a property of the lines;
# Chapter 3 removes it. The few narrow upward spikes are residuals of bright sky emission lines. Bottom: a 150 Å
# zoom with the ±1σ error band (orange; it widens at the sky-line residual near 15 659 Å) and the pixels that have
# at least one mask bit set (red; here mostly persistence flags, see §0.4). Almost every dip is an absorption line or a blend of lines; the strongest here are
# Mg I lines near 15 745–15 770 Å (identified in Chapter 1).
#
# ## 0.3 Signal-to-noise ratio
#
# The catalogue's `snr` is one number per star; per pixel, the S/N is simply flux / error. Its median over the
# pixels with data should track the catalogue value.

# %%
snr_pix = np.nanmedian(flux / err, axis=1)                # median per-pixel S/N of every spectrum
ratio = snr_pix / tr.snr.to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].loglog(tr.snr, snr_pix, '.', ms=3)
ax[0].plot([100, 2000], [100, 2000], 'k-', lw=0.8, label='equal')
ax[0].set(xlabel='catalogue SNR', ylabel='median of flux / error per pixel')
ax[0].legend()
ax[1].hist(ratio, bins=50)
ax[1].set(xlabel='per-pixel median S/N / catalogue SNR', ylabel='stars')
plt.tight_layout()
print(f'median ratio {np.median(ratio):.3f}; 1st-99th percentile {np.percentile(ratio, 1):.3f}-{np.percentile(ratio, 99):.3f}')
# Test: the catalogue S/N is the median per-pixel flux/error (98% of the stars agree to 0.1%).
assert abs(np.median(ratio) - 1) < 1e-3 and np.mean(np.abs(ratio - 1) < 1e-3) > 0.95

# %% [markdown]
# **Figure 0.4.** Left: median per-pixel S/N (flux/error) of each training spectrum against the catalogue `snr`;
# the line marks equality. Right: histogram of their ratio. They are identical to 0.1% for almost every star: the
# catalogue S/N *is* the median per-pixel S/N, so a "S/N = 100" APOGEE spectrum has errors of ≈ 1% of the flux in a
# typical pixel. Note how high it is for the training set (median ≈ 300, up to ≈ 2000): these spectra were chosen to
# be nearly noiseless.
#
# ## 0.4 The bitmask
#
# Each bit of `APOGEE_PIXMASK` flags a specific problem (definitions:
# [SDSS DR17 bitmasks](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)). How often each bit is set depends on
# wavelength. Two families matter most: pixels affected by the **sky** (bright OH emission lines from the Earth's
# atmosphere, bit 12 `SIG_SKYLINE`; telluric absorption, bit 13) and pixels affected by the **detector** (bad
# pixels, cosmic rays, and "persistence" — a ghost of earlier bright exposures in parts of one detector, bits 9–11).

# %%
BITS = {0: 'BADPIX', 1: 'CRPIX', 2: 'SATPIX', 3: 'UNFIXABLE', 4: 'BADDARK', 5: 'BADFLAT', 6: 'BADERR', 7: 'NOSKY',
        8: 'LITTROW_GHOST', 9: 'PERSIST_HIGH', 10: 'PERSIST_MED', 11: 'PERSIST_LOW', 12: 'SIG_SKYLINE',
        13: 'SIG_TELLURIC', 14: 'NOT_ENOUGH_PSF'}
frac = {b: ((mask >> b) & 1).mean(axis=0) for b in BITS}  # fraction of training spectra with bit b set, per pixel
fig, ax = plt.subplots(figsize=(14, 4))
for b in [0, 1, 9, 10, 11, 12, 13]:
    ax.plot(wave, frac[b], lw=0.7, label=f'{b} {BITS[b]}')
ax.set(xlabel='vacuum wavelength [Å]', ylabel='fraction of spectra with the bit set', ylim=(0, 1))
ax.legend(fontsize=8, ncol=4, loc='upper center')
plt.tight_layout()
print('fraction of all pixels with data that have the bit set:')
print({BITS[b]: round(float(frac[b][has_data].mean()), 4) for b in BITS if frac[b][has_data].mean() > 0})

# %% [markdown]
# **Figure 0.5.** For each wavelength, the fraction of training spectra in which a given `APOGEE_PIXMASK` bit is
# set (selected bits; the others are rare). `SIG_SKYLINE` (bit 12) is set in narrow spikes at the wavelengths of
# the atmospheric OH emission lines, `SIG_TELLURIC` (bit 13) in broader bands of telluric absorption; the
# persistence bits (9–11) cover large parts of the middle detector (15 860–16 440 Å) for stars observed on the
# affected fibres. These pixels are not lost — the flux is
# there, sometimes with an underestimated error.
#
# **What we do with flagged pixels.** We never delete pixels: The Cannon compares the same pixels in every
# spectrum. Instead we give pixels with the detector and sky bits 0–7 and 12 an error of infinity (zero weight),
# so that they do not contribute to any fit, while the pixel stays in the spectrum. Persistence is not masked
# (Casey et al. 2016 ignored the persistence bits because in their DR12 spectra they often flagged every pixel of one detector; in our DR17 files they cover large parts of the middle detector, §0.4). Whether this choice is
# good enough is tested in Chapter 4, where we compare the model residuals of flagged and unflagged pixels.

# %%
BAD_BITS = sum(1 << b for b in [0, 1, 2, 3, 4, 5, 6, 7, 12])   # bitwise OR of the bits we treat as bad
bad = (mask & BAD_BITS) > 0
print(f'BAD_BITS = {BAD_BITS}; fraction of pixels with data that get zero weight: {bad[:, has_data].mean():.4f}')
err_used = np.where(bad, np.inf, err)                      # the error array used by all fits

# %% [markdown]
# ## 0.5 How much precision does the storage cost?
#
# To keep the downloads small, the errors are stored as 16-bit floating-point numbers ("float16"), which have 11
# significant bits: rounding changes a number $x$ by at most $2^{-11}|x| \approx 4.9\times10^{-4}|x|$. For an error
# that is harmless (σ is known to 0.05%). Could the *flux* be stored the same way? The flux is stored as float32 (24
# significant bits); here we round it to float16 and measure what that would do, compared with the noise.

# %%
f32 = spec['flux']                                          # as stored (float32)
r16 = np.abs(f32.astype(np.float16).astype(float) - f32) / spec['err']   # actual float16 rounding, in units of sigma
r32 = 2.0**-24 * np.abs(f32) / spec['err']                  # upper bound of the float32 rounding, in units of sigma
ok = np.isfinite(r16) & ~bad
print(f'float16 rounding of the flux / noise: rms {np.sqrt(np.mean(r16[ok]**2)):.3f}, 99.9th percentile '
      f'{np.percentile(r16[ok], 99.9):.3f}, max {r16[ok].max():.3f}')
print(f'float32 rounding / noise: at most {r32[ok].max():.1e}; dtypes in the file: flux {f32.dtype}, error {spec["err"].dtype}')
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(r16[ok], bins=100, log=True, histtype='step', label='float16 (not used for the flux)')
ax.hist(r32[ok] * 1e4, bins=100, log=True, histtype='step', label=r'float32 (used), upper bound $\times 10^4$')
ax.set(xlabel='rounding error of the flux / 1σ noise', ylabel='pixels')
ax.legend(fontsize=8)
plt.tight_layout()
# Tests: float32 rounding is negligible (< 1e-3 sigma everywhere); float16 would round 0.1% of the pixels by more
# than 0.25 sigma (an error comparable to the noise in the best spectra).
assert spec['flux'].dtype == np.float32 and r32[ok].max() < 1e-3
assert np.percentile(r16[ok], 99.9) > 0.25

# %% [markdown]
# **Figure 0.6.** Histogram (log scale) over all pixels of all training spectra of the rounding error of the flux
# divided by its 1σ error: the actual error if the flux were rounded to float16 (blue), and the upper bound for
# float32 (orange, multiplied by $10^4$). The training spectra are so good — up to S/N ≈ 2000 per pixel — that
# float16 would round some pixels by a sizeable fraction of the noise (printed above); since rounding errors are not
# random in the way photon noise is (they depend on the value), we avoid them. (A first version of the data files
# stored the flux as float16; this test caught it.) With float32 the rounding is below $10^{-4}$σ.
#
# ## Summary
#
# - An APOGEE spectrum is 8575 pixels on a log-linear vacuum wavelength grid (15 101–16 999 Å), on three
#   detectors, with an error and a bitmask per pixel; the catalogue S/N is the per-pixel S/N.
# - Flagged pixels keep their place in the spectrum but get zero weight.
# - The labels are ASPCAP's calibrated $T_{\rm eff}$, $\log g$, [Fe/H] and [X/Fe]; the training set covers
#   3800–5300 K, $\log g$ 0.8–3.6 and [Fe/H] −1.5 to +0.4 evenly.
#
# **Exercises.**
# 1. Plot the spectra of the hottest and the coolest training star on top of each other. Which features change the
#    most?
# 2. For stars with persistence flags (bits 9–11) on the middle detector, compare their flux there with that of
#    stars of similar labels without the flags.

# %% [markdown]
# ## References
#
# - Abdurro'uf et al. 2022, *The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar, and APOGEE-2 Data*, ApJS 259, 35. [ADS](https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A)
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - García Pérez et al. 2016, *ASPCAP: The APOGEE Stellar Parameter and Chemical Abundances Pipeline*, AJ 151, 144. [ADS](https://ui.adsabs.harvard.edu/abs/2016AJ....151..144G)
# - Holtzman et al. 2015, *Abundances, Stellar Parameters, and Spectra from the SDSS-III/APOGEE Survey*, AJ 150, 148. [ADS](https://ui.adsabs.harvard.edu/abs/2015AJ....150..148H)
# - Jönsson et al. 2020, *APOGEE Data and Spectral Analysis from SDSS Data Release 16: Seven Years of Observations Including First Results from APOGEE-South*, AJ 160, 120. [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J)
# - Majewski et al. 2017, *The Apache Point Observatory Galactic Evolution Experiment (APOGEE)*, AJ 154, 94. [ADS](https://ui.adsabs.harvard.edu/abs/2017AJ....154...94M)
# - Nidever et al. 2015, *The Data Reduction Pipeline for the Apache Point Observatory Galactic Evolution Experiment*, AJ 150, 173. [ADS](https://ui.adsabs.harvard.edu/abs/2015AJ....150..173N)
# - Zasowski et al. 2013, *Target Selection for the Apache Point Observatory Galactic Evolution Experiment (APOGEE)*, AJ 146, 81. [ADS](https://ui.adsabs.harvard.edu/abs/2013AJ....146...81Z)
