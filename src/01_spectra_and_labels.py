# %% [markdown]
# # Chapter 1 — How spectra depend on labels
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/01_spectra_and_labels.ipynb)
#
# **Goal.** Before building any model, look at how the spectrum of a red giant changes when its temperature,
# gravity or metallicity changes, identify the lines responsible, and fit the flux of a single pixel as a
# function of the labels. That one-pixel fit, repeated at every pixel, *is* the training step of The Cannon.
#
# **You will learn**
# 1. to identify absorption lines in an APOGEE spectrum with a line list (and convert air to vacuum wavelengths);
# 2. which features respond to $T_{\rm eff}$, $\log g$ and [Fe/H];
# 3. to fit the flux of one pixel as a linear and as a quadratic function of three labels, and to read the
#    coefficients as derivatives of the spectrum.

# %%
#@@LOADER

labels = load('labels')
tr = labels[labels['sample'] == 'train'].reset_index(drop=True)
spec = load('spectra_train')
lines = load('linelist')                                   # strongest lines in the APOGEE window (see below)
NPIX = 8575
wave = 10**(4.179 + 6e-6 * np.arange(NPIX))                # vacuum wavelength grid [Angstrom] (Chapter 0)
BAD_BITS = sum(1 << b for b in [0, 1, 2, 3, 4, 5, 6, 7, 12])   # pixels given zero weight (Chapter 0)
flux = spec['flux'].astype(float)
err = np.where((spec['mask'] & BAD_BITS) > 0, np.inf, spec['err'].astype(float))
teff, logg, feh = tr.teff.to_numpy(), tr.logg.to_numpy(), tr.fe_h.to_numpy()
print(f'{len(tr)} training stars; {len(lines)} lines; species:', ', '.join(sorted(lines.species.unique())))

# %% [markdown]
# ## 1.1 A line list
#
# The file `linelist` is extracted from the line list that APOGEE used to compute its synthetic spectra for DR17
# (Smith et al. 2021, [AJ 161, 254](https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S)). For each line it gives
# the species (e.g. `FE I` = neutral iron, `CA II` = singly ionised calcium; `CN`, `CO`, `OH` = molecules), the
# wavelength, the excitation energy $\chi$ of the lower level and the oscillator strength $\log gf$. Line lists
# give **air** wavelengths, APOGEE spectra are on a **vacuum** scale; the conversion is
# $\lambda_{\rm vac} = n(\lambda)\,\lambda_{\rm air}$ with the refractive index of air $n \approx 1.00027$ in the
# $H$ band (Morton 2000, [ApJS 130, 403](https://ui.adsabs.harvard.edu/abs/2000ApJS..130..403M), eq. 8): a shift of
# ≈ 4.3 Å, more than a resolution element (0.7 Å). The file holds the 40 strongest lines of each atomic species and
# 60 of each molecule, ranked by a rough strength
#
# $$ S = \log\varepsilon_\odot(X) + \log gf - \theta\chi, \qquad \theta = 5040/T = 1.12\ {\rm eV}^{-1}\ (T = 4500\ {\rm K}), $$
#
# i.e. abundance × oscillator strength × Boltzmann factor (solar abundances $\log\varepsilon_\odot$ from Asplund et
# al. 2009, [ARA&A 47, 481](https://ui.adsabs.harvard.edu/abs/2009ARA&A..47..481A)). $S$ ignores ionisation and
# molecule formation, so only compare lines of one species with it.

# %%
print(lines.sort_values('strength', ascending=False).groupby('species').head(1)
      .sort_values('strength', ascending=False).head(12)[['species', 'wave_air', 'wave_vac', 'chi_ev', 'loggf', 'strength']]
      .round(3).to_string(index=False))
# the vacuum - air shift in the H band
print(f"vacuum - air: {np.median(lines.wave_vac - lines.wave_air):.2f} A")
assert 4.0 < np.median(lines.wave_vac - lines.wave_air) < 4.6

# %% [markdown]
# A spectrum is not normalised yet (Chapter 3 does that properly). To compare stars in a narrow window we divide
# each spectrum by its own 90th percentile within the window — a crude local "continuum" that is good enough
# for looking.

# %%
def window_norm(lo, hi):
    """Pixels of the window [lo, hi] A, and the flux of all stars there divided by each star's 90th percentile."""
    w = (wave > lo) & (wave < hi)
    f = np.where(np.isfinite(err[:, w]), flux[:, w], np.nan)      # flagged pixels (sky lines etc.) -> NaN
    return w, f / np.nanpercentile(f, 90, axis=1)[:, None]


lo, hi = 15730, 15800                                    # a window on the blue detector with Mg I, Fe I and molecules
w, fw = window_norm(lo, hi)
i = np.argmin(np.abs(teff - 4500) + 300 * np.abs(logg - 2.0) + 1000 * np.abs(feh))   # a typical giant
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(wave[w], fw[i], 'k', lw=1, drawstyle='steps-mid')
col = {'FE I': 'C0', 'MG I': 'C3', 'SI I': 'C2', 'CN': 'C1', 'CO': 'C4', 'OH': 'C5'}
win = lines[(lines.wave_vac > lo) & (lines.wave_vac < hi) & lines.species.isin(col)]
for sp, d in win.groupby('species'):
    strongest = d[d.strength > d.strength.max() - 1.5]   # within a factor ~30 of the species' strongest line here
    ax.vlines(strongest.wave_vac, 1.04, 1.10, color=col[sp], lw=1.5, label=sp)
ax.set(xlabel='vacuum wavelength [Å]', ylabel='flux / 90th percentile', ylim=(0.35, 1.12),
       title=f'{tr.apogee_id[i]}: {teff[i]:.0f} K, log g {logg[i]:.2f}, [Fe/H] {feh[i]:.2f}')
ax.legend(ncol=6, fontsize=8, loc='lower right')
plt.tight_layout()

# %% [markdown]
# **Figure 1.1.** A 70 Å window of the spectrum of a typical training giant (black), divided by its 90th
# percentile in the window. Ticks mark lines from the APOGEE DR17 line list (vacuum wavelengths), coloured by
# species: for each species only the lines within 1.5 dex of its strongest line in this window. The three deep
# lines at 15 745, 15 753 and 15 770 Å are the Mg I triplet (excitation energy ≈ 5.9 eV); Fe I and Si I lines are
# weaker (flagged pixels, e.g. at sky lines, are blank), and many shallow features are molecular (OH, CN, CO), which are abundant in the cool atmospheres of
# giants. Most features are blends: at R ≈ 22 500 lines closer than ≈ 0.7 Å are not resolved.
#
# ## 1.2 Change one label at a time
#
# Labels of real stars are correlated (giants that are cooler have lower gravity), so we cannot change one label
# while keeping the others exactly fixed. We can choose stars in a narrow range of two labels and order them by
# the third.

# %%
fe_line = lines[lines.species == 'FE I'].sort_values('strength').iloc[-1]   # the strongest Fe I line
lo2, hi2 = fe_line.wave_vac - 12, fe_line.wave_vac + 12
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
sets = [('[Fe/H]', feh, (np.abs(teff - 4600) < 100) & (np.abs(logg - 2.3) < 0.3), 'viridis'),
        (r'$T_{\rm eff}$', teff, (np.abs(feh + 0.2) < 0.1) & (np.abs(logg - 2.3) < 0.4), 'plasma'),
        (r'$\log g$', logg, (np.abs(feh + 0.2) < 0.1) & (np.abs(teff - 4600) < 150), 'cividis')]
for a, (nm, val, sel, cm) in zip(ax, sets):
    w2, fw2 = window_norm(lo2, hi2)
    idx = np.flatnonzero(sel)
    idx = idx[np.argsort(val[idx])][::max(1, len(idx) // 12)]     # up to ~12 stars spread along the label
    c = plt.get_cmap(cm)((val[idx] - val[idx].min()) / np.ptp(val[idx]))
    for j, cc in zip(idx, c):
        a.plot(wave[w2], fw2[j], color=cc, lw=0.8)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(val[idx].min(), val[idx].max()))
    plt.colorbar(sm, ax=a, label=nm)
    a.axvline(fe_line.wave_vac, color='k', ls=':', lw=0.8)
    a.set(xlabel='vacuum wavelength [Å]', title=f'{len(idx)} stars ordered by {nm}')
ax[0].set(ylabel='flux / 90th percentile')
plt.tight_layout()
print(f"strongest Fe I line: {fe_line.wave_vac:.2f} A (vacuum), chi = {fe_line.chi_ev} eV, log gf = {fe_line.loggf}")

# %% [markdown]
# **Figure 1.2.** A 24 Å window around the strongest Fe I line of the list (dotted line, 15 298.7 Å vacuum).
# Left: stars with $T_{\rm eff} = 4600 \pm 100$ K and $\log g = 2.3 \pm 0.3$, coloured by [Fe/H]: every line deepens
# with metallicity. Middle: stars with [Fe/H] $= -0.2 \pm 0.1$ and $\log g = 2.3 \pm 0.4$, coloured by
# $T_{\rm eff}$: cooler stars have deeper lines, mostly from molecules and low-excitation atomic lines. Right: the
# same metallicity and $T_{\rm eff} = 4600 \pm 150$ K, coloured by $\log g$: the effect is subtle and partly
# confused with the remaining spread in temperature. These derivatives — large for [Fe/H] and $T_{\rm eff}$, small
# and blended for $\log g$ — are what The Cannon has to learn, at every pixel.
#
# ## 1.3 One pixel as a function of the labels
#
# Take the pixel at the core of this Fe I line. Its flux is a function of the three labels. In the narrow slice
# of the left panel, it should fall with [Fe/H]; a straight line measures the derivative.

# %%
pix = np.argmin(np.abs(wave - fe_line.wave_vac))              # pixel at the line core
w2, fw2 = window_norm(lo2, hi2)
f_core = fw2[:, np.flatnonzero(w2) == pix][:, 0]                 # its locally normalised flux, every star
e_core = err[:, pix] / np.nanpercentile(flux[:, w2], 90, axis=1)
sel = sets[0][2] & np.isfinite(f_core) & np.isfinite(e_core)
A = np.vstack([np.ones(sel.sum()), feh[sel]]).T                  # straight line: f = a + b [Fe/H]
coef, res, *_ = np.linalg.lstsq(A, f_core[sel], rcond=None)
resid = f_core[sel] - A @ coef
cov = np.linalg.inv(A.T @ A) * resid.var(ddof=2)                 # errors from the scatter of the residuals
slope, slope_err = coef[1], np.sqrt(cov[1, 1])
print(f'pixel {pix} ({wave[pix]:.2f} A): d flux / d[Fe/H] = {slope:.3f} +- {slope_err:.3f} per dex '
      f'({sel.sum()} stars with T_eff 4600 +- 100 K, log g 2.3 +- 0.3)')
# Test: the line core gets deeper with metallicity, significantly (> 5 sigma).
assert slope < 0 and abs(slope / slope_err) > 5

# %% [markdown]
# Now use *all* training stars and all three labels at once. Write the flux of the pixel as a polynomial of the
# labels, shifted to a pivot (the median label) and divided by a scale, so that all labels are numbers of order
# one: $x_k = (\ell_k - p_k)/q_k$. A **linear** model has 4 coefficients,
# $f = \theta_0 + \theta_1 x_{T} + \theta_2 x_{g} + \theta_3 x_{\rm Fe}$; a **quadratic** one adds the 6 products
# $x_T^2, x_T x_g, \dots$ (10 coefficients). Both are linear in the coefficients, so ordinary least squares fits
# them. Each star is weighted by $1/\sigma^2$.

# %%
L = np.vstack([teff, logg, feh]).T
pivot, scale = np.median(L, axis=0), (np.percentile(L, 97.5, axis=0) - np.percentile(L, 2.5, axis=0)) / 2
x = (L - pivot) / scale
k, l = np.triu_indices(3)
X_lin = np.hstack([np.ones((len(x), 1)), x])                  # (N, 4)
X_quad = np.hstack([X_lin, x[:, k] * x[:, l]])                # (N, 10)
ok = np.isfinite(f_core) & np.isfinite(e_core)
fits = {}
for nm, X in [('linear', X_lin), ('quadratic', X_quad)]:
    wsq = 1 / e_core[ok]                                      # weighted least squares = ordinary on rows scaled by 1/sigma
    th = np.linalg.lstsq(X[ok] * wsq[:, None], f_core[ok] * wsq, rcond=None)[0]
    fits[nm] = (th, f_core[ok] - X[ok] @ th)
    print(f'{nm:9s}: rms residual {fits[nm][1].std():.4f}; median flux error {np.median(e_core[ok]):.4f}; '
          f'theta_0 = {th[0]:.3f}, dflux/dx = {th[1:4].round(3)}')

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
pred_q = X_quad[ok] @ fits['quadratic'][0]
sc = ax[0].scatter(feh[ok], f_core[ok], c=teff[ok], s=4, cmap='plasma')
plt.colorbar(sc, ax=ax[0], label=r'$T_{\rm eff}$ [K]')
ax[0].set(xlabel='[Fe/H]', ylabel='flux at the Fe I line core')
ax[1].scatter(pred_q, f_core[ok], c=teff[ok], s=4, cmap='plasma')
ax[1].plot([0.3, 1], [0.3, 1], 'k-', lw=0.8)
ax[1].set(xlabel='quadratic model', ylabel='flux at the Fe I line core')
for nm in fits:
    ax[2].hist(fits[nm][1], bins=np.linspace(-0.1, 0.1, 81), histtype='step', label=f'{nm}: rms {fits[nm][1].std():.4f}')
ax[2].set(xlabel='data − model', ylabel='stars')
ax[2].legend(fontsize=8)
plt.tight_layout()
# Test: the quadratic model describes the pixel better than the linear one (lower residual rms).
assert fits['quadratic'][1].std() < fits['linear'][1].std()

# %% [markdown]
# **Figure 1.3.** The flux at the core of the Fe I line at 15 298.7 Å (locally normalised) for all training
# stars. Left: against [Fe/H], coloured by $T_{\rm eff}$: at fixed metallicity, cooler stars have deeper cores, so
# the scatter about any single trend with [Fe/H] is mostly temperature. Middle: against the prediction of the
# quadratic model in ($T_{\rm eff}$, $\log g$, [Fe/H]); the line is the identity. Right: residuals of the linear and
# the quadratic model. The quadratic model removes the curvature (the response of a strong line saturates as it
# deepens), but the residuals are still far larger than the flux errors (≈ 0.3% at this S/N): either the labels
# are imperfect, or other labels (abundance ratios, microturbulence) matter, or our crude normalisation adds
# scatter. The Cannon calls this excess the **intrinsic scatter** $s_\lambda$ of a pixel.
#
# ## 1.4 Every pixel
#
# Repeat the quadratic fit at every pixel of the window, and plot the coefficients as a function of wavelength:
# $\theta_0$ is the spectrum of a star with the pivot labels, and $\theta_1, \theta_2, \theta_3$ are the changes of
# the flux per unit of scaled $T_{\rm eff}$, $\log g$, [Fe/H] (N15, Fig. 5).

# %%
w3, fw3 = window_norm(15730, 15800)
ew3 = err[:, w3] / np.nanpercentile(flux[:, w3], 90, axis=1)[:, None]
theta_w = np.full((w3.sum(), 10), np.nan)
for j in range(w3.sum()):                                     # one weighted least-squares fit per pixel
    o = np.isfinite(fw3[:, j]) & np.isfinite(ew3[:, j])
    theta_w[j] = np.linalg.lstsq(X_quad[o] / ew3[o, j, None], fw3[o, j] / ew3[o, j], rcond=None)[0]
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax[0].plot(wave[w3], theta_w[:, 0], 'k', lw=1)
ax[0].set(ylabel=r'$\theta_0$ (pivot spectrum)')
for d, nm in zip([1, 2, 3], [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]']):
    ax[1].plot(wave[w3], theta_w[:, d], lw=1, label=r'$\partial f/\partial x$ for ' + nm)
ax[1].axhline(0, color='k', lw=0.5)
for wl in win[win.species == 'MG I'].wave_vac:
    ax[1].axvline(wl, color='C3', lw=0.5, ls=':')
ax[1].set(xlabel='vacuum wavelength [Å]', ylabel='first-order coefficient')
ax[1].legend(fontsize=8)
plt.tight_layout()
mg = [np.argmin(np.abs(wave[w3] - wl)) for wl in [15745.0, 15753.3, 15770.1]]   # the Mg I triplet cores
print('d flux / d x_[Fe/H] at the Mg I cores:', theta_w[mg, 3].round(3), '; T_eff:', theta_w[mg, 1].round(3))

# %% [markdown]
# **Figure 1.4.** A quadratic model fitted pixel by pixel in the window of Figure 1.1. Top: the zeroth-order
# coefficient — the spectrum of a giant with the pivot labels ($T_{\rm eff} \approx 4600$ K, $\log g \approx 2.4$,
# [Fe/H] ≈ −0.3). Bottom: the first-order coefficients, the change of the flux per unit of scaled label (one unit
# is ≈ 0.6–0.9 of the training range). The [Fe/H] coefficient is negative in almost every line: more metals, deeper
# lines. The $T_{\rm eff}$ coefficient is mostly positive (hotter, shallower lines), but not everywhere; the
# $\log g$ coefficient is smaller and changes sign from line to line. The dotted lines mark the Mg I lines: their
# cores deepen with [Fe/H] (with this three-label model, [Mg/Fe] is not a label, so [Fe/H] stands in for Mg too) and
# get shallower with $T_{\rm eff}$. The Cannon is this computation at all 8575 pixels, with a proper continuum
# normalisation and with the noise model of Chapter 2.
#
# ## Summary
#
# - Absorption lines identified with a line list (vacuum wavelengths) show which features respond to which label.
# - At one pixel, the flux is a smooth, slightly curved function of the labels: a quadratic polynomial is a good
#   first description, with an excess scatter beyond the measurement errors.
# - Repeating this fit at every pixel gives the coefficients of The Cannon, which can be read as derivatives of
#   the spectrum with respect to the labels.
#
# **Exercises.**
# 1. Repeat Figure 1.3 at the core of the strongest OH line in the list. Which label dominates?
# 2. Repeat the fit of §1.3 with $\log g$ left out. How much larger is the residual rms, and at which pixels of the
#    window of §1.4?

# %% [markdown]
# ## References
#
# - Asplund et al. 2009, *The Chemical Composition of the Sun*, ARA&A 47, 481. [ADS](https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A)
# - Morton 2000, *Atomic Data for Resonance Absorption Lines. II. Wavelengths Longward of the Lyman Limit for Heavy Elements*, ApJS 130, 403. [ADS](https://ui.adsabs.harvard.edu/abs/2000ApJS..130..403M)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Smith et al. 2021, *The APOGEE Data Release 16 Spectral Line List*, AJ 161, 254. [ADS](https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S)
