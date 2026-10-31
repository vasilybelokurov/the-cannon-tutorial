# %% [markdown]
# # Appendix — Our Cannon against a published implementation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/A_annieslasso.ipynb)
#
# **Goal.** Check the few dozen lines of The Cannon written in Chapter 2 against a published code: *AnniesLasso*
# (the code of *The Cannon 2*, Casey et al. 2016), version 0.2.96 (October 2025), installed with
# `pip install annieslasso==0.2.96`; it is imported as `thecannon`. Both codes are trained on the same APOGEE spectra
# and labels (Chapter 4) and label the same 1500 test stars.
#
# **You will learn**
# 1. to separate a *bug* from a *difference in method* when two codes disagree: first run them the same way, where
#    they must agree to numerical precision, then the way each is meant to be run;
# 2. that two published ways to train the model — ours (Chapter 2) and that of AnniesLasso — give scatters
#    $s_\lambda$ that differ by a factor of two, but labels that differ by much less than their errors.
#
# **Reading the AnniesLasso code** (`thecannon/model.py`, `thecannon/fitting.py`) shows how it trains a model
# without regularisation ($\Lambda = 0$):
# - the labels are scaled as ours, by the median and the 2.5–97.5 percentile range (we pass our scale, half that
#   range, so that the coefficients $\boldsymbol\theta_\lambda$ of the two codes can be compared one to one; the
#   order of the terms in the label vector is the same as ours);
# - at each pixel $\boldsymbol\theta_\lambda$ is fitted with the scatter set to **zero** (weights $1/\sigma^2$),
#   by an L-BFGS minimisation of $\chi^2$ started from the linear-algebra solution;
# - then the scatter is chosen, once, so that the **median** of $r^2/(\sigma^2+s^2)$ over the training stars is 1.
#
# Our `train` (Chapter 2) instead iterates $\boldsymbol\theta_\lambda$ with weights $1/(\sigma^2+s^2)$ and $s_\lambda$
# by maximum likelihood. The test step is the same in both: a non-linear least-squares fit with weights
# $1/(\sigma^2+s^2)$ and the analytic Jacobian, started at the median labels.

# %%
#@@LOADER
#@@NORM
#@@CANNON
#@@MORE
import time
import logging
import io
import contextlib
import subprocess
import types
import scipy.optimize
import sys
try:
    import thecannon as tc
except ImportError:                                              # e.g. in Colab
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'annieslasso==0.2.96', 'six'], check=True)
    import thecannon as tc

logging.getLogger('thecannon').setLevel(logging.ERROR)   # AnniesLasso logs every spectrum at DEBUG level
if not hasattr(np.linalg, 'linalg'):
    np.linalg.linalg = np.linalg   # AnniesLasso 0.2.96 catches np.linalg.linalg.LinAlgError, removed in numpy 2
_fit_pixel = tc.fitting.fit_pixel_fixed_scatter


def fit_pixel_scalar_s2(*args, **kwargs):
    """AnniesLasso 0.2.96 returns s^2 as a length-1 array, which numpy 2 will not store in a scalar slot."""
    theta, s2, meta = _fit_pixel(*args, **kwargs)
    return theta, float(np.squeeze(s2)), meta


tc.fitting.fit_pixel_fixed_scatter = fit_pixel_scalar_s2


def fmin_l_bfgs_b(*args, disp=None, iprint=None, **kwargs):
    """AnniesLasso 0.2.96 passes `disp` (printing only), which scipy >= 1.18 no longer accepts."""
    return scipy.optimize.fmin_l_bfgs_b(*args, **kwargs)


tc.fitting.op = types.SimpleNamespace(fmin=scipy.optimize.fmin, fmin_powell=scipy.optimize.fmin_powell,
                                      leastsq=scipy.optimize.leastsq, fmin_l_bfgs_b=fmin_l_bfgs_b)
labels = load('labels')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES, TEX, UNITS = ['teff', 'logg', 'fe_h'], [r'$T_{\rm eff}$', r'$\log g$', '[Fe/H]'], ['K', 'dex', 'dex']
data = {}
for nm in ['train', 'test']:
    lab = labels[labels['sample'] == nm].reset_index(drop=True)
    f, e = cheb_normalise(*prepare(load(f'spectra_{nm}')), cont)
    data[nm] = (lab[NAMES].to_numpy(), f, e)
L, flux, err = data['train']
Lt, ft, et = data['test']
pivot = np.median(L, axis=0)
scale = (np.percentile(L, 97.5, axis=0) - np.percentile(L, 2.5, axis=0)) / 2
x = (L - pivot) / scale


def to_ivar(f, e):
    """AnniesLasso takes inverse variances; pixels we ignore (err = inf) get ivar = 0 and a finite dummy flux."""
    ivar = np.where(np.isfinite(e), 1 / e**2, 0.0)
    return np.where(ivar > 0, f, 1.0), ivar


print(f'{len(L)} training stars, {len(Lt)} test stars, {flux.shape[1]} pixels, AnniesLasso {tc.__version__}')

# %% [markdown]
# ## A.1 Training both codes
#
# AnniesLasso fits each pixel with an iterative optimiser, ours solves all pixels at once by linear algebra; both
# take about ten seconds. Running AnniesLasso 0.2.96 with numpy 2 and scipy 1.18 needs four small workarounds, all in
# the cells above and below: a keyword for the label scale that it rejects, an exception class that numpy 2 removed,
# a scatter returned as an array of length one, and a printing option that scipy 1.18 removed. None of them changes
# a number.

# %%
fa, ia = to_ivar(flux, err)
vec = tc.vectorizer.PolynomialVectorizer(NAMES, 2)
al = tc.CannonModel(L, fa, ia, vec, regularization=0.0)
# use our scale (half AnniesLasso's); the keyword __scale_labels_function meant for this is rejected by 0.2.96,
# so set the scale and rebuild the design matrix as CannonModel.__init__ does
al._scales = scale
al._design_matrix = vec((L - al._fiducials) / al._scales).T
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):     # AnniesLasso draws a progress bar per pixel
    th_al, s2_al, _ = al.train()
t_al = time.time() - t0
s_al = np.sqrt(s2_al)
t0 = time.time()
th_us, _, s_us = train(x, flux, err)
t_us = time.time() - t0
print(f'training time: AnniesLasso {t_al:.0f} s, ours {t_us:.1f} s')
print('same label vector:', vec.human_readable_label_vector)
assert np.allclose(al._fiducials, pivot) and np.allclose(al._scales, scale)

# %% [markdown]
# ## A.2 Run the same way, the two codes must agree
#
# **Training step.** With the scatter set to zero our `fit_theta` (Chapter 2) solves the same weighted least-squares
# problem that AnniesLasso minimises. Compare the two spectral models at the labels of the training stars, in units
# of the flux errors. The pixels that AnniesLasso skips (too little weight: it sets $\boldsymbol\theta = (1, 0, \dots)$
# and $s^2 = \infty$) are the pixels that our `prepare` gave zero weight; they are left out.
#
# **Test step.** Give our `test` the coefficients and scatter of AnniesLasso; both must then find the same labels,
# unless one of the optimisers stops before the minimum. Where they differ, the $\chi^2$ of the two solutions under
# the same model says which one stopped early.

# %%
V = vectorize(x)
th0, _ = fit_theta(V, flux, err**2)                         # our weighted least squares with s = 0
use = np.isfinite(s2_al)
print(f'pixels used: AnniesLasso {use.sum()}, ours {(th_us[:, 0] != 0).sum()}')
dmod = (V @ (th0[use] - th_al[use]).T) / np.where(np.isfinite(err[:, use]), err[:, use], np.inf)
print(f'|model difference| / sigma: median {np.median(np.abs(dmod)):.1e}, 99.9th percentile '
      f'{np.percentile(np.abs(dmod), 99.9):.1e}, max {np.abs(dmod).max():.1e}')

fta, ita = to_ivar(ft, et)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    lab_al, cov_al, meta_al = al.test(fta, ita)
t_test_al = time.time() - t0
s_for_us = np.where(use, s_al, 0.0)
th_for_us = np.where(use[:, None], th_al, 0.0)                # theta_0 = 0: our test skips these pixels
t0 = time.time()
out = [test(ft[n], et[n], th_for_us, s_for_us) for n in range(len(ft))]
t_test_us = time.time() - t0
lab_same = np.array([o[0] for o in out]) * scale + pivot
d_same = lab_same - lab_al
print(f'test step time: AnniesLasso {t_test_al:.0f} s, ours {t_test_us:.0f} s')
sig_same = np.array([np.sqrt(np.diag(o[1])) for o in out]) * scale
z_same = np.abs(d_same / sig_same).max(axis=1)                  # largest difference over the 3 labels, in sigma


def chi2(n, lab):
    """chi^2 of spectrum n at physical labels lab under the AnniesLasso model (the function minimised by both)."""
    ok = np.isfinite(ft[n]) & np.isfinite(et[n]) & use
    r = (ft[n] - th_for_us @ vectorize((lab - pivot) / scale)[0]) / np.sqrt(et[n]**2 + s_for_us**2)
    return np.sum(r[ok]**2)


far = np.flatnonzero(z_same > 0.1)
dchi = np.array([chi2(n, lab_al[n]) - chi2(n, lab_same[n]) for n in far])
print(f'|ours - AnniesLasso| / formal error, largest of the 3 labels: median {np.median(z_same):.4f}, '
      f'99th percentile {np.percentile(z_same, 99):.3f}')
print(f'{len(far)} stars differ by more than 0.1 sigma; chi^2(AnniesLasso) - chi^2(ours) for them: '
      f'min {dchi.min():.2f}, median {np.median(dchi):.1f}, max {dchi.max():.1f}')

# %% [markdown]
# **Checks.** The spectral models agree to a millionth of the noise. The labels of 99% of the stars agree to a few
# hundredths of their formal errors; for the two that differ by more than 0.1 of their errors, AnniesLasso's solution
# has the higher (or the same) $\chi^2$: its optimiser (`scipy.optimize.leastsq`) stopped before the minimum. (AnniesLasso returns `NaN`
# labels when the optimiser does not move from the starting point; there are none.)

# %%
n_nan = np.isnan(lab_al).any(axis=1).sum()
print('AnniesLasso NaN labels:', n_nan)
assert np.abs(dmod).max() < 1e-3, 'the two least-squares solutions differ'
assert n_nan == 0
assert np.percentile(z_same, 99) < 0.05, 'the test steps disagree'
assert np.all(dchi > -1e-2), 'AnniesLasso found a lower chi^2 than our test step'

# %% [markdown]
# ## A.3 Each code as it is meant to be run
#
# Now compare AnniesLasso ($\boldsymbol\theta$ fitted without scatter, then $s$ from "median $\chi^2 = 1$") with our
# `train` ($\boldsymbol\theta$ and $s$ iterated, $s$ by maximum likelihood). The labels of the 1500 test stars from
# both are compared with each other and with ASPCAP. To see which of the two differences matters, also label the
# test stars with our $\boldsymbol\theta$ and AnniesLasso's $s$, and the other way round.

# %%
def label_all(theta, s):
    """Our test step on all test spectra: physical labels and formal errors."""
    out = [test(ft[n], et[n], theta, s) for n in range(len(ft))]
    assert all(o[4] for o in out)
    return np.array([o[0] for o in out]) * scale + pivot, np.array([np.sqrt(np.diag(o[1])) for o in out]) * scale


lab_us, sig_us = label_all(th_us, s_us)
lab_mix = {'our theta, AL s': label_all(th_us, s_for_us)[0], 'AL theta, our s': label_all(th_for_us, s_us)[0]}
d_codes = lab_us - lab_al
both = use & (th_us[:, 0] != 0)
print(f'scatter, median over pixels: ours {np.median(s_us[both]):.4f}, AnniesLasso {np.median(s_al[both]):.4f}; '
      f'median ratio AnniesLasso / ours (s > 0.002) = {np.median((s_al / s_us)[both & (s_us > 0.002)]):.2f}')
r = flux[:, use] - V @ th_al[use].T
z2 = np.where(np.isfinite(err[:, use]), r**2 / (err[:, use]**2 + s2_al[use]), np.nan)
print(f'at the AnniesLasso scatter, training stars: median chi^2 per pixel {np.median(np.nanmedian(z2, axis=0)):.2f}, '
      f'mean chi^2 per pixel {np.median(np.nanmean(z2, axis=0)):.2f} (medians over pixels)')
rob = lambda d: [scatter_stats(d[:, k])['robust'] for k in range(3)]
print(f"{'robust std of':34s}" + ''.join(f'{n:>10s}' for n in NAMES))
for name, d in [('ours - ASPCAP', lab_us - Lt), ('AnniesLasso - ASPCAP', lab_al - Lt), ('ours - AnniesLasso', d_codes)] + \
               [(f'ours - ({k})', lab_us - v) for k, v in lab_mix.items()]:
    print(f'{name:34s}' + ''.join(f'{v:10.3f}' for v in rob(d)))
print(f"{'ours - AnniesLasso, in formal errors':34s}" + ''.join(f'{v:10.2f}' for v in rob(d_codes / sig_us)))

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ok = both & (s_us > 0) & (s_al > 0)
ax[0].loglog(s_us[ok], s_al[ok], ',', color='k', alpha=0.5)
ax[0].plot([1e-4, 1], [1e-4, 1], 'r--', lw=1)
ax[0].set(xlim=(1e-4, 0.3), ylim=(1e-4, 0.3), xlabel=r'$s_\lambda$, ours (maximum likelihood)',
          ylabel=r'$s_\lambda$, AnniesLasso (median $\chi^2=1$)', title='(a) scatter at each pixel')
for k, c in enumerate(['C0', 'C1', 'C2']):
    ax[1].hist(d_codes[:, k] / sig_us[:, k], bins=np.linspace(-5, 5, 81), histtype='step', color=c, lw=1.5,
               label=TEX[k])
ax[1].set(xlabel=r'(ours $-$ AnniesLasso) / formal error', ylabel='test stars', title='(b) the two codes')
ax[1].legend()
k = 2
for lab, c, nm in [(lab_us, 'C3', 'ours'), (lab_al, 'C0', 'AnniesLasso')]:
    ax[2].hist(lab[:, k] - Lt[:, k], bins=np.linspace(-0.3, 0.3, 61), histtype='step', color=c, lw=1.5, label=nm)
ax[2].set(xlabel='[Fe/H] $-$ [Fe/H]$_{\\rm ASPCAP}$ [dex]', ylabel='test stars', title='(c) against ASPCAP')
ax[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure A.1.** Our Cannon and AnniesLasso 0.2.96 trained on the same 1703 APOGEE giants and applied to the same
# 1500 test stars (S/N 50–100). **(a)** The scatter $s_\lambda$ at every pixel: AnniesLasso's rule (median
# $\chi^2 = 1$ over the training stars) against our maximum-likelihood value; the dashed line is equality.
# AnniesLasso's scatter is smaller at every pixel, by a factor of about two. **(b)** The difference between the
# labels from the two codes, in units of the formal error of our labels (which Chapter 6 showed to be 2–4 times
# smaller than the real errors); the differences are systematic, they depend on where the star lies in label space.
# **(c)** [Fe/H] from each code minus ASPCAP: the two distributions nearly coincide.
#
# Why the scatters differ: the residuals of the training stars at a pixel have heavy tails (a few stars that the
# three labels do not describe). The median ignores them; the likelihood does not, and gives a larger $s$. At
# AnniesLasso's scatter the median $\chi^2$ is 1 by construction, but the mean $\chi^2$ is about 3 (printed above).
# For Gaussian residuals the rule "median $\chi^2 = 1$" would instead *overestimate* $s$: the median of $\chi^2$
# with one degree of freedom is 0.45, not 1.

# %% [markdown]
# **Checks.** The two codes agree with each other much better than either agrees with ASPCAP (the difference is
# about a fifth of the scatter about ASPCAP). Most of the difference comes from $\boldsymbol\theta$, not from $s$:
# swapping only the scatter moves the labels half as much as swapping only the coefficients. On these data ours is
# slightly closer to ASPCAP in every label (by 4–10%): weighting the training stars by $1/(\sigma^2+s^2)$ rather
# than $1/\sigma^2$ stops the few highest-S/N stars from dominating the fit at pixels where the labels do not
# describe the spectra to within their noise.

# %%
ru, ra, rc = map(np.array, (rob(lab_us - Lt), rob(lab_al - Lt), rob(d_codes)))
assert np.all(s_al[both] <= s_us[both])
assert np.all(rc < 0.3 * ru), 'the two codes differ by more than a third of the scatter about ASPCAP'
assert np.all(ru <= ra), 'ours is not closer to ASPCAP'
assert np.all(np.array(rob(lab_us - lab_mix['our theta, AL s'])) < np.array(rob(lab_us - lab_mix['AL theta, our s'])))

# %% [markdown]
# ## A.4 The regularised training step
#
# Chapter 10 added L1 regularisation to our code: at each pixel, minimise
# $\chi^2 + \Lambda \sum_{d \ge 1} |\theta_d|$ (C16, eq. 2 and 5). AnniesLasso minimises the same function, with
# weights $1/\sigma^2$ (no scatter), by L-BFGS-B — an optimiser for smooth functions — using $\mathrm{sign}(\theta)$ as
# the gradient of $|\theta|$. Ours uses ADMM. Set up the same problem in both: the scale of C16 ($f = 2$ times the
# 2.5–97.5 percentile range, i.e. four times our usual scale), their $\Lambda = 10^3$, weights $1/\sigma^2$. Compare
# the value of the function each code reaches at every pixel, and how many coefficients each sets exactly to zero.

# %%
LAM, scale_c16 = 1e3, 4 * scale
al_l1 = tc.CannonModel(L, fa, ia, vec, regularization=LAM)
al_l1._scales = scale_c16
al_l1._design_matrix = vec((L - al_l1._fiducials) / scale_c16).T
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    th_al1, _, _ = al_l1.train()
t_al1 = time.time() - t0
V16 = vectorize((L - pivot) / scale_c16)
t0 = time.time()
th_us1, _ = fit_theta_l1(V16, flux, err**2, LAM, tol=1e-9)
t_us1 = time.time() - t0
w16 = np.where(np.isfinite(err), 1 / err**2, 0.0)
f16 = np.where(w16 > 0, flux, 0.0)


def objective(th):
    """chi^2 + Lambda * sum |theta_{1:}| at every pixel (the function both codes minimise)."""
    return np.sum(w16 * (f16 - V16 @ th.T)**2, axis=0) + LAM * np.abs(th[:, 1:]).sum(axis=1)


use1 = np.isfinite(th_al1).all(axis=1) & (th_us1[:, 0] != 0)
dF = (objective(th_al1) - objective(th_us1))[use1]
dmod1 = (V16 @ (th_al1[use1] - th_us1[use1]).T) / np.where(np.isfinite(err[:, use1]), err[:, use1], np.inf)
print(f'training time: AnniesLasso {t_al1:.0f} s, ours {t_us1:.1f} s')
print(f'objective, AnniesLasso - ours, over {use1.sum()} pixels: min {dF.min():.2e}, median {np.median(dF):.2e}, '
      f'max {dF.max():.2e}')
print(f'coefficients exactly zero: AnniesLasso {np.mean(th_al1[use1, 1:] == 0):.3f}, ours {np.mean(th_us1[use1, 1:] == 0):.3f}')
print(f'where ours is zero, AnniesLasso |theta|: median {np.median(np.abs(th_al1[use1, 1:][th_us1[use1, 1:] == 0])):.1e}')
big = np.abs(dmod1).max(axis=0)                              # largest model difference over the training stars
differ = big > 0.1
print(f'pixels where the models differ by more than 0.1 sigma for some star: {differ.sum()} of {use1.sum()}; '
      f'there AnniesLasso ends higher by {dF[differ].min():.2f} to {dF[differ].max():.0f}')
print(f'pixels where AnniesLasso ends higher by more than 1: {np.sum(dF > 1)}')

# %% [markdown]
# **Checks.** At most pixels both codes reach the same minimum (the median difference is $\sim10^{-7}$ in a function
# of order $10^3$). At 8% of the pixels the two spectral models differ by more than a tenth of the noise for at least
# one training star, and at every one of them AnniesLasso's function value is the higher: its L-BFGS stopped before
# the minimum (for an L1 problem the model spectrum of the training stars at the minimum is unique). Our ADMM never
# ends higher. A lower value shows that AnniesLasso's solution is worse, not that ours is exactly optimal; the
# optimality of ours is tested in `tests/test_cannon.py`. This check uses three labels and no scatter; the
# twelve-label fits of Chapter 10 are covered only by those unit tests. Where our solution has
# exact zeros, AnniesLasso's coefficients are $\sim10^{-13}$: numerically zero, but an optimiser for smooth
# functions never sets them exactly to zero, so sparsity must be read with a threshold.

# %%
assert dF.min() > -1e-3, 'AnniesLasso found a lower minimum than our ADMM'
assert np.median(np.abs(dF)) < 1e-3 and differ.mean() < 0.15
assert np.all(dF[differ] > 0.05), 'the models differ at a pixel where both reached the same minimum'
zero_us = th_us1[use1, 1:] == 0
assert zero_us.mean() > 0.05 and np.median(np.abs(th_al1[use1, 1:][zero_us])) < 1e-8

# %% [markdown]
# ## Summary
#
# - Run the same way, our code and AnniesLasso 0.2.96 agree: the spectral models to $10^{-6}$ of the noise, the labels
#   of 99% of the test stars to a few hundredths of their formal errors. Where the labels differ more (2 of the 1500
#   stars), AnniesLasso's optimiser stopped early, at a higher $\chi^2$ (or at the same $\chi^2$ in a flat valley). The implementation of Chapter 2 has no
#   bug that this comparison can see.
# - Run as designed, the codes differ: AnniesLasso fits $\boldsymbol\theta$ without the scatter and sets $s$ by
#   "median $\chi^2 = 1$", we iterate both with $s$ by maximum likelihood. Its scatter is half ours; its labels differ
#   from ours by about a fifth of the scatter about ASPCAP, mostly because of the weights in the fit of
#   $\boldsymbol\theta$, and are slightly further from ASPCAP.
# - With L1 regularisation (A.4) both codes reach the same minimum at most pixels; at 8% of the pixels
#   AnniesLasso's optimiser stops early, and only there do the models differ. Ours sets coefficients exactly to zero, AnniesLasso's leaves them at
#   $\sim10^{-13}$.
# - Running AnniesLasso 0.2.96 with numpy 2 and scipy 1.18 needed four small workarounds (no change to any number).
#
# ## References
#
# - Casey, A. R., Hogg, D. W., Ness, M., et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for
#   detailed chemical abundance analyses*, [arXiv:1603.03040](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
#   — the code: [AnniesLasso on GitHub](https://github.com/andycasey/AnniesLasso),
#   [annieslasso on PyPI](https://pypi.org/project/annieslasso/), [documentation](https://annieslasso.readthedocs.io).
# - Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, *The Cannon: a data-driven approach to
#   stellar label determination*, [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N).
