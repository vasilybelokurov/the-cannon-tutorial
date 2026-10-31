# %% [markdown]
# # Chapter 2 — The Cannon on synthetic spectra
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/02_cannon_toy.ipynb)
#
# **Goal.** Write The Cannon (Ness et al. 2015, [ApJ 808, 16](https://doi.org/10.1088/0004-637X/808/1/16);
# hereafter N15) from scratch and check it where we know the right answer: on synthetic spectra drawn from a known
# model. Chapter 1 fitted the flux of *one* pixel as a function of the labels; The Cannon does this at every pixel
# at once, with a noise model, and then turns the fit around to measure the labels of new stars.
#
# **You will learn**
# 1. the model: at every pixel the flux is a quadratic polynomial of the (scaled) labels, plus Gaussian noise
#    whose variance is the measurement variance plus an intrinsic **scatter** $s_\lambda^2$;
# 2. the **training step** (labels known, coefficients unknown): weighted linear least squares at each pixel,
#    and the maximum-likelihood scatter — and why the likelihood needs its $\ln V$ term;
# 3. the **test step** (coefficients known, labels unknown): non-linear least squares over all pixels of one star,
#    with formal errors from the Jacobian;
# 4. how to check an estimator with **pulls** (error-normalised residuals), and what noisy training labels do.
#
# No data are downloaded in this chapter.

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

rng = np.random.default_rng(42)           # fixed seed: every run of this notebook gives the same numbers

# %% [markdown]
# ## 2.1 The model
#
# Star $n$ has labels $\boldsymbol\ell_n$ (here $T_{\rm eff}$, $\log g$, [Fe/H]) and a continuum-normalised
# spectrum $f_{n\lambda}$ with known measurement errors $\sigma_{n\lambda}$. N15 (their eq. 2–4) assume that at
# every pixel $\lambda$
#
# $$ f_{n\lambda} = \boldsymbol\theta_\lambda^{\sf T} \cdot \mathbf v(\boldsymbol\ell_n) + \varepsilon_{n\lambda},
#    \qquad \varepsilon_{n\lambda} \sim \mathcal N\!\left(0,\; \sigma_{n\lambda}^2 + s_\lambda^2\right), $$
#
# where $\mathbf v(\boldsymbol\ell)$ is the **label vector** and $\boldsymbol\theta_\lambda$ the coefficients of
# pixel $\lambda$. The scatter $s_\lambda$ absorbs whatever the model does not describe at that pixel (labels that
# are missing, imperfect training labels, a too simple polynomial). For $K$ labels the quadratic label vector is
#
# $$ \mathbf v(\mathbf x) = \big[\,1,\; x_1, \dots, x_K,\; x_1^2,\; x_1 x_2,\; \dots,\; x_K^2\,\big], $$
#
# with $D = 1 + K + K(K+1)/2$ entries (10 for $K = 3$). The labels enter **scaled**,
# $x_k = (\ell_k - p_k)/q_k$, with a pivot $p_k$ (the median of the training set) and a scale $q_k$ (half the
# 2.5–97.5 percentile range; Casey et al. 2016, [arXiv:1603.03040](https://arxiv.org/abs/1603.03040), eq. 4).
# Without scaling, $T_{\rm eff}^2 \approx 2 \times 10^7$ and $[{\rm Fe/H}]^2 \approx 0.1$ would sit in one matrix and
# the least-squares problem would be badly conditioned. With the pivot at the centre of the training set,
# $\theta_{\lambda 0}$ is the spectrum of a typical star (the **baseline spectrum**) and $\theta_{\lambda k}$ are the
# derivatives $\partial f_\lambda / \partial x_k$ there.

# %%
def vectorize(x):
    """Quadratic label vector v(x) for scaled labels x of shape (N, K): returns shape (N, D).

    Columns: 1, x_1..x_K, then the products x_k x_l with k <= l (np.triu_indices lists those pairs).
    """
    x = np.atleast_2d(x)
    k, l = np.triu_indices(x.shape[1])                      # e.g. K = 2: k = [0, 0, 1], l = [0, 1, 1]
    return np.hstack([np.ones((len(x), 1)), x, x[:, k] * x[:, l]])


def dvectorize(x):
    """Derivative dv/dx of the label vector of ONE star (x of shape (K,)): returns shape (D, K)."""
    K = len(x)
    k, l = np.triu_indices(K)
    J = np.zeros((1 + K + len(k), K))
    J[1:1 + K] = np.eye(K)                                  # d x_k / d x_j = delta_kj
    for i, (a, b) in enumerate(zip(k, l)):                  # d (x_a x_b) / d x_j = delta_aj x_b + x_a delta_bj
        J[1 + K + i, a] += x[b]
        J[1 + K + i, b] += x[a]
    return J


names = ['T_eff', 'log g', '[Fe/H]']
x_demo = np.array([[0.5, -1.0, 2.0]])
print('v(x) for x =', x_demo[0], ':', vectorize(x_demo)[0])
# check the analytic derivative against finite differences (a wrong Jacobian silently ruins the test step)
eps = 1e-6
num = np.array([(vectorize(x_demo + eps * np.eye(3)[j]) - vectorize(x_demo - eps * np.eye(3)[j]))[0] / (2 * eps)
                for j in range(3)]).T                       # shape (D, K), like dvectorize
assert np.allclose(num, dvectorize(x_demo[0]), atol=1e-8)

# %% [markdown]
# ## 2.2 A synthetic survey with a known answer
#
# We invent a "true" model with 600 pixels: a baseline spectrum with absorption lines, and first- and second-order
# coefficients that are non-zero only inside the lines (as in real spectra, where the continuum barely depends on
# the labels). The scatter is $s_\lambda = 0.01$ on the first half of the pixels and exactly zero on the second
# half, so that we can test both cases. From this model we draw
#
# - a **training set** of 1000 stars at S/N = 100 ($\sigma_{n\lambda} = 0.01$) with labels spread uniformly over a
#   red-giant range (3800–5300 K, 0.8–3.6 dex, −1.5–0.4 dex), and
# - a **test set** of 300 new stars at S/N = 30.

# %%
P, K = 600, 3
lo, hi = np.array([3800, 0.8, -1.5]), np.array([5300, 3.6, 0.4])      # label ranges [K, dex, dex]
D = 1 + K + K * (K + 1) // 2                                            # 10 coefficients per pixel

# 40 Gaussian absorption lines at random pixel positions, widths 1.5-3 pixels
pix = np.arange(P)
centres, widths = rng.uniform(10, P - 10, 40), rng.uniform(1.5, 3, 40)
profiles = np.exp(-0.5 * ((pix[None, :] - centres[:, None]) / widths[:, None])**2)   # (40, P), peak 1
theta_true = np.zeros((P, D))
theta_true[:, 0] = 1 - rng.uniform(0.1, 0.5, 40) @ profiles             # baseline: continuum 1 minus lines
# every line responds to every label with its own random amplitude (linear and quadratic terms), so that
# coefficients are large inside lines and zero in the continuum
for d in range(1, D):
    theta_true[:, d] = rng.normal(0, 0.06 if d <= K else 0.02, 40) @ profiles
s_true = np.where(pix < P // 2, 0.01, 0.0)                              # scatter: 0.01 or exactly 0


def draw(n, snr):
    """n stars with uniform random labels; flux from the true model plus noise of variance (1/snr)^2 + s^2."""
    labels = rng.uniform(lo, hi, (n, K))
    x = (labels - (lo + hi) / 2) / ((hi - lo) / 2)                       # true scaled labels in [-1, 1]
    err = np.full((n, P), 1 / snr)
    flux = vectorize(x) @ theta_true.T + rng.normal(size=(n, P)) * np.sqrt(err**2 + s_true**2)
    return labels, flux, err


lab_tr, flux_tr, err_tr = draw(1000, 100)
lab_te, flux_te, err_te = draw(300, 30)

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(pix, theta_true[:, 0], 'k', lw=1, label=r'baseline $\theta_0$')
for n in range(3):
    ax[0].plot(pix, flux_tr[n] + 0.4 * (n + 1), lw=0.6, label=f'training star {n}, S/N 100 (offset +{0.4 * (n + 1):.1f})')
ax[0].set(ylabel='normalised flux')
ax[0].legend(fontsize=8, ncol=2, loc='lower left')
for d, nm in enumerate(names, start=1):
    ax[1].plot(pix, theta_true[:, d], lw=0.8, label=rf'$\theta_{{{nm}}}$ (true)')
ax[1].axvspan(0, P // 2, color='0.9', zorder=-1)
ax[1].text(5, 0.12, r'$s_\lambda = 0.01$', fontsize=9)
ax[1].text(P // 2 + 5, 0.12, r'$s_\lambda = 0$', fontsize=9)
ax[1].set(xlabel='pixel', ylabel='first-order coefficient')
ax[1].legend(fontsize=8, ncol=3)
plt.tight_layout()

# %% [markdown]
# **Figure 2.1.** The synthetic survey. Top: the true baseline spectrum $\theta_{\lambda 0}$ (black; 40 absorption
# lines of depth 0.1–0.5 on a continuum of 1) and the noisy spectra of three training stars (S/N = 100, offset
# vertically by 0.4, 0.8 and 1.2). Bottom: the true first-order coefficients — how the flux of each pixel changes
# per unit of scaled $T_{\rm eff}$, $\log g$ and [Fe/H]. They are non-zero only inside lines, with different signs
# and amplitudes for different lines; a pixel in a line therefore "knows" about a combination of the labels, and
# only the ensemble of pixels can separate them. The grey half has an intrinsic scatter $s_\lambda = 0.01$, the
# white half none.
#
# ## 2.3 The training step
#
# In the training step the labels of the training stars are known and we fit, independently at each pixel, the
# $D$ coefficients $\boldsymbol\theta_\lambda$ and the scatter $s_\lambda$ by maximising the Gaussian likelihood
# (N15 eq. 4–5)
#
# $$ \ln \mathcal L_\lambda(\boldsymbol\theta_\lambda, s_\lambda) = -\frac12 \sum_{n=1}^{N}
#    \left[ \frac{(f_{n\lambda} - \boldsymbol\theta_\lambda^{\sf T} \mathbf v_n)^2}{V_{n\lambda}}
#    + \ln V_{n\lambda} \right], \qquad V_{n\lambda} = \sigma_{n\lambda}^2 + s_\lambda^2 . $$
#
# - **At fixed $s_\lambda$** this is weighted linear least squares with weights $w_{n\lambda} = 1/V_{n\lambda}$:
#   $\boldsymbol\theta_\lambda = (\mathbf V^{\sf T} \mathbf W_\lambda \mathbf V)^{-1} \mathbf V^{\sf T}
#   \mathbf W_\lambda \mathbf f_\lambda$, where $\mathbf V$ is the $N \times D$ matrix whose rows are the label
#   vectors (the design matrix) and $\mathbf W_\lambda$ the diagonal matrix of weights. Its inverse
#   $(\mathbf V^{\sf T} \mathbf W_\lambda \mathbf V)^{-1}$ is the covariance of $\boldsymbol\theta_\lambda$.
# - **At fixed $\boldsymbol\theta_\lambda$** the likelihood is a function of the single number $s_\lambda$; we
#   maximise it on a grid of values.
#
# We alternate the two a few times. All pixels are solved at once with numpy: the $D \times D$ matrices
# $\mathbf V^{\sf T} \mathbf W_\lambda \mathbf V$ of all pixels are built with one matrix product.

# %%
def fit_theta(V, flux, var):
    """Weighted least squares at every pixel: coefficients theta (P, D) and their covariances (P, D, D).

    V: design matrix (N, D); flux, var: (N, P); the weights are 1/var.
    """
    w = 1 / var
    N, Dv = V.shape
    # A_p = sum_n w_np v_n v_n^T for every pixel p: the outer products v_n v_n^T, flattened to D*D numbers per
    # star (shape (N, D*D)), times the weights (N, P), give (D*D, P); reshape to (P, D, D).
    A = ((V[:, :, None] * V[:, None, :]).reshape(N, Dv * Dv).T @ w).T.reshape(-1, Dv, Dv)
    b = (V.T @ (w * flux)).T                                # sum_n w_np f_np v_n, shape (P, D)
    cov = np.linalg.inv(A)                                  # batched inverse of P matrices of size D x D
    return np.einsum('pde,pe->pd', cov, b), cov             # theta_p = A_p^-1 b_p for every pixel


S_GRID = np.r_[0, np.logspace(-4, 0, 41)]                   # candidate scatters: 0, then 1e-4 ... 1 in 0.1-dex steps


def fit_scatter(resid, err2):
    """Maximum-likelihood scatter s at every pixel for fixed residuals (N, P): the best value on the grid S_GRID,
    refined by a parabola through it and its two neighbours (in log s)."""
    nll = np.array([np.sum(resid**2 / (err2 + s**2) + np.log(err2 + s**2), axis=0) for s in S_GRID])  # (grid, P)
    return refine(nll)


def refine(nll):
    """Grid minimum of nll (grid, P), refined by a parabola in log s; s = 0 (grid index 0) is kept as is."""
    j = np.clip(np.argmin(nll, axis=0), 2, len(S_GRID) - 2)        # neighbours j-1, j+1 exist on the log grid
    p = np.arange(nll.shape[1])
    a, b, c = nll[j - 1, p], nll[j, p], nll[j + 1, p]
    step = np.log(S_GRID[2] / S_GRID[1])                            # constant step in ln s
    shift = np.clip(0.5 * (a - c) / np.where(a - 2 * b + c > 0, a - 2 * b + c, np.inf), -1, 1)   # vertex offset in grid steps
    s = S_GRID[j] * np.exp(shift * step)
    return np.where(np.argmin(nll, axis=0) == 0, 0.0, s)


def train(x, flux, err, n_iter=3):
    """Training step. x: scaled labels (N, K); flux, err: (N, P). Returns theta (P, D), its covariance, s (P,)."""
    V, err2 = vectorize(x), err**2
    s = np.zeros(flux.shape[1])
    for _ in range(n_iter):                                 # alternate: theta at fixed s, then s at fixed theta
        theta, cov = fit_theta(V, flux, err2 + s**2)
        s = fit_scatter(flux - V @ theta.T, err2)
    theta, cov = fit_theta(V, flux, err2 + s**2)            # final theta for the final s
    return theta, cov, s


# label scaling from the training set (pivot = median, scale = half the 2.5-97.5 percentile range)
pivot = np.median(lab_tr, axis=0)
scale = (np.percentile(lab_tr, 97.5, axis=0) - np.percentile(lab_tr, 2.5, axis=0)) / 2
to_x = lambda lab: (lab - pivot) / scale                    # labels -> scaled labels
to_lab = lambda x: x * scale + pivot                        # and back
t0 = time.time()
theta, cov, s = train(to_x(lab_tr), flux_tr, err_tr)
print(f'training: {time.time() - t0:.2f} s for {len(lab_tr)} stars x {P} pixels; pivot {pivot.round(2)}, scale {scale.round(2)}')

# %% [markdown]
# Our scaling differs from the one used to generate the data (the training set's median and percentiles instead
# of the exact centre and half-range of the boxes), so the fitted $\boldsymbol\theta$ cannot be compared with
# `theta_true` coefficient by coefficient. What can be compared is the **model spectrum** at any set of labels,
# which must not depend on how the labels are scaled. We predict the noise-free spectra of the 300 test stars at
# their true labels with both models, and compare the difference with the uncertainty of the prediction,
# $\sigma_{\rm pred}^2 = \mathbf v^{\sf T} \, {\rm Cov}(\boldsymbol\theta_\lambda)\, \mathbf v$. If the training
# step and its errors are right, the **pulls** (difference / $\sigma_{\rm pred}$) are standard normal.

# %%
v_te = vectorize(to_x(lab_te))                                          # (300, D), our scaling
x_gen = (lab_te - (lo + hi) / 2) / ((hi - lo) / 2)                       # the scaling used to make the data
pred = v_te @ theta.T                                                    # our model, noise-free, (300, P)
truth = vectorize(x_gen) @ theta_true.T
sig_pred = np.sqrt(np.einsum('nd,pde,ne->np', v_te, cov, v_te))           # v^T Cov v for every star and pixel
pull = (pred - truth) / sig_pred

# the scatter: fitted vs true
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(pull.ravel(), bins=np.linspace(-5, 5, 101), density=True, histtype='step', label='pulls (300 stars x 600 pixels)')
g = np.linspace(-5, 5, 200)
ax[0].plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), 'k--', label='standard normal')
ax[0].set(xlabel='(model − truth) / predicted error', ylabel='density')
ax[0].legend(fontsize=8)
ax[1].plot(pix, s, '.', ms=3, label=r'fitted $s_\lambda$')
ax[1].plot(pix, s_true, 'k-', lw=1, label=r'true $s_\lambda$')
ax[1].set(xlabel='pixel', ylabel=r'scatter $s_\lambda$', ylim=(-0.001, 0.02))
ax[1].legend(fontsize=8)
plt.tight_layout()
inside = np.mean(np.abs(pull) < 3)
s_ratio = np.median(s[pix < P // 2] / 0.01)
print(f'pull std {pull.std():.3f}; fraction within 3 sigma {inside:.4f} (Gaussian: 0.9973)')
print(f'scatter where s = 0.01: median fitted / true = {s_ratio:.3f}; where s = 0: 95th percentile of fitted s = '
      f'{np.percentile(s[pix >= P // 2], 95):.4f} (noise sigma 0.01)')
# Tests: the errors of the trained model are right (pull std 1 within 10%, >= 99% within 3 sigma); the scatter is
# recovered within 10% where it is as large as the noise, and stays well below the noise (< 0.3 sigma) where it is 0.
assert 0.9 < pull.std() < 1.1 and inside > 0.99
assert 0.9 < s_ratio < 1.1 and np.percentile(s[pix >= P // 2], 95) < 0.003

# %% [markdown]
# **Figure 2.2.** Checking the training step. Left: pulls of the trained model, (model − true) noise-free flux
# divided by the predicted error $\sqrt{\mathbf v^{\sf T}{\rm Cov}(\boldsymbol\theta)\mathbf v}$, for the 300
# test stars at all 600 pixels (blue), and the standard normal density (dashed). The two agree: the coefficients
# are unbiased and their covariance is right. Right: the fitted scatter at each pixel (dots) and the truth (line).
# Where $s_\lambda = 0.01$ (as large as the measurement noise) it is recovered to a few per cent (the grid of 41
# values per four decades is refined by a parabola through the best grid point and its neighbours); where
# $s_\lambda = 0$ the fit returns zero or small values, because a scatter much smaller than the noise cannot be
# measured: $s_\lambda^2 = 10^{-6}$ changes the variance $\sigma^2 = 10^{-4}$ by only 1%.
#
# **Why the $\ln V$ term matters.** The first term of $\ln\mathcal L$ alone, $\sum r^2/(\sigma^2+s^2)$, always
# prefers a larger $s$ — it would make every residual look small. The $\ln V$ term (the normalisation of the
# Gaussian) penalises a large variance. Only their sum has a maximum at the right place. Figure 2.3 shows this for
# one pixel.

# %%
p0 = np.argmax(np.abs(theta_true[:P // 2, 1]))              # a pixel in a strong line with s_true = 0.01
r = flux_tr[:, p0] - vectorize(to_x(lab_tr)) @ theta[p0]
sg = np.logspace(-4, -0.5, 200)
term1 = np.array([np.sum(r**2 / (0.01**2 + x**2)) for x in sg])
term2 = len(r) * np.log(0.01**2 + sg**2)                   # sum over the N stars of ln V (V is the same for all)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(sg, term1 - term1.min(), label=r'$\sum r^2/V$ only')
ax.plot(sg, term1 + term2 - (term1 + term2).min(), label=r'$\sum [r^2/V + \ln V]$ = $-2\ln\mathcal{L}$ + const')
ax.axvline(0.01, color='k', ls=':', label='true $s$')
ax.set(xscale='log', yscale='symlog', xlabel=r'scatter $s$', ylabel='objective − its minimum (symlog)', ylim=(0, 1e5))
ax.legend(fontsize=8)
plt.tight_layout()
s_best = sg[np.argmin(term1 + term2)]
print(f'pixel {p0}: maximum-likelihood s = {s_best:.4f}; minimising sum r^2/V alone runs to the largest s on the grid')
assert abs(s_best / 0.01 - 1) < 0.15 and np.argmin(term1) == len(sg) - 1

# %% [markdown]
# **Figure 2.3.** The objective of the scatter fit at one pixel (pixel in a strong line, true $s = 0.01$, 1000
# training stars with $\sigma = 0.01$), as a function of the trial scatter $s$. Orange: $-2\ln\mathcal L$, with a
# clear minimum at the true value (dotted). Blue: the residual term alone, which decreases monotonically: without
# the normalisation of the Gaussian the "best" scatter would be infinite. (Vertical axis linear below 1,
# logarithmic above.)
#
# ## 2.4 The test step
#
# Now the model ($\boldsymbol\theta_\lambda$, $s_\lambda$) is fixed and the labels of a new star are unknown.
# We maximise the same likelihood, now over the $K$ labels of one star and summed over all its pixels (N15 eq. 7):
#
# $$ \hat{\mathbf x}_m = \arg\min_{\mathbf x} \sum_\lambda \frac{\big(f_{m\lambda} - \boldsymbol\theta_\lambda^{\sf T}\mathbf v(\mathbf x)\big)^2}{\sigma_{m\lambda}^2 + s_\lambda^2}. $$
#
# (The $\ln V$ term is now a constant.) Because $\mathbf v$ is quadratic in $\mathbf x$ this is a *non-linear*
# least-squares problem, which `scipy.optimize.least_squares` solves from a starting point, here the pivot
# ($\mathbf x = 0$). We give it the analytic Jacobian of the residuals,
# $\partial r_\lambda / \partial x_k = -\boldsymbol\theta_\lambda^{\sf T}\, \partial\mathbf v/\partial x_k \,/\sqrt{V_\lambda}$.
# Near the minimum the covariance of the labels is $(\mathbf J^{\sf T}\mathbf J)^{-1}$: the **formal error**. It
# assumes that the model is exact; the formal error therefore ignores the uncertainty of $\boldsymbol\theta$ and
# any model imperfection (Chapter 5 compares it with the real scatter).

# %%
def test(flux, err, theta, s, x0=None):
    """Test step for ONE spectrum. Returns scaled labels (K,), their covariance (K, K), chi^2, the number of pixels,
    and whether the optimiser converged.

    Pixels with non-finite flux or error are ignored; x0 is the starting point (default: the pivot, x = 0).
    """
    K = (np.sqrt(8 * theta.shape[1] + 1) - 3) / 2              # invert D = 1 + K + K(K+1)/2
    K = int(round(K))
    ok = np.isfinite(flux) & np.isfinite(err)
    w = 1 / np.sqrt(err[ok]**2 + s[ok]**2)                      # 1/sqrt(V): residuals in units of their sigma
    th, f = theta[ok], flux[ok]
    res = lambda x: (f - th @ vectorize(x)[0]) * w               # (number of pixels,)
    jac = lambda x: -(th @ dvectorize(x)) * w[:, None]           # (number of pixels, K)
    r = least_squares(res, np.zeros(K) if x0 is None else x0, jac=jac, method='lm')
    # r.jac is the Jacobian at the optimum; pinv instead of inv: no crash if a label is not constrained by the data.
    # r.success tells whether the optimiser converged: check it on real data.
    return r.x, np.linalg.pinv(r.jac.T @ r.jac), np.sum(r.fun**2), ok.sum(), r.success


t0 = time.time()
out = [test(flux_te[m], err_te[m], theta, s) for m in range(len(lab_te))]
print(f'test step: {time.time() - t0:.2f} s for {len(lab_te)} stars')
lab_fit = to_lab(np.array([o[0] for o in out]))                 # back to physical labels
lab_err = np.array([np.sqrt(np.diag(o[1])) for o in out]) * scale
chi2nu = np.array([o[2] / (o[3] - K) for o in out])            # chi^2 per degree of freedom

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
units = ['K', 'dex', 'dex']
for k in range(K):
    ax[0, k].errorbar(lab_te[:, k], lab_fit[:, k], lab_err[:, k], fmt='.', ms=3, lw=0.5)
    ax[0, k].plot([lo[k], hi[k]], [lo[k], hi[k]], 'k-', lw=0.8)
    d = lab_fit[:, k] - lab_te[:, k]
    ax[0, k].set(xlabel=f'true {names[k]} [{units[k]}]', ylabel=f'fitted {names[k]} [{units[k]}]')
    ax[0, k].set_title(f'bias {d.mean():.2g}, rms {d.std():.2g}\nmedian formal error {np.median(lab_err[:, k]):.2g}', fontsize=10)
    ax[1, k].hist(d / lab_err[:, k], bins=np.linspace(-4, 4, 33), density=True, histtype='step')
    ax[1, k].plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), 'k--')
    ax[1, k].set(xlabel=f'pull of {names[k]}', xlim=(-4, 4))
plt.tight_layout()
pulls = (lab_fit - lab_te) / lab_err
print('pull std per label:', pulls.std(0).round(3), '; median chi2/nu:', np.median(chi2nu).round(3))
# Tests: the formal errors are right (pull std 1 within 10%, 300 stars: sampling error of the std ~ 0.04)
# and the model fits the test spectra (chi^2 per degree of freedom ~ 1).
assert np.all(np.abs(pulls.std(0) - 1) < 0.1) and np.all(np.abs(pulls.mean(0)) < 0.2)
assert abs(np.median(chi2nu) - 1) < 0.05

# %% [markdown]
# **Figure 2.4.** The test step on 300 synthetic stars at S/N = 30 (three times noisier than the training set).
# Top: fitted vs true labels with formal error bars; the line is the identity. Each panel gives the mean
# difference (bias), its rms, and the median formal error. Bottom: the pulls (fitted − true)/(formal error) with
# the standard normal (dashed). The labels are unbiased and the formal errors describe the scatter: a model
# trained on high-S/N spectra measures labels of noisier spectra, because the noise model is explicit — the
# property that distinguishes The Cannon from a generic regression of labels on fluxes (N15 §1).
#
# The starting point matters little here. Restart from random points to see whether the optimiser ever ends in a
# different (local) minimum:

# %%
starts = rng.uniform(-1.5, 1.5, (5, K))
same = np.mean([[np.allclose(test(flux_te[m], err_te[m], theta, s, x0)[0], out[m][0], atol=1e-3) for x0 in starts]
                for m in range(50)])
print(f'fraction of (star, start) pairs that reach the same labels as the start at the pivot: {same:.3f}')
assert same > 0.95

# %% [markdown]
# ## 2.5 Noisy training labels
#
# The training step treats the training labels as exact. Real reference labels have errors: the uncertainties in
# the APOGEE DR17 catalogue are small internal ones (≈ 10 K, 0.03 dex, 0.01 dex for bright giants), while
# comparisons with independent temperature scales give a scatter of ≈ 100 K for $T_{\rm eff}$ (Jönsson et al. 2020,
# [AJ 160, 120](https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J), for DR16). What happens then? Regressing a quantity on a noisy predictor biases the slope
# towards zero (**regression dilution**): the model's derivatives $\theta_{\lambda k}$ are too small, and the test
# step must move the label further to explain the same change of flux. The labels that come out are stretched
# around the pivot. Below we perturb the training labels by Gaussian noise of 0, 5, 10 and 20% of each label's
# scale $q_k$ and measure the slope of fitted vs true labels on the test set.

# %%
fracs = [0, 0.05, 0.1, 0.2]
slopes = []
for fr in fracs:
    noisy = lab_tr + rng.normal(size=lab_tr.shape) * fr * scale       # training labels with errors
    th_n, _, s_n = train(to_x(noisy), flux_tr, err_tr)
    fit_n = to_lab(np.array([test(flux_te[m], err_te[m], th_n, s_n)[0] for m in range(len(lab_te))]))
    slopes.append([np.polyfit(lab_te[:, k], fit_n[:, k], 1)[0] for k in range(K)])
slopes = np.array(slopes)
# Prediction (classical regression dilution): the trained derivatives are attenuated by
# lambda = var(labels) / [var(labels) + var(noise)], so the test step stretches labels by 1/lambda = 1 + (noise/spread)^2.
spread = lab_tr.std(0)                                       # spread of the training labels (uniform: range / sqrt(12))
pred_slope = 1 + (np.array(fracs)[:, None] * scale / spread)**2
fig, ax = plt.subplots(figsize=(6.5, 4))
for k in range(K):
    ax.plot(np.array(fracs) * 100, slopes[:, k], 'o', color=f'C{k}', label=f'{names[k]} (measured)')
    ax.plot(np.array(fracs) * 100, pred_slope[:, k], '-', color=f'C{k}', lw=0.8)
ax.axhline(1, color='k', lw=0.8)
ax.set(xlabel='noise on the training labels [% of the label scale $q_k$]', ylabel='slope of fitted vs true label')
ax.legend()
plt.tight_layout()
print('measured slopes (rows: noise 0, 5, 10, 20%):\n', slopes.round(3))
print('predicted 1 + (noise/spread)^2:\n', pred_slope.round(3))
# Test: with exact labels the slope is 1 (within 2%); with noisy labels it follows the dilution prediction
# within 0.03 (the sampling error of a slope from 300 stars at S/N 30 is ~ 0.005-0.01).
assert np.all(np.abs(slopes[0] - 1) < 0.02) and np.all(np.abs(slopes - pred_slope) < 0.03)

# %% [markdown]
# **Figure 2.5.** Slope of a straight-line fit of fitted against true test labels (dots), for models trained on
# labels with Gaussian errors of 0–20% of each label's scale $q_k$ (≈ 0–150 K in $T_{\rm eff}$), and the
# regression-dilution prediction $1 + (\sigma_{\rm noise}/\sigma_{\rm labels})^2$ (lines), where
# $\sigma_{\rm labels}$ is the standard deviation of the training labels. With exact training labels the slope is
# 1. With noisy ones it grows above 1 as predicted: the trained derivatives are diluted, so the test step
# overshoots — a star hotter than the pivot is returned hotter still. The effect is second order in the noise
# (≈ 1% at 5%, ≈ 10% at 20%). The size of the effect depends on the noise
# relative to the spread of the training labels, which is one reason why N15 and Casey et al. 2016 use only the
# highest-S/N stars with the best labels for training.
#
# ## Summary
#
# - The Cannon is a per-pixel quadratic model of the flux in the labels, with an explicit noise model
#   $\sigma_{n\lambda}^2 + s_\lambda^2$.
# - Training is linear least squares per pixel (plus a one-dimensional scatter fit, which needs the full Gaussian
#   likelihood); testing is a small non-linear least-squares problem per star.
# - On synthetic data with a known answer, the coefficients, the scatter, the labels and their formal errors are
#   all recovered (pulls ≈ standard normal), at a test S/N three times lower than the training S/N.
# - Errors in the training labels bias the model; the bias is a stretch of the label scale.
#
# **Exercises.**
# 1. Use a *linear* label vector, $[1, x_1, \dots, x_K]$, to fit the same quadratic data. How large are the label
#    biases, and where in label space?
# 2. Train with only 50 stars. What happens to the pulls of the test labels, and why do the formal errors no
#    longer describe the scatter? (Hint: the formal error ignores the uncertainty of $\boldsymbol\theta$.)
# 3. Draw test stars outside the training box (e.g. [Fe/H] = −2.5). Are their labels right? Is their χ² larger?

# %% [markdown]
# ## References
#
# - Casey et al. 2016, *The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses*, arXiv:1603.03040. [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C)
# - Jönsson et al. 2020, *APOGEE Data and Spectral Analysis from SDSS Data Release 16: Seven Years of Observations Including First Results from APOGEE-South*, AJ 160, 120. [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J)
# - Ness et al. 2015, *The Cannon: A data-driven approach to Stellar Label Determination*, ApJ 808, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)
# - Rix et al. 2016, *Constructing Polynomial Spectral Models for Stars*, ApJL 826, L25. [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...826L..25R)
