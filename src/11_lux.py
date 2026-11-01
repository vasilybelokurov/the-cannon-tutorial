# %% [markdown]
# # Chapter 11 — Beyond The Cannon: a latent-variable model (Lux)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/11_lux.ipynb)
#
# **Goal.** Write *Lux* (Horta et al. 2025, H25) — a data-driven model in which each star has a hidden ("latent")
# vector from which *both* its labels and its spectrum are generated — and compare it with The Cannon on the twelve
# labels of Chapter 10. Find out what treating the labels as noisy outputs, rather than exact inputs, does and does
# not buy.
#
# **You will learn**
# 1. how a latent-variable model differs from The Cannon, and why its training is a sequence of weighted
#    least-squares fits;
# 2. that a seemingly harmless choice — how to solve a least-squares problem that has more unknowns than data
#    constrain — decides how many latent dimensions the model really uses;
# 3. to check a new method against simple baselines (ridge regression) and against the "measured or predicted?" test
#    of Chapter 8 before crediting it with a gain;
# 4. where Lux is clearly better than The Cannon: when the training labels are noisy.
#
# **The model** (H25 §3). Star $n$ has a latent vector $\mathbf z_n$ of length $P$. Its $M$ labels and its spectrum are
#
# $$\boldsymbol\ell_n = A\,\mathbf z_n + \text{noise}, \qquad \mathbf f_n = B\,\mathbf z_n + \text{noise},$$
#
# with $A$ an $M\times P$ and $B$ a $\Lambda\times P$ matrix ($\Lambda$ pixels). The label noise has the catalogue
# errors; the flux noise has the flux errors plus a fitted scatter $s_\lambda$ per pixel, as in The Cannon. In The
# Cannon the labels are exact inputs and the spectrum is a quadratic function of them; in Lux both are outputs, linear
# in $\mathbf z$, and the labels' errors enter the fit. To label a new star, find its $\mathbf z$ from the spectrum
# alone (a weighted least-squares fit, with no local minima), then $\boldsymbol\ell = A\mathbf z$.
#
# **Training** (H25 §4.1–4.2). Start from $\mathbf z_n = [1, \text{scaled labels}, 0, \dots, 0]$ and random $A$, $B$.
# Stage 1, five rounds without scatter: fit $A$ to the labels (with $\mathbf z$ fixed), $B$ to the spectra, then each
# $\mathbf z_n$ to its star's labels and spectrum. Stage 2, once: fit the scatter, then $B$ and $\mathbf z$ again with
# it (and a weak Gaussian prior of width $\tau = 1000$ on $\mathbf z$). The fits of $A$, $B$ and $\mathbf z$ are
# linear in the parameters they fit and are solved exactly by weighted least squares; the scatter is fitted by maximum
# likelihood on a grid, as in Chapter 2 — the block `lux.py` below, about sixty lines. For the same least-squares steps
# our code and the independent *pollux* package agree to $10^{-9}$ of the errors (Appendix B).
#
# One detail of H25's published code (its last step used the previous $B$ and scatter) differed from the paper; the
# author fixed it on 2026-11-01 after we reported it. We follow the paper.

# %%
#@@LOADER
#@@NORM
#@@CANNON
#@@MORE
#@@LUX
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
    good = np.isfinite(lab[NAMES]).all(axis=1).to_numpy()
    data[nm] = (lab[NAMES].to_numpy()[good], lab[[n + '_err' for n in NAMES]].to_numpy()[good], f[good], e[good])
L, Lerr, flux, err = data['train']
Lt, _, ft, et = (a[::2] for a in data['test'])               # every second test star (S/N 50-100), as in Chapter 10
val = np.random.default_rng(10).permutation(len(L)) % 5 == 0  # the validation stars of Chapter 10
tr = np.flatnonzero(~val)
print(f'training {len(tr)}, validation {val.sum()}, test {len(Lt)} stars; median label errors:',
      dict(zip(NAMES, np.median(Lerr, axis=0).round(3))))
rob = lambda d: np.array([scatter_stats(d[:, k])['robust'] for k in range(d.shape[1])])   # robust std per label
RES = {}                                                       # robust std on validation and test, per model


def score(name, lab_val, lab_test):
    """Robust std of each label minus ASPCAP, on validation and test stars; printed and stored."""
    RES[name] = {'validation': rob(lab_val - L[val]), 'test': rob(lab_test - Lt)}
    v, t = RES[name]['validation'], RES[name]['test']
    print(f'{name:38s} validation: T_eff {v[0]:5.1f} K, log g {v[1]:.3f}, [Fe/H] {v[2]:.3f}, median [X/Fe] '
          f'{np.median(v[3:]):.4f} | test: {t[0]:5.1f} K, {t[1]:.3f}, {t[2]:.3f}, {np.median(t[3:]):.4f}')


def cannon(idx, labs, error=err, error_t=et):
    """Twelve-label quadratic Cannon (Chapter 10) trained on stars idx with labels labs; labels of validation and test
    stars. error, error_t: flux errors of the training set and of the test stars (infinite where pixels are hidden)."""
    p, c = np.median(labs, axis=0), (np.percentile(labs, 97.5, axis=0) - np.percentile(labs, 2.5, axis=0)) / 2
    th, s = train_more((labs - p) / c, flux[idx], error[idx])
    run = lambda F, E: np.array([test_more(F[n], E[n], th, s)[0] for n in range(len(F))]) * c + p
    return run(flux[val], error[val]), run(ft, error_t)


def lux(idx, labs, labs_err, P=49, error=err, error_t=et, incremental=True):
    """Lux trained on stars idx; labels of validation and test stars, and the latents of the training stars."""
    A, B, s, Z, _ = lux_train(labs, labs_err, flux[idx], error[idx], P=P, incremental=incremental)
    out = [lux_test(F, E, A, B, s) for F, E in [(flux[val], error[val]), (ft, error_t)]]
    assert all(np.all(o[3] == 0) for o in out), 'some labels depend on latent directions the spectra do not constrain'
    return out[0][0], out[1][0], Z

# %% [markdown]
# ## 11.1 Lux against The Cannon — and against ridge regression
#
# Train Lux ($P = 49$, the value of H25's notebook; their paper says $4M = 48$) and the twelve-label Cannon of
# Chapter 10 on the same 1308 stars, and label the validation and test stars. Lux's labels are a *linear* function of
# the flux, $\boldsymbol\ell = A(B^\top W B)^{-1}B^\top W\mathbf f$. The simplest model with that property is ridge
# regression of the labels on the fluxes (Hoerl & Kennard 1970): it is the baseline any linear method must beat. Its
# one parameter is chosen on the validation stars, so ridge's validation scores are slightly optimistic; its test
# scores are not. (As in Chapter 10, the validation stars helped to choose the continuum pixels in Chapter 3, which
# Chapter 5 found to change the precision by less than 2%.)

# %%
t0 = time.time()
score('Cannon (12 labels, quadratic)', *cannon(tr, L[tr]))
lv, lt, Z49 = lux(tr, L[tr], Lerr[tr])
score('Lux (P = 49)', lv, lt)
fill = lambda f, e: np.where(np.isfinite(e), f, 1.0)            # ridge needs a complete flux matrix


def ridge(idx, labs, lam, E=err, E_t=et):
    """Ridge regression of labels on fluxes (dual form: N x N system); labels of validation and test stars."""
    X = fill(flux[idx], E[idx])
    mu, ym = X.mean(axis=0), labs.mean(axis=0)
    W = (X - mu).T @ np.linalg.solve((X - mu) @ (X - mu).T + lam * np.eye(len(X)), labs - ym)
    return (fill(flux[val], E[val]) - mu) @ W + ym, (fill(ft, E_t) - mu) @ W + ym


spread = np.percentile(L[tr], 97.5, axis=0) - np.percentile(L[tr], 2.5, axis=0)
lam_r = min([1e-2, 1e-1, 1, 10], key=lambda lam: np.median(rob(ridge(tr, L[tr], lam)[0] - L[val]) / spread))
score(f'ridge regression (lambda = {lam_r:g})', *ridge(tr, L[tr], lam_r))
print(f'{time.time() - t0:.0f} s')

# %% [markdown]
# **How many latent dimensions does Lux use?** The starting latents $[1, \text{scaled labels}, 0, \dots, 0]$ have only
# $M + 1 = 13$ non-zero columns. Fitting $A$ and $B$ to them is a least-squares problem with 49 unknowns per label or
# pixel of which only 13 are constrained. The usual "minimum-norm" solution sets the other 36 to zero, and the next
# latent fit then keeps the extra 36 latent coordinates at zero too: the model stays at 13 dimensions whatever $P$ is.
# H25's optimiser instead updates from the current (random) values, keeping their unconstrained part, which activates
# the extra dimensions; our `lsq_update(..., incremental=True)` does the same. Compare.

# %%
t0 = time.time()
ranks = {}
for P, inc in [(13, True), (25, True), (49, False)]:
    name = f'Lux (P = {P}' + (', minimum-norm steps)' if not inc else ')')
    lv_, lt_, Z = lux(tr, L[tr], Lerr[tr], P=P, incremental=inc)
    score(name, lv_, lt_)
    ranks[name] = np.linalg.matrix_rank(Z)
ranks['Lux (P = 49)'] = np.linalg.matrix_rank(Z49)
print('rank of the training latents:', ranks)
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
models = ['Cannon (12 labels, quadratic)', f'ridge regression (lambda = {lam_r:g})', 'Lux (P = 13)', 'Lux (P = 25)',
          'Lux (P = 49)', 'Lux (P = 49, minimum-norm steps)']
short = ['Cannon', 'ridge', 'Lux P=13', 'Lux P=25', 'Lux P=49', 'Lux P=49\nmin-norm']
for a, (k, lab) in zip(ax, [(0, r'$T_{\rm eff}$ [K]'), (1, r'$\log g$'), (None, 'median of the nine [X/Fe]')]):
    for j, split in enumerate(['validation', 'test']):
        v = [RES[m][split][k] if k is not None else np.median(RES[m][split][3:]) for m in models]
        a.bar(np.arange(len(models)) + 0.38 * j - 0.19, v, 0.36, color=['C0', 'C1'][j],
              label=split + (' (high S/N)' if j == 0 else ' (S/N 50–100)'))
    a.set_xticks(range(len(models)), short, fontsize=8)
    a.set(ylabel=f'robust std vs ASPCAP, {lab}')
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 11.1.** Precision of $T_{\rm eff}$, $\log g$ and the abundances (median over the nine [X/Fe]) for the
# twelve-label Cannon, ridge regression of the labels on the fluxes, and Lux with $P = 13, 25, 49$ latent dimensions,
# all trained on the same 1308 stars; blue: validation stars (high S/N), orange: test stars (S/N 50–100). Lux and
# ridge — both linear in the flux — measure the abundances better than The Cannon and the stellar parameters worse,
# much worse at low S/N, where a linear model's many coefficients also fit noise. Most of Lux's abundance gain is
# already in ridge regression. Lux with $P = 49$ and minimum-norm steps gives *exactly* the same numbers as $P = 13$:
# its latents have rank 13 (printed above).
#
# **Checks.**

# %%
cn, lx, rg = RES['Cannon (12 labels, quadratic)'], RES['Lux (P = 49)'], RES[f'ridge regression (lambda = {lam_r:g})']
assert ranks['Lux (P = 49, minimum-norm steps)'] == 13 and ranks['Lux (P = 49)'] == 49
assert np.allclose(RES['Lux (P = 49, minimum-norm steps)']['validation'], RES['Lux (P = 13)']['validation'])
assert np.median(lx['validation'][3:]) < np.median(cn['validation'][3:]), 'Lux abundances not better than Cannon'
assert lx['test'][0] > 1.3 * cn['test'][0], 'Lux T_eff at low S/N no longer clearly worse than Cannon'
assert np.median(rg['validation'][3:]) < np.median(cn['validation'][3:])

# %% [markdown]
# ## 11.2 Measured or predicted?
#
# Chapter 8 showed that [Mg/Fe] can be *predicted* from the other labels, because the abundances of disc stars are
# correlated. Is Lux's better [Mg/Fe] read from the Mg lines? Hide every pixel within 1.5 Å of the 40 strongest Mg I
# lines of our line list (weaker Mg lines and blends remain), in training and test spectra, and retrain all three
# models.

# %%
t0 = time.time()
w_mg = lines[lines.species == 'MG I'].wave_vac.to_numpy()
mg = np.min(np.abs(wave[:, None] - w_mg[None, :]), axis=1) < 1.5
err_nomg, et_nomg = np.where(mg[None, :], np.inf, err), np.where(mg[None, :], np.inf, et)
print(f'hiding {mg.sum()} pixels near Mg I lines')
score('Cannon, Mg lines hidden', *cannon(tr, L[tr], err_nomg, et_nomg))
score('Lux (P = 49), Mg lines hidden', *lux(tr, L[tr], Lerr[tr], error=err_nomg, error_t=et_nomg)[:2])
score('ridge, Mg lines hidden', *ridge(tr, L[tr], lam_r, err_nomg, et_nomg))
k = NAMES.index('mg_fe')
pairs = [('Cannon (12 labels, quadratic)', 'Cannon, Mg lines hidden'),
         (f'ridge regression (lambda = {lam_r:g})', 'ridge, Mg lines hidden'),
         ('Lux (P = 49)', 'Lux (P = 49), Mg lines hidden')]
cost = {(a_, sp): RES[b_][sp][k] / RES[a_][sp][k] - 1 for a_, b_ in pairs for sp in ['validation', 'test']}
print('increase of the [Mg/Fe] scatter when the Mg I windows are hidden:',
      {f'{a_.split()[0]}, {sp}': f'{v:+.0%}' for (a_, sp), v in cost.items()})
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for i, (a_, b_) in enumerate(pairs):
    for j, split in enumerate(['validation', 'test']):
        x = i + 0.38 * j - 0.19
        ax.bar(x, RES[a_][split][k], 0.36, color=['C0', 'C1'][j], label=f'{split}, all pixels' if i == 0 else None)
        ax.plot([x - 0.17, x + 0.17], [RES[b_][split][k]] * 2, 'k-', lw=2,
                label='Mg I lines hidden' if i == 0 and j == 0 else None)
ax.set_xticks(range(3), ['Cannon', 'ridge', 'Lux'])
ax.set(ylabel='robust std of [Mg/Fe] vs ASPCAP [dex]')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 11.2.** Precision of [Mg/Fe] for the three models with all pixels (bars) and with the 393 pixels around the
# 40 strongest Mg I lines hidden in training and test (black lines). All three read those lines — hiding them costs
# 20–43% (printed above) — but Lux with them hidden is still better than The Cannon with all pixels, on validation and
# test stars. Much of the precision of [Mg/Fe], for every model, therefore comes from other pixels: weaker Mg lines and
# blends that remain, and, as Chapter 8 found, the correlations of the training labels. "Better abundances" of a
# data-driven model must be read with this in mind (Ness et al. 2019, 2022).
#
# **Checks.**

# %%
assert all(v > 0.1 for v in cost.values()), 'hiding the Mg I windows does not cost every model > 10%'
for sp in ['validation', 'test']:
    assert RES['Lux (P = 49), Mg lines hidden'][sp][k] < RES['Cannon (12 labels, quadratic)'][sp][k]

# %% [markdown]
# ## 11.3 Missing labels
#
# Because Lux treats labels as data with errors, a star with a missing label can still be used: its error is set to
# infinity and the label drops out of the fit (the pivots, scales and starting latents use only the labels present).
# The Cannon needs every label of every training star. Hide [N/Fe] for a random half of the training stars (each star
# with probability ½).

# %%
t0 = time.time()
miss = np.random.default_rng(20).random(len(tr)) < 0.5
L_miss = L[tr].copy()
L_miss[miss, NAMES.index('n_fe')] = np.nan
print(f'[N/Fe] hidden for {miss.sum()} of {len(tr)} training stars; {(~miss).sum()} keep it')
score('Lux, all stars, half without [N/Fe]', *lux(tr, L_miss, Lerr[tr])[:2])
score('Lux, only the stars with [N/Fe]', *lux(tr[~miss], L[tr][~miss], Lerr[tr][~miss])[:2])
score('Cannon, only the stars with [N/Fe]', *cannon(tr[~miss], L[tr][~miss]))
kn = NAMES.index('n_fe')
for m in ['Lux, all stars, half without [N/Fe]', 'Lux, only the stars with [N/Fe]', 'Cannon, only the stars with [N/Fe]']:
    print(f"{m:38s} [N/Fe]: validation {RES[m]['validation'][kn]:.3f}, test {RES[m]['test'][kn]:.3f}")
print(f'{time.time() - t0:.0f} s')

# %% [markdown]
# Training Lux on all 1308 stars, 682 of them without [N/Fe], gives the same precision as training it on the 626
# stars with [N/Fe] only: in this random draw the extra spectra do not help. The Cannon trained on those 626 stars
# measures [N/Fe] about as well. Being able to use incomplete labels is a convenience; here it brought no measurable
# gain (one draw; other data, or a label missing for most stars, may differ).
#
# **Check.**

# %%
a_, b_ = RES['Lux, all stars, half without [N/Fe]'], RES['Lux, only the stars with [N/Fe]']
assert abs(a_['validation'][kn] / b_['validation'][kn] - 1) < 0.1

# %% [markdown]
# ## 11.4 Noisy training labels
#
# Real training labels have errors, and a cheaper survey's are larger than ASPCAP's (median 8 K in $T_{\rm eff}$,
# 0.01–0.02 dex in abundances). Add Gaussian noise of five times the ASPCAP error to the training labels — about 40 K,
# 0.12 dex in $\log g$ and 0.07 dex in abundances — and tell Lux the true total error, $\sqrt{26}\,\sigma$. Score all
# models against the original ASPCAP labels of the validation and test stars.

# %%
t0 = time.time()
noisy = L[tr] + np.random.default_rng(21).normal(0, 1, L[tr].shape) * 5 * Lerr[tr]
score('Cannon, noisy training labels', *cannon(tr, noisy))
score('ridge, noisy training labels', *ridge(tr, noisy, lam_r))
score('Lux, noisy training labels', *lux(tr, noisy, np.sqrt(26) * Lerr[tr])[:2])
print(f'{time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
trio = [('Cannon (12 labels, quadratic)', 'Cannon, noisy training labels', 'Cannon'),
        (f'ridge regression (lambda = {lam_r:g})', 'ridge, noisy training labels', 'ridge'),
        ('Lux (P = 49)', 'Lux, noisy training labels', 'Lux')]
for a, (k_, lab) in zip(ax, [(None, 'median of the nine [X/Fe] [dex]'), (0, r'$T_{\rm eff}$ [K]')]):
    for j, split in enumerate(['validation', 'test']):
        for i, (c_, n_, nm) in enumerate(trio):
            get = lambda m: np.median(RES[m][split][3:]) if k_ is None else RES[m][split][k_]
            x = i + 0.38 * j - 0.19
            a.bar(x, get(c_), 0.36, color=['C0', 'C1'][j], alpha=0.5, label=f'{split}, ASPCAP labels' if i == 0 else None)
            a.plot([x - 0.17, x + 0.17], [get(n_)] * 2, 'k-', lw=2,
                   label='noisy training labels (5 sigma)' if i == 0 and j == 0 else None)
    a.set_xticks(range(3), [t[2] for t in trio])
    a.set(ylabel=f'robust std vs ASPCAP, {lab}')
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 11.3.** Precision of the abundances (left) and of $T_{\rm eff}$ (right) when the models are trained on the
# ASPCAP labels (bars) and on the same labels with noise of 5× the ASPCAP errors added (black lines). The Cannon's
# abundances become twice as scattered. Its labels are *inputs* to the polynomial, and the degradation is what noise
# in inputs does to fitted coefficients (the errors-in-variables problem; the regression dilution of Chapter 2),
# although this experiment does not isolate that cause. Ridge regression, which treats the labels as outputs, suffers
# less; Lux, which treats them as outputs with known errors and passes them through a low-dimensional latent space,
# hardly at all — which of these features matters most, this comparison cannot say. This is the clearest advantage of
# Lux on our data. Caveats: one noise draw; Lux was told the true total errors; ridge's strength was chosen on the
# clean labels and not re-tuned.
#
# **Checks.**

# %%
mx = lambda m, split: np.median(RES[m][split][3:])
for split in ['validation', 'test']:
    assert mx('Cannon, noisy training labels', split) > 1.5 * mx('Cannon (12 labels, quadratic)', split)
    assert mx('Lux, noisy training labels', split) < 1.15 * mx('Lux (P = 49)', split)
    assert mx('ridge, noisy training labels', split) > mx('Lux, noisy training labels', split)

# %% [markdown]
# ## Summary
#
# - Lux generates labels and spectra from one latent vector per star; trained by a sequence of weighted least-squares
#   fits (and a grid fit of the scatter), it takes half a minute on 1308 stars. For the same least-squares steps our
#   sixty lines agree with the *pollux* package to $10^{-9}$ (Appendix B).
# - How an under-determined least-squares step is solved matters: minimum-norm steps leave the model at $M + 1 = 13$
#   latent dimensions whatever $P$ is.
# - On our data Lux measures abundances better than The Cannon and stellar parameters worse (much worse at low S/N);
#   ridge regression gets most of the way, and much of the [Mg/Fe] precision survives hiding the strongest Mg lines.
# - Training on stars with a missing label works, but did not help in our test.
# - Lux's clear strength is robustness to noisy training labels: at 5× the ASPCAP errors its abundances barely change,
#   while The Cannon's scatter doubles.
# - For research use: the *pollux* package (Price-Whelan), which implements Lux and The Cannon in JAX with priors,
#   probabilistic fits and any number of outputs — Appendix B shows how to use it.
#
# ## References
#
# - Hoerl, A. E., & Kennard, R. W. 1970, *Ridge regression: biased estimation for nonorthogonal problems*,
#   Technometrics 12, 55, [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
# - Horta, D., Price-Whelan, A. M., Hogg, D. W., Ness, M. K., & Casey, A. R. 2025, *Lux: a generative, multi-output,
#   latent-variable model for astronomical data with noisy labels*,
#   [AJ 169, 314](https://ui.adsabs.harvard.edu/abs/2025AJ....169..314H) — code:
#   [github.com/DHortaD/Lux](https://github.com/DHortaD/Lux) (MIT licence), [ascl:2510.013](https://ui.adsabs.harvard.edu/abs/2025ascl.soft10013H).
# - Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, *The Cannon: a data-driven approach to
#   stellar label determination*, [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N).
# - Ness et al. 2019, *In the Galactic disk, stellar [Fe/H] and age predict orbits and precise [X/Fe]*,
#   [ApJ 883, 177](https://ui.adsabs.harvard.edu/abs/2019ApJ...883..177N).
# - Ness et al. 2022, *The homogeneity of the star-forming environment of the Milky Way disk over time*,
#   [ApJ 926, 144](https://ui.adsabs.harvard.edu/abs/2022ApJ...926..144N).
# - Price-Whelan, A. M., *pollux: a JAX framework for data-driven latent variable models*,
#   [github.com/adrn/pollux](https://github.com/adrn/pollux), [documentation](https://pollux-astro.readthedocs.io).
