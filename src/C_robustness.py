# %% [markdown]
# # Appendix C — How stable are the comparisons?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/C_robustness.ipynb)
#
# Chapters 10 and 11 use a fixed split to teach the methods. Here we investigate the unexpected training-size
# effect and repeat the missing/noisy-label experiments. The targets remain ASPCAP labels, not independent truth.
# This appendix is more expensive than the worked examples: it fits several dozen models.
#
# **Protocol.** Select continuum pixels using only training stars, before examining validation or test scores.
# For the size experiment use three splits and nested training subsets, with a common label scaling within each
# split so that coefficients and scatter can be exchanged. For missing and noisy labels use three random draws
# on the first split. Retune ridge on validation stars in each experiment. The low-S/N sample is reserved for
# reporting, never tuning. Paired bootstrap intervals resample held-out stars together for both methods; they
# describe sampling uncertainty conditional on the fitted models. Repeated splits/draws separately show training
# variation; three repetitions cannot establish population-wide robustness. The intervals are pointwise, not
# corrected for multiple comparisons, and the shared test stars make estimates across splits correlated.
#
# We print bias, robust scatter and RMSE per label and metallicity bin. Bias and RMSE are necessary because a
# narrow distribution of differences can still be biased. The initial experiment had no pass/fail requirement
# on which method won. The measured regression bounds below were frozen after inspecting that run; they protect
# the worked example against changes and are not hypothesis tests.

# %%
#@@LOADER
#@@NORM
#@@CANNON
#@@MORE
#@@LUX
#@@ROBUSTNESS
import time
from IPython.display import HTML, display

t0 = time.time()
names = ['teff', 'logg', 'fe_h'] + [e + '_fe' for e in ['c', 'n', 'o', 'mg', 'al', 'si', 'ca', 'mn', 'ni']]
catalogue = load('labels')
data = {}
for sample in ['train', 'test']:
    table = catalogue[catalogue['sample'] == sample].reset_index(drop=True)
    good = np.isfinite(table[names]).all(axis=1).to_numpy()
    f, e = prepare(load('spectra_' + sample))
    data[sample] = (table.loc[good].reset_index(drop=True), f[good], e[good])
tab, raw, rawerr = data['train']
testtab, test_raw, test_rawerr = (a[::2] for a in data['test'])
L, LE = tab[names].to_numpy(), tab[[n + '_err' for n in names]].to_numpy()
truth = testtab[names].to_numpy()
quality = (tab[[n + '_flag' for n in names[3:]]].to_numpy() == 0).all(axis=1)
assert np.all(np.isfinite(LE) & (LE > 0)), 'training label errors must be usable'
print('Stars:', len(L), 'high S/N;', len(truth), 'low S/N; all nine abundance flags clear:', quality.sum())


def fit_cannon(idx, labs, F, E, pivot=None, scale=None):
    if pivot is None:
        pivot = np.median(labs, axis=0)
        scale = np.diff(np.percentile(labs, [2.5, 97.5], axis=0), axis=0)[0] / 2
    th, s = train_more((labs - pivot) / scale, F[idx], E[idx])
    return th, s, pivot, scale


def predict_cannon(model, F, E, starts=None):
    th, s, pivot, scale = model
    out = [test_more(f, e, th, s, starts=starts) for f, e in zip(F, E)]
    assert all(o[4] for o in out), 'test-step convergence'
    return np.array([o[0] for o in out]) * scale + pivot, np.array([o[2] for o in out])


def statistics(pred, ref):
    d = pred - ref
    return pd.DataFrame({'label': names, 'bias': d.mean(axis=0), 'robust': robust_std(d),
                         'rmse': np.sqrt(np.mean(d**2, axis=0))})


def show_table(table, title):
    """Keep the per-label evidence available without making the rendered chapter hundreds of lines long."""
    display(HTML('<details><summary>' + title + '</summary>' +
                 table.to_html(index=False, float_format=lambda x: f'{x:.5g}') + '</details>'))


def compare(a, b, ref):
    sa, sb = robust_std(a - ref), robust_std(b - ref)
    delta = np.r_[sa - sb, np.median(sa[3:]) - np.median(sb[3:])]
    interval = paired_interval(a - ref, b - ref)
    return delta, interval

# %% [markdown]
# ## C.1 Why can fewer training stars give tighter validation abundances?
#
# Use splits 10 (the original split), 13 and 14. Within each, the 300-star set is a subset of the full set.
# Both models use the same continuum selection and label coordinates. On split 10 also exchange the fitted
# coefficients and scatter, require all nine abundance flags to be clear, remove stars with mean training
# chi-squared above 3, and use three starting points when fitting validation labels. These are diagnostics,
# not alternative models selected by their test scores. Quality cuts and residual cuts change the population
# as well as its size, so any improvement alone does not identify a unique cause.

# %%
size_results, size_tables = [], []
base = {}
for seed in [10, 13, 14]:
    val = np.random.default_rng(seed).permutation(len(L)) % 5 == 0
    idx = np.flatnonzero(~val)
    small = idx[np.random.default_rng(11).permutation(len(idx))[:300]]
    cont = training_continuum(L[idx, :3], raw[idx], rawerr[idx])
    assert all(cont[(wave > lo) & (wave < hi)].sum() >= 3 for lo, hi in CHIPS)
    F, E = cheb_normalise(raw, rawerr, cont)
    Ft, Et = cheb_normalise(test_raw, test_rawerr, cont)
    pivot = np.median(L[idx], axis=0)
    scale = np.diff(np.percentile(L[idx], [2.5, 97.5], axis=0), axis=0)[0] / 2
    models = {label: fit_cannon(rows, L[rows], F, E, pivot, scale)
              for label, rows in [('full', idx), ('300', small)]}
    if seed == 10:
        full, few = models['full'], models['300']
        models['full theta, 300 scatter'] = (full[0], few[1], pivot, scale)
        models['300 theta, full scatter'] = (few[0], full[1], pivot, scale)
        clean_idx = idx[quality[idx]]
        models['all flags clear'] = fit_cannon(clean_idx, L[clean_idx], F, E, pivot, scale)
        residual = F[idx] - vectorize_n((L[idx] - pivot) / scale) @ full[0].T
        z2 = np.where(np.isfinite(E[idx]), residual**2 / (E[idx]**2 + full[1]**2), np.nan)
        keep = np.nanmean(z2, axis=1) < 3
        models['residual cut'] = fit_cannon(idx[keep], L[idx[keep]], F, E, pivot, scale)
        print('Split 10: continuum pixels', cont.sum(), '; flag-cut stars', len(clean_idx),
              '; residual-cut stars', keep.sum())
    predictions = {}
    for label, model in models.items():
        predictions[label] = [predict_cannon(model, f, e)[0] for f, e in [(F[val], E[val]), (Ft, Et)]]
        print('split', seed, label, 'predictions complete', flush=True)
    if seed == 10:
        starts = [np.zeros(12), np.full(12, -0.5), np.full(12, 0.5)]
        for label in ['full', '300']:
            predictions[label + ', multi-start'] = []
            for f, e in [(F[val], E[val]), (Ft, Et)]:
                single, chi = predict_cannon(models[label], f, e)
                multi, chi_multi = predict_cannon(models[label], f, e, starts)
                assert np.all(chi_multi <= chi + 1e-6), 'multi-start includes the original start'
                predictions[label + ', multi-start'].append(multi)
        base = dict(idx=idx, val=val, F=F, E=E, Ft=Ft, Et=Et, model=models['full'],
                    pred=predictions['full'], cont=cont)
    for label, pred in predictions.items():
        for j, (split, ref) in enumerate([('validation', L[val]), ('test', truth)]):
            delta, ci = compare(pred[j], predictions['300'][j], ref)
            size_results.append(dict(seed=seed, model=label, split=split, delta=delta[-1], lo=ci[0, -1], hi=ci[1, -1]))
            table = statistics(pred[j], ref).assign(seed=seed, model=label, split=split)
            size_tables.append(table)
    print('finished split', seed, 'elapsed seconds', round(time.time() - t0), flush=True)
size_results = pd.DataFrame(size_results)
size_tables = pd.concat(size_tables, ignore_index=True)
print(size_results.round(5).to_string(index=False))
show_table(size_tables, 'Per-label bias, robust scatter and RMSE for the size experiment (K, dex)')

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, split in zip(axes[0], ['validation', 'test']):
    d = size_results[(size_results.split == split) & (size_results.model != '300')]
    y = np.arange(len(d))
    ax.hlines(y, d.lo, d.hi, color='C0')
    ax.plot(d.delta, y, 'o', color='C0')
    ax.set_yticks(y, [f'{r.model} (split {r.seed})' for r in d.itertuples()])
    ax.axvline(0, color='k', lw=0.8)
    ax.set(xlabel='median abundance scatter minus 300-star model [dex]', title=split)
for ax, split in zip(axes[1], ['validation', 'test']):
    d = size_tables[(size_tables.seed == 10) & (size_tables.split == split)]
    full = d[d.model == 'full'].set_index('label').loc[names[3:]]
    small = d[d.model == '300'].set_index('label').loc[names[3:]]
    for metric in ['robust', 'rmse']:
        ax.plot(range(9), small[metric] / full[metric], 'o-', label=metric)
    ax.axhline(1, color='k', lw=0.8)
    ax.set_xticks(range(9), [n[:-3].capitalize() for n in names[3:]])
    ax.set(xlabel='element', ylabel='300-star / full-model error', title=split + ', split 10')
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure C.1.** Top: difference in median robust scatter over nine [X/Fe] labels between each diagnostic model and
# the 300-star model, for high-S/N validation stars (left) and the separate S/N 50–100 test stars (right).
# Points are measured differences in dex; horizontal bars are paired 95% bootstrap intervals over the stars.
# Positive values favour 300 stars. Exchanging scatter at fixed coefficients isolates the effect of test-step
# pixel weights; exchanging coefficients includes how training weights affected the learned spectrum. The three
# splits test whether the original size comparison persists. Bottom: the 300-star/full-model ratio of robust
# scatter and RMSE, element by element on split 10 (both dimensionless). Below one favours the smaller model.
# The robust scatter describes the central distribution; RMSE includes bias and large errors.
#
# Figure C.1 shows that on split 10 the full model has 0.0114 dex more median abundance scatter than the 300-star
# model. Replacing its
# test-time scatter with the 300-star scatter leaves 0.0100 dex; keeping the 300-star coefficients and replacing
# their scatter changes the score by only −0.0004 dex. The effect therefore follows the learned coefficients,
# rather than mainly the final pixel weights. All-element quality cuts, a residual cut and extra starts leave
# the difference. Training weights can still affect the coefficients themselves; these swaps do not separate
# that mechanism from differences in sample composition or label correlations.
#
# The other splits give smaller differences (0.0047 and 0.0058 dex), with intervals crossing zero. The low-S/N
# differences are also small. Thus the original observation survives several controls but is not evidence for
# a generally optimal training size of 300. In fact, on split 10 the smaller model has worse RMSE for five of
# the nine abundances, despite tighter robust scatter for all nine: a narrower core does not imply smaller
# overall errors. The remaining question is why the learned coefficients change the shape of those residual
# distributions; these diagnostics do not identify a unique physical cause.
#
# **Regression checks.** Preserve the split-10 measured difference to 0.004 dex, and the near-invariance of
# the 300-star coefficients under the scatter exchange to 0.003 dex. These are checks on this fixed dataset.

# %%
check = size_results[(size_results.seed == 10) & (size_results.split == 'validation')].set_index('model')
assert abs(check.loc['full', 'delta'] - 0.0114) < 0.004
assert abs(check.loc['300 theta, full scatter', 'delta']) < 0.003
tab10 = size_tables[(size_tables.seed == 10) & (size_tables.split == 'validation')]
full = tab10[tab10.model == 'full'].set_index('label').loc[names[3:]]
small = tab10[tab10.model == '300'].set_index('label').loc[names[3:]]
assert (small.rmse > full.rmse).sum() >= 4, 'the narrower core does not uniformly improve RMSE'

# %% [markdown]
# ## C.2 Lux, ridge and The Cannon with training-only continuum pixels
#
# Use split 10 from C.1. Tune ridge on validation labels, then report all models on the same stars (Figure C.2).
# The tables
# include metallicity bins; sparse bins should be interpreted cautiously. Ridge's validation score has been used
# to choose its penalty and is optimistic. We do not use its test score for this choice.

# %%
idx, val, F, E, Ft, Et = [base[k] for k in ['idx', 'val', 'F', 'E', 'Ft', 'Et']]
refs = [L[val], truth]
eval_flux = [(F[val], E[val]), (Ft, Et)]


def fit_predict_lux(rows, labs, errors):
    A, B, s, Z, history = lux_train(labs, errors, F[rows], E[rows])
    out = [lux_test(f, e, A, B, s) for f, e in eval_flux]
    assert all(np.all(o[3] == 0) for o in out), 'labels depend on unconstrained spectral directions'
    assert all(np.all(np.isfinite(o[0])) for o in out)
    return [o[0] for o in out], (A, B, s)


def tuned_ridge(rows, labs):
    """Fit each available output on its labelled stars; choose one penalty on clean validation labels.
    Missing [N/Fe] excludes a star only from that output, not from the other eleven fits.
    """
    X = np.where(np.isfinite(E[rows]), F[rows], 1.0)
    Xeval = [np.where(np.isfinite(e), f, 1.0) for f, e in eval_flux]
    predictions = {}
    # Group outputs with the same missing-data pattern to reuse each matrix solve.
    patterns = {}
    for k in range(labs.shape[1]):
        good = np.isfinite(labs[:, k])
        patterns.setdefault(good.tobytes(), (good, []))[1].append(k)
    for lam in [1e-2, 1e-1, 1, 10]:
        pred = [np.zeros_like(ref) for ref in refs]
        for good, cols in patterns.values():
            x, y = X[good], labs[good][:, cols]
            mu, ym = x.mean(axis=0), y.mean(axis=0)
            x = x - mu
            coef = x.T @ np.linalg.solve(x @ x.T + lam * np.eye(len(x)), y - ym)
            for p, xe in zip(pred, Xeval):
                p[:, cols] = (xe - mu) @ coef + ym
        predictions[lam] = pred
    spread = np.diff(np.nanpercentile(labs, [2.5, 97.5], axis=0), axis=0)[0]
    lam = min(predictions, key=lambda l: np.median(robust_std(predictions[l][0] - refs[0]) / spread))
    return predictions[lam], lam


lux_pred, lux_model = fit_predict_lux(idx, L[idx], LE[idx])
ridge_pred, ridge_lam = tuned_ridge(idx, L[idx])
baseline = {'Cannon': base['pred'], 'Lux': lux_pred, 'ridge': ridge_pred}
print('Clean-label ridge penalty:', ridge_lam)
summary = []
for model, preds in baseline.items():
    for split, pred, ref in zip(['validation', 'test'], preds, refs):
        for lo, hi in [(-np.inf, np.inf), (-1.5, -0.5), (-0.5, 0), (0, 0.5)]:
            use = (ref[:, 2] >= lo) & (ref[:, 2] < hi)
            if use.sum() < 10:
                continue
            summary.append(statistics(pred[use], ref[use]).assign(model=model, split=split,
                            metallicity=f'[{lo}, {hi})', stars=use.sum()))
summary = pd.concat(summary, ignore_index=True)
show_table(summary, 'Baseline comparisons by label and metallicity bin (K, dex)')
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k, label in zip(axes, [0, 1, None], ['T_eff [K]', 'log g [dex]', 'median [X/Fe] [dex]']):
    for j, (split, ref) in enumerate(zip(['validation', 'test'], refs)):
        scores = [robust_std(pred[j] - ref) for pred in baseline.values()]
        ax.bar(np.arange(3) + (j - 0.5) * 0.35,
               [s[k] if k is not None else np.median(s[3:]) for s in scores], 0.35, label=split)
    ax.set_xticks(range(3), list(baseline))
    ax.set(ylabel='robust scatter, ' + label)
axes[0].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure C.2.** Robust scatter about ASPCAP for temperature (kelvin), gravity (dex), and the median over nine
# abundances (dex), using continuum pixels selected from training stars only. Blue bars show high-S/N validation
# stars and orange bars the S/N 50–100 test sample. Compare this ranking with Figure 11.1; the accompanying tables
# give bias and RMSE as well as scatter, separately for metallicity bins.

# %% [markdown]
# ## C.3 Missing and noisy labels, repeated
#
# Figure C.3 compares three draws in which nitrogen is hidden for half the training stars. Compare Lux using all
# available data with
# Lux using only complete stars, Cannon using complete stars, and ridge using every available label separately.
# Then add noise of five times the catalogue error to all training labels. Retune ridge for that draw, and fit
# Lux both with the correct total errors and with the original, underestimated errors. Each comparison uses
# identical perturbations for all methods. Held-out labels and spectra are never perturbed. Continuum selection
# is held fixed from C.1 to isolate label fitting; each model's label scaling uses only its available, perturbed
# training labels. This does not simulate re-estimating the continuum from a noisy reference catalogue.

# %%
experiments, experiment_tables = [], []
for seed in [20, 21, 22]:
    rng = np.random.default_rng(seed)
    missing = rng.random(len(idx)) < 0.5
    labs = L[idx].copy()
    labs[missing, names.index('n_fe')] = np.nan
    complete = idx[~missing]
    cm = fit_cannon(complete, L[complete], F, E)
    preds = {'Lux all': fit_predict_lux(idx, labs, LE[idx])[0],
             'Lux complete': fit_predict_lux(complete, L[complete], LE[complete])[0],
             'Cannon complete': [predict_cannon(cm, f, e)[0] for f, e in eval_flux]}
    preds['ridge available'], lam = tuned_ridge(idx, labs)
    print('Missing draw', seed, 'ridge penalty', lam, flush=True)
    noisy = L[idx] + rng.normal(size=L[idx].shape) * 5 * LE[idx]
    cm = fit_cannon(idx, noisy, F, E)
    noise_preds = {'Cannon': [predict_cannon(cm, f, e)[0] for f, e in eval_flux],
                   'Lux correct errors': fit_predict_lux(idx, noisy, np.sqrt(26) * LE[idx])[0],
                   'Lux underestimated errors': fit_predict_lux(idx, noisy, LE[idx])[0]}
    noise_preds['ridge retuned'], lam = tuned_ridge(idx, noisy)
    print('Noise draw', seed, 'ridge penalty', lam, 'elapsed seconds', round(time.time() - t0), flush=True)
    for experiment, collection in [('missing', preds), ('noisy', noise_preds)]:
        for model, predictions in collection.items():
            for j, (split, ref) in enumerate(zip(['validation', 'test'], refs)):
                reference = preds['Lux complete'][j] if experiment == 'missing' else baseline[
                    'Lux' if model.startswith('Lux') else ('ridge' if model.startswith('ridge') else 'Cannon')][j]
                delta, ci = compare(predictions[j], reference, ref)
                k = names.index('n_fe') if experiment == 'missing' else -1
                experiments.append(dict(experiment=experiment, seed=seed, model=model, split=split,
                                        delta=delta[k], lo=ci[0, k], hi=ci[1, k]))
                experiment_tables.append(statistics(predictions[j], ref).assign(
                    experiment=experiment, seed=seed, model=model, split=split))
experiments = pd.DataFrame(experiments)
experiment_tables = pd.concat(experiment_tables, ignore_index=True)
print(experiments.round(5).to_string(index=False))
show_table(experiment_tables, 'Per-label results for every perturbation draw (K, dex)')
assert np.isfinite(experiments[['delta', 'lo', 'hi']]).all().all()

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for row, experiment in enumerate(['missing', 'noisy']):
    for col, split in enumerate(['validation', 'test']):
        ax = axes[row, col]
        d = experiments[(experiments.experiment == experiment) & (experiments.split == split)]
        y = np.arange(len(d))
        ax.hlines(y, d.lo, d.hi, color='C0')
        ax.plot(d.delta, y, 'o', color='C0')
        ax.set_yticks(y, [f'{r.model} (draw {r.seed})' for r in d.itertuples()])
        ax.axvline(0, color='k', lw=0.8)
        ax.set(title=experiment + ', ' + split, xlabel='change in robust scatter [dex]')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure C.3.** Missing nitrogen (top) and added label noise (bottom), for validation stars (left) and low-S/N
# test stars (right), with three independent perturbation draws. The horizontal axis is the change in robust
# scatter in dex: nitrogen relative to Lux trained on complete stars in the top row; median abundance scatter
# relative to each method's clean-label baseline in the bottom row. Points show estimates and bars paired 95%
# bootstrap intervals. Negative values mean improvement. Compare correct and underestimated errors for Lux,
# and the retuned ridge control, before attributing robustness specifically to the use of label uncertainties.
#
# None of the three missing-label draws establishes a nitrogen improvement from the extra, incompletely labelled
# stars. For added noise, The Cannon's median abundance scatter increases by roughly 0.05–0.07 dex; Lux changes
# by less than 0.01 dex in these draws, even with underestimated errors. Retuning ridge matters and its change
# varies with the draw. The comparison supports Lux's robustness here, but does not isolate accurate error bars
# as its cause: treating labels as outputs and the latent representation also differ from The Cannon.
#
# **Regression checks.** For this fixed sample and noise level, allow a broad 0.03 dex lower bound on Cannon's
# degradation and a 0.02 dex upper bound on the absolute Lux change, on both held-out samples.

# %%
noise = experiments[experiments.experiment == 'noisy']
assert (noise[noise.model == 'Cannon'].delta > 0.03).all()
assert (noise[noise.model.str.startswith('Lux')].delta.abs() < 0.02).all()

# %% [markdown]
# ## C.4 Predicting spectra and measuring labels are different questions
#
# The Cannon predicts a spectrum from labels. Lux with 49 latents and 12 labels does not define a unique such
# prediction: labels constrain at most 12 latent directions. Below we choose the minimum-norm latent solution
# from labels alone and show explicitly that adding a null-space direction preserves the labels but changes
# the spectrum. Its spectral residual is conditional on that arbitrary convention; it is not a fair ranking
# of the two models' ability to predict spectra. Fitting latents to the very spectrum being predicted would
# instead measure reconstruction. A predictive distribution would need an additional, specified latent prior.
#
# Label response is well defined when spectra constrain the needed directions (Figure C.4). For a representative
# validation
# star compare d[Mg/Fe]/dflux for both methods, using the same pixels, and verify each response against a small
# numerical flux perturbation. The Cannon response uses the local Jacobian approximation; the finite difference
# checks its adequacy at this star, rather than assuming it is exact for nonlinear residuals.

# %%
A, B, s_lux = lux_model
th, s_can, pivot, scale = base['model']
ref = refs[0]
z_labels = ref @ np.linalg.pinv(A).T
predicted_lux = z_labels @ B.T
predicted_cannon = vectorize_n((ref - pivot) / scale) @ th.T
_, _, vh = np.linalg.svd(A, full_matrices=True)
null = vh[-1]
assert np.max(np.abs(A @ null)) < 1e-8
assert np.linalg.norm(B @ null) > 1e-6, 'chosen latent direction must change spectra'
good = np.isfinite(E[val]) & (th[:, 0] != 0)
variance = E[val]**2 + s_can**2
for name, prediction in [('Cannon', predicted_cannon), ('Lux, minimum-norm label latents', predicted_lux)]:
    q = np.sum(np.where(good, (F[val] - prediction)**2 / variance, 0), axis=1) / good.sum(axis=1)
    print(name, 'median spectral chi-squared per pixel:', np.median(q))
n = np.argmin(np.abs(ref[:, 0] - np.median(ref[:, 0])))
f, e = F[val][n], E[val][n]
ok = np.isfinite(e) & (th[:, 0] != 0)
mg = names.index('mg_fe')
fit, _ = predict_cannon(base['model'], f[None], e[None])
x = (fit[0] - pivot) / scale
J = th[ok] @ dvectorize_n(x)
w = 1 / (e[ok]**2 + s_can[ok]**2)
response_c = (np.linalg.pinv((J.T * w) @ J) @ (J.T * w))[mg] * scale[mg]
w_lux = 1 / (e[ok]**2 + s_lux[ok]**2)
response_l = (A @ np.linalg.pinv((B[ok].T * w_lux) @ B[ok], rcond=1e-12) @ (B[ok].T * w_lux))[mg]
# A symmetric perturbation avoids the first-order error of a forward finite difference.
p = np.argmax(np.abs(response_l))
pixel = np.flatnonzero(ok)[p]
h = 1e-5
fplus, fminus = f.copy(), f.copy()
fplus[pixel] += h
fminus[pixel] -= h
num_l = (lux_test(fplus[None], e[None], A, B, s_lux)[0][0, mg] -
         lux_test(fminus[None], e[None], A, B, s_lux)[0][0, mg]) / (2 * h)
num_c = (predict_cannon(base['model'], fplus[None], e[None], [x])[0][0, mg] -
         predict_cannon(base['model'], fminus[None], e[None], [x])[0][0, mg]) / (2 * h)
print('Response at pixel', pixel, '(analytic, finite difference): Lux', response_l[p], num_l,
      '; Cannon', response_c[p], num_c)
assert np.isclose(num_l, response_l[p], rtol=1e-3, atol=1e-4)
assert np.isclose(num_c, response_c[p], rtol=0.2, atol=0.01), 'local Jacobian approximation at this star'
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
window = (wave > 15730) & (wave < 15785)
axes[0].plot(wave[window], (B @ null)[window], label='flux change along one label-null direction')
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set(ylabel='change in normalised predicted flux')
axes[0].legend()
for response, label in [(response_c, 'Cannon'), (response_l, 'Lux')]:
    window_ok = window[ok]
    axes[1].plot(wave[ok][window_ok], response[window_ok], label=label)
axes[1].set(xlabel='vacuum wavelength [Å]', ylabel='d[Mg/Fe]/dflux [dex per unit flux]')
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure C.4.** A validation giant near the median temperature, in a window around Mg I. Top: two different
# Lux predictions differ by the plotted flux change when one unit null-space latent vector is added, although
# their labels are identical; predicting spectra from labels therefore requires an extra convention. Bottom:
# local response of inferred [Mg/Fe]
# (dex per unit normalised flux) to each pixel, for Cannon and Lux. The horizontal axis is vacuum wavelength in
# angstroms. These are inverse responses, not the forward flux derivatives plotted in earlier chapters.
#
# ## References
#
# This appendix extends the models and experimental controls in Chapters 3, 10 and 11:
# - Ness et al. 2015, [The Cannon](https://doi.org/10.1088/0004-637X/808/1/16).
# - Horta et al. 2025, [Lux](https://arxiv.org/abs/2502.01745).
#
# %%
print('Total runtime:', round(time.time() - t0), 'seconds')
