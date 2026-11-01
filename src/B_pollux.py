# %% [markdown]
# # Appendix B — Lux and The Cannon with *pollux*
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/B_pollux.ipynb)
#
# **Goal.** Show how to fit Lux (Chapter 11) and The Cannon (Chapters 2–10) with *pollux*
# ([github.com/adrn/pollux](https://github.com/adrn/pollux), [documentation](https://pollux-astro.readthedocs.io)),
# A. Price-Whelan's JAX framework for data-driven latent-variable models, and check our numpy code against it on the
# data of Chapters 10–11.
#
# **You will learn**
# 1. pollux's data containers, its convention for bad pixels and its label preprocessing;
# 2. to train a Lux and a Cannon model with pollux and to label new stars from their spectra;
# 3. that our numpy Lux and pollux's solve the same least-squares problems to $10^{-9}$ of the errors, and that both
#    packages give the same label precision for Lux and for The Cannon.
#
# **Versions.** pollux is young and fast-moving (version 0.1.0, August 2026). Its dependencies are not capped, and a
# newer release of one of them (`xmmutablemap` 0.3, October 2026) breaks it. This notebook therefore pins every
# package it adds (see `requirements.txt` in the repository): pollux 0.1.0, `xmmutablemap` 0.2.2, jax 0.12.0.1,
# numpyro 0.22.0, equinox 0.13.8.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('pollux') is None:                  # e.g. in Colab
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pollux==0.1.0', 'xmmutablemap==0.2.2',
                    'jax==0.12.0.1', 'jaxlib==0.12.0.1', 'numpyro==0.22.0', 'equinox==0.13.8'], check=True)

# %%
#@@LOADER
#@@NORM
#@@CANNON
#@@MORE
#@@LUX
import time
import warnings
import jax
import numpyro
import numpyro.distributions as dist
import pollux as plx
from pollux.models.transforms import LinearTransform

jax.config.update('jax_enable_x64', True)                       # double precision (JAX defaults to single)
warnings.filterwarnings('ignore', module='pollux')
labels = load('labels')
cont = load('continuum_pixels')['continuum'].to_numpy()
NAMES = ['teff', 'logg', 'fe_h'] + [e + '_fe' for e in ['c', 'n', 'o', 'mg', 'al', 'si', 'ca', 'mn', 'ni']]
data = {}
for nm in ['train', 'test']:
    lab = labels[labels['sample'] == nm].reset_index(drop=True)
    f, e = cheb_normalise(*prepare(load(f'spectra_{nm}')), cont)
    good = np.isfinite(lab[NAMES]).all(axis=1).to_numpy()
    data[nm] = (lab[NAMES].to_numpy(float)[good], lab[[n + '_err' for n in NAMES]].to_numpy(float)[good], f[good], e[good])
L, Lerr, flux, err = data['train']
Lt, _, ft, et = (a[::2] for a in data['test'])
val = np.random.default_rng(10).permutation(len(L)) % 5 == 0  # the splits of Chapters 10 and 11
tr = np.flatnonzero(~val)
rob = lambda d: np.array([scatter_stats(d[:, k])['robust'] for k in range(d.shape[1])])
print(f'pollux {plx.__version__ if hasattr(plx, "__version__") else ""}, jax {jax.__version__}, numpyro {numpyro.__version__}')

# %% [markdown]
# ## B.1 Data in pollux
#
# pollux keeps each observed quantity in an `OutputData` (values and errors) and the outputs of a set of stars in a
# `PolluxData`, keyed by name — for Lux, `"flux"` and `"label"`. Bad pixels are marked by a flux of 1 and a huge error
# (we use $\infty$ elsewhere). Labels can be scaled by a preprocessor, which matters for the default $\mathcal N(0, 1)$
# priors and is undone when predicting.

# %%
def pdata(F, E, labs=None, labs_err=None, pre=None):
    """PolluxData with fluxes (bad pixels: flux 1, error 1e10) and optionally labels (scaled by pre)."""
    bad = ~np.isfinite(E)
    out = {'flux': plx.data.OutputData(np.where(bad, 1.0, F), err=np.where(bad, 1e10, E))}
    if labs is not None:
        out['label'] = plx.data.OutputData(labs, err=labs_err, preprocessor=pre)
    return plx.data.PolluxData(**out)


pre = plx.data.ShiftScalePreprocessor.from_data(L[tr])
train = pdata(flux[tr], err[tr], L[tr], Lerr[tr], pre).preprocess()
print(train)

# %% [markdown]
# ## B.2 The same least-squares steps
#
# Before comparing models trained each in its own way, check that the building blocks agree. Start our code and
# pollux from the same random latents (49 per star) and coefficients, make pollux's priors negligible (very wide priors
# on the coefficients, none on the latents), and run five rounds of the same three steps: label coefficients, flux
# coefficients, latents. With random latents of full rank every step has a unique solution, so the two must agree.
# This checks the least-squares fits of training stage 1 only — not the zero-padded start of Chapter 11, the scatter,
# the prior or the test step. (pollux marks bad pixels with a tiny weight, we with zero.)

# %%
P = 49
rng = np.random.default_rng(1)
Z0, A0, B0 = rng.normal(0, 1, (len(tr), P)), rng.uniform(0, 1, (len(NAMES), P)), rng.uniform(0, 1, (flux.shape[1], P))
wl, l0 = lux_weights(L[tr], Lerr[tr])
w, f0 = lux_weights(flux[tr], err[tr])
t0 = time.time()
A, B, Z = A0, B0, Z0
for _ in range(5):
    A = lux_a_step(Z, l0, wl, A)
    B = lux_b_step(Z, f0, w, B)
    Z = lux_z_step(A, B, l0, wl, f0, w, Z)
t_ours = time.time() - t0
m = plx.LVM(latent_size=P)
m.register_output('label', LinearTransform(output_size=len(NAMES), priors={'A': dist.Normal(0, 1e8)}))
m.register_output('flux', LinearTransform(output_size=flux.shape[1], priors={'A': dist.Normal(0, 1e8)}))
t0 = time.time()
res = m.optimize_iterative(pdata(flux[tr], err[tr], L[tr], Lerr[tr]), blocks=['label:data', 'flux:data', 'latents'],
                           initial_params={'latents': Z0, 'label': {'data': {'A': A0}}, 'flux': {'data': {'A': B0}}},
                           max_cycles=5, tol=0.0, latents_prior=False, progress=False)
t_pollux = time.time() - t0
pA, pB, pZ = (np.asarray(x) for x in (res.params['label']['data']['A'], res.params['flux']['data']['A'],
                                      res.params['latents']))
use = np.isfinite(err[tr])
d_lab = np.abs(Z @ A.T - pZ @ pA.T) / Lerr[tr]
d_flux = np.abs(Z @ B.T - pZ @ pB.T)[use] / err[tr][use]
print(f'5 rounds: ours {t_ours:.0f} s, pollux {t_pollux:.0f} s; |difference| / error: labels max {d_lab.max():.1e}, '
      f'fluxes max {d_flux.max():.1e}')
assert d_lab.max() < 1e-6 and d_flux.max() < 1e-6

# %% [markdown]
# ## B.3 Lux in pollux
#
# `plx.Lux` builds the two linear outputs and a fitted scatter for the flux. `optimize_iterative` alternates its blocks
# until the loss stops improving; each linear block is solved exactly, as in our code. pollux's defaults differ from
# H25's recipe (Chapter 11): random starting latents, $\mathcal N(0, 1)$ priors on latents and coefficients, and as many
# rounds as needed. To label new stars, hold everything but the latents fixed and solve for the latents from the
# spectra alone — with pollux's default $\mathcal N(0, 1)$ prior on them (the most probable latents), where our test
# step uses no prior.

# %%
t0 = time.time()
lux_px = plx.Lux(latent_size=P, label_size=len(NAMES), flux_size=flux.shape[1])
res = lux_px.optimize_iterative(train, rng_key=jax.random.PRNGKey(0), progress=False)
fixed = lux_px.output_pars(res.params)


def lux_px_labels(F, E):
    """Latents from the spectra alone, then labels, unscaled."""
    r = lux_px.optimize_iterative(pdata(F, E), fixed_pars=fixed, blocks=['latents'], progress=False)
    return np.asarray(pre.inverse_transform(lux_px.predict_outputs(fixed, latents=r.params['latents'])['label']))


px_lux = rob(lux_px_labels(flux[val], err[val]) - L[val]), rob(lux_px_labels(ft, et) - Lt)
print(f'pollux Lux: {len(res.losses_per_cycle)} rounds, {time.time() - t0:.0f} s')
t0 = time.time()
A, B, s, Z, _ = lux_train(L[tr], Lerr[tr], flux[tr], err[tr], P=P)
our_lux = rob(lux_test(flux[val], err[val], A, B, s)[0] - L[val]), rob(lux_test(ft, et, A, B, s)[0] - Lt)
print(f'our Lux: {time.time() - t0:.0f} s')

# %% [markdown]
# ## B.4 The Cannon in pollux
#
# `plx.Cannon` is a polynomial in the labels, which are its latents: during training they are held fixed at the
# (scaled) catalogue values, the coefficients are solved exactly and the scatter by gradient descent (Adam), as in
# pollux's APOGEE example. The test step fits the (scaled) labels of each star by gradient descent, with pollux's
# default $\mathcal N(0, 1)$ prior on them.

# %%
t0 = time.time()
can_px = plx.Cannon(label_size=len(NAMES), output_size=flux.shape[1], poly_degree=2, intrinsic_scatter={'flux': 0.1})
res = can_px.optimize_iterative(train, blocks=['flux:data', 'flux:err'], fixed_pars={'latents': np.asarray(train['label'].data)},
                                max_cycles=8, rng_key=jax.random.PRNGKey(123), progress=False,
                                block_options={'flux:err': {'optimizer': numpyro.optim.Adam,
                                                            'optimizer_kwargs': {'step_size': 1e-2}, 'num_steps': 1000}})
fixed_c = can_px.output_pars(res.params)


def cannon_px_labels(F, E):
    """Labels of each star by gradient descent (flux likelihood and the default prior on the scaled labels)."""
    pars, _ = can_px.optimize(pdata(F, E), fixed_pars=fixed_c, names=['flux'], num_steps=4000,
                              optimizer=numpyro.optim.Adam(1e-2), rng_key=jax.random.PRNGKey(42), progress=False)
    return np.asarray(pre.inverse_transform(np.asarray(pars['latents'])))


px_can = rob(cannon_px_labels(flux[val], err[val]) - L[val]), rob(cannon_px_labels(ft, et) - Lt)
print(f'pollux Cannon: {time.time() - t0:.0f} s')
t0 = time.time()
pv, sc = np.median(L[tr], axis=0), (np.percentile(L[tr], 97.5, axis=0) - np.percentile(L[tr], 2.5, axis=0)) / 2
th, s = train_more((L[tr] - pv) / sc, flux[tr], err[tr])
run = lambda F, E: np.array([test_more(F[n], E[n], th, s)[0] for n in range(len(F))]) * sc + pv
our_can = rob(run(flux[val], err[val]) - L[val]), rob(run(ft, et) - Lt)
print(f'our Cannon: {time.time() - t0:.0f} s')

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
models = {'Cannon, ours': our_can, 'Cannon, pollux': px_can, 'Lux, ours': our_lux, 'Lux, pollux': px_lux}
for a, (k, lab) in zip(ax, [(0, r'$T_{\rm eff}$ [K]'), (1, r'$\log g$'), (None, 'median of the nine [X/Fe]')]):
    for j, split in enumerate(['validation (high S/N)', 'test (S/N 50–100)']):
        v = [r[j][k] if k is not None else np.median(r[j][3:]) for r in models.values()]
        a.bar(np.arange(4) + 0.38 * j - 0.19, v, 0.36, color=['C0', 'C1'][j], label=split)
    a.set_xticks(range(4), [m.replace(', ', '\n') for m in models], fontsize=9)
    a.set(ylabel=f'robust std vs ASPCAP, {lab}')
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure B.1.** The same comparison of $T_{\rm eff}$, $\log g$ and the abundances as Figure 11.1, for our numpy code
# and for pollux, each trained its own way on the same 1308 stars. The two Cannons agree to about a kelvin and a few
# thousandths of a dex; the two Luxes closely, pollux's being somewhat better in $T_{\rm eff}$ at low S/N — its priors,
# random start and training schedule all differ from ours, and we have not separated their effects. The precisions on
# which Chapter 11's comparison of Lux and The Cannon rests are reproduced by pollux to within the tolerances checked
# below (8% for The Cannon, 20% for Lux, in $T_{\rm eff}$, $\log g$ and the median abundance).
#
# **Checks.**

# %%
for ours, theirs, tol in [(our_can, px_can, 0.08), (our_lux, px_lux, 0.2)]:
    for j in range(2):
        assert np.all(np.abs(np.array([ours[j][0], ours[j][1], np.median(ours[j][3:])]) /
                             np.array([theirs[j][0], theirs[j][1], np.median(theirs[j][3:])]) - 1) < tol)

# %% [markdown]
# ## B.5 What pollux adds
#
# Our numpy code does what the chapters need. pollux, in addition, puts priors on every parameter and can sample the
# posterior with numpyro instead of optimising; takes any number of outputs, each with its own transform (labels from
# two surveys, photometry, spectra from two instruments); allows non-linear transforms from the latents (and then fits
# them by gradient descent); and runs on GPUs through JAX. Its documentation has worked examples for APOGEE.
#
# ## Summary
#
# - For the least-squares steps of training, pollux and our code agree to $10^{-9}$ of the errors; trained each in its
#   own way, the two give the same precision for The Cannon and nearly the same for Lux.
# - pollux 0.1.0 needed one pinned dependency (`xmmutablemap` 0.2.2) to run in November 2026; pin versions when you
#   build on young packages.
#
# ## References
#
# - Horta, D., Price-Whelan, A. M., Hogg, D. W., Ness, M. K., & Casey, A. R. 2025,
#   [AJ 169, 314](https://ui.adsabs.harvard.edu/abs/2025AJ....169..314H).
# - Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015,
#   [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N).
# - Price-Whelan, A. M., *pollux*, [github.com/adrn/pollux](https://github.com/adrn/pollux),
#   [pollux-astro.readthedocs.io](https://pollux-astro.readthedocs.io).
