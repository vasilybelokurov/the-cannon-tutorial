"""Make _static/cannon_principle.png, the summary figure of the landing page, from the tutorial's own data.

Usage: python scripts/make_principle_figure.py   (needs the data files in notebooks/data, e.g. after running Chapter 4)

The model is the three-label Cannon of Chapter 4 (continuum pixels of Chapter 3, quadratic label vector, per-pixel
scatter), trained on all pixels of the training set; the figure shows a 50 A window.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "notebooks" / "data"
wave = 10**(4.179 + 6e-6 * np.arange(8575))
CHIPS = [(15150, 15800), (15865, 16425), (16480, 16950)]
BAD_BITS = sum(1 << b for b in [0, 1, 2, 3, 4, 5, 6, 7, 12])
NAMES = ["teff", "logg", "fe_h"]


def prepare(sp):
    on = np.any([(wave > lo) & (wave < hi) for lo, hi in CHIPS], axis=0)
    f, e = sp["flux"].astype(float), sp["err"].astype(float)
    bad = ((sp["mask"] & BAD_BITS) > 0) | ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0) | ~on
    return np.where(bad, 0.0, f), np.where(bad, np.inf, e)


def normalise(f, e, cont):
    fn, en = np.zeros_like(f), np.full_like(e, np.inf)
    for lo, hi in CHIPS:
        c = (wave > lo) & (wave < hi)
        B = np.polynomial.chebyshev.chebvander((wave[c] - (lo + hi) / 2) / ((hi - lo) / 2), 2)
        u, w = cont[c], 1 / e[:, c][:, cont[c]]**2
        a = np.linalg.solve(np.einsum("pi,np,pj->nij", B[u], w, B[u]),
                            np.einsum("pi,np->ni", B[u], w * f[:, c][:, u])[:, :, None])[:, :, 0]
        fn[:, c], en[:, c] = f[:, c] / (a @ B.T), e[:, c] / (a @ B.T)
    return fn, en


def vec(x):
    x = np.atleast_2d(x)
    k, l = np.triu_indices(x.shape[1])
    return np.hstack([np.ones((len(x), 1)), x, x[:, k] * x[:, l]])


def train(x, f, e):
    V, e2, s = vec(x), e**2, np.zeros(f.shape[1])
    grid = np.r_[0, np.logspace(-4, 0, 41)]
    for _ in range(3):
        w = 1 / (e2 + s**2)
        fw = np.where(w > 0, f, 0)
        A = ((V[:, :, None] * V[:, None, :]).reshape(len(V), -1).T @ w).T.reshape(-1, V.shape[1], V.shape[1])
        th = np.einsum("pde,pe->pd", np.linalg.pinv(A), (V.T @ (w * fw)).T)
        r2 = np.where(np.isfinite(e2), (f - V @ th.T)**2, 0)
        ee = np.where(np.isfinite(e2), e2, 1e30)
        s = grid[np.argmin([np.sum(r2 / (ee + g**2) + np.log(ee + g**2), axis=0) for g in grid], axis=0)]
    return th, s


def test(f, e, th, s):
    ok = np.isfinite(e) & (th[:, 0] != 0)
    w = 1 / np.sqrt(e[ok]**2 + s[ok]**2)
    r = least_squares(lambda x: (f[ok] - th[ok] @ vec(x)[0]) * w, np.zeros(3), method="lm")
    return r.x, np.sqrt(np.diag(np.linalg.inv(r.jac.T @ r.jac)))


lab = pd.read_parquet(DATA / "labels.parquet")
tr = lab[lab["sample"] == "train"].reset_index(drop=True)
te = lab[lab["sample"] == "test"].reset_index(drop=True)
cont = pd.read_parquet(DATA / "continuum_pixels.parquet").continuum.to_numpy()
f, e = normalise(*prepare(dict(np.load(DATA / "spectra_train.npz"))), cont)
L = tr[NAMES].to_numpy()
pv, sc = np.median(L, 0), (np.percentile(L, 97.5, 0) - np.percentile(L, 2.5, 0)) / 2
th, s = train((L - pv) / sc, f, e)
ft, et = normalise(*prepare(dict(np.load(DATA / "spectra_test.npz"))), cont)
j = int(np.argmin(np.abs(te.snr - 60) + 0.1 * np.abs(te.teff - 4700)))     # a test star at S/N ~ 60
xj, sj = test(ft[j], et[j], th, s)
lab_j, err_j = xj * sc + pv, sj * sc

win = (wave > 15742) & (wave < 15792)
pix = np.argmin(np.abs(wave - 15753.29))                                   # core of a Mg I line

# ---------------------------------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.42, left=0.05, right=0.98, top=0.86, bottom=0.07)
fig.text(0.5, 0.965, "The Cannon in one picture: learn how the spectrum depends on the labels, then invert",
         ha="center", fontsize=15, weight="bold")
fig.text(0.05, 0.915, "① TRAINING STEP  (reference stars: spectra and labels known → coefficients, one pixel at a time)",
         fontsize=11.5, color="C0", weight="bold")
fig.text(0.05, 0.455, "② TEST STEP  (new star: spectrum known, labels unknown → labels, one star at a time)",
         fontsize=11.5, color="C3", weight="bold")

# A: training spectra coloured by T_eff
a = fig.add_subplot(gs[0, 0])
sel = np.flatnonzero((np.abs(tr.fe_h + 0.3) < 0.1) & (np.abs(tr.logg - 2.4) < 0.5))
sel = sel[np.argsort(tr.teff[sel])][::max(1, len(sel) // 9)]
cm = plt.get_cmap("plasma")
for n in sel:
    a.plot(wave[win], np.where(np.isfinite(e[n, win]), f[n, win], np.nan), color=cm((tr.teff[n] - 4000) / 1300), lw=1)
a.axvline(wave[pix], color="k", ls=":", lw=0.8)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(4000, 5300))
plt.colorbar(sm, ax=a, label=r"$T_{\rm eff}$ [K]", pad=0.01)
a.set(xlabel="wavelength [Å]", ylabel="normalised flux", ylim=(0.45, 1.08),
      title=f"Reference spectra with known labels\n({len(tr)} APOGEE giants; 9 shown, [Fe/H] ≈ −0.3)")

# B: one pixel against two labels, with the quadratic model
b = fig.add_subplot(gs[0, 1])
ok = np.isfinite(e[:, pix])
b.scatter(tr.fe_h[ok], f[ok, pix], c=tr.teff[ok], cmap=cm, vmin=4000, vmax=5300, s=4, alpha=0.7)
fe = np.linspace(-1.5, 0.4, 50)
for t in [4200, 4700, 5200]:
    lg = np.interp(t, [4000, 5300], [1.2, 3.3])                             # a typical giant-branch log g at this T_eff
    xx = (np.c_[np.full(50, t), np.full(50, lg), fe] - pv) / sc
    b.plot(fe, vec(xx) @ th[pix], color=cm((t - 4000) / 1300), lw=2.5)
    b.plot(fe, vec(xx) @ th[pix], color="k", lw=0.6)
b.set(xlabel="[Fe/H] of the reference star", ylabel=f"flux at {wave[pix]:.1f} Å",
      title="At every pixel: fit the flux as a quadratic\nfunction of the labels (curves: model at 3 $T_{\\rm eff}$)")
b.text(0.03, 0.04, r"$f_{n\lambda} = \boldsymbol{\theta}_\lambda \cdot \mathbf{v}(\ell_n) + \varepsilon$,"
       "\n" r"$\varepsilon \sim \mathcal{N}(0,\ \sigma_{n\lambda}^2 + s_\lambda^2)$", transform=b.transAxes, fontsize=10,
       bbox=dict(fc="w", ec="0.7"))

# C: the model = coefficient spectra (baseline with scatter on top, derivatives below)
sub = gs[0, 2].subgridspec(2, 1, hspace=0.08)
c = fig.add_subplot(sub[0])
c.plot(wave[win], th[win, 0], "k", lw=1.2, label=r"$\theta_0$ (median labels)")
c.fill_between(wave[win], th[win, 0] - 3 * s[win], th[win, 0] + 3 * s[win], color="0.6", alpha=0.6, label=r"$\pm 3 s_\lambda$")
c.axvline(wave[pix], color="k", ls=":", lw=0.8)
c.set(ylabel="flux", ylim=(0.45, 1.1), title="The model: a baseline spectrum, its derivatives\nwith respect to each label, and a scatter")
c.tick_params(labelbottom=False)
c.legend(fontsize=7, loc="lower right", ncol=2)
c2 = fig.add_subplot(sub[1], sharex=c)
for k, (nm, col, fac) in enumerate([("T_{\\rm eff}", "C1", 1e3), ("\\log g", "C2", 1), ("[{\\rm Fe/H}]", "C0", 1)]):
    c2.plot(wave[win], th[win, k + 1] / sc[k] * fac, color=col, lw=1,
            label=rf"$\partial f/\partial {nm}$" + (" /1000 K" if k == 0 else " /dex"))
c2.axhline(0, color="k", lw=0.4)
c2.axvline(wave[pix], color="k", ls=":", lw=0.8)
c2.set(xlabel="wavelength [Å]", ylabel="derivative", ylim=(-0.25, 0.32))
c2.legend(fontsize=7, loc="lower right", ncol=3, handlelength=1)

# D: test star, observed vs model at the fitted labels
d = fig.add_subplot(gs[1, :2])
okw = win & np.isfinite(et[j])
d.errorbar(wave[okw], ft[j, okw], et[j, okw], fmt=".", color="0.3", ms=3, lw=0.5, label=f"new star, S/N = {te.snr[j]:.0f}")
d.plot(wave[win], vec((lab_j - pv) / sc)[0] @ th[win].T, "C3", lw=2, label="model at the inferred labels")
d.plot(wave[win], th[win, 0], "k", lw=0.8, ls="--", label="model at the median labels (starting point)")
d.set(xlabel="wavelength [Å]", ylabel="normalised flux", ylim=(0.45, 1.12),
      title="Find the labels whose model spectrum best matches the new spectrum (fit to all ≈ 7500 pixels; 50 Å shown)")
d.legend(fontsize=8, loc="lower left")
d.text(0.99, 0.05, f"inferred:  $T_{{\\rm eff}}$ = {lab_j[0]:.0f} K,  log g = {lab_j[1]:.2f},  [Fe/H] = {lab_j[2]:+.2f}\n"
       f"APOGEE:   $T_{{\\rm eff}}$ = {te.teff[j]:.0f} K,  log g = {te.logg[j]:.2f},  [Fe/H] = {te.fe_h[j]:+.2f}",
       transform=d.transAxes, ha="right", va="bottom", fontsize=9.5, family="monospace", bbox=dict(fc="w", ec="C3"))

# E: what it is good for
g = fig.add_subplot(gs[1, 2])
g.axis("off")
g.text(0.0, 0.97, "Why it works, and its limits", va="top", fontsize=11, weight="bold")
g.text(0.0, 0.85, ""
       "• No physics in the model: it is learned from\n  reference stars, so its labels are on their scale.\n"
       "• An explicit noise model: trained at high S/N,\n  it labels noisy spectra (here S/N 50–100).\n"
       "• Fast: training is linear algebra; the test step\n  fits only a few numbers per star.\n"
       "• It interpolates, it cannot extrapolate: stars unlike\n  the reference set get wrong labels (Ch. 5, 7).\n"
       "• A label can be predicted from correlations rather\n  than measured from its own lines (Ch. 8).",
       va="top", fontsize=10, linespacing=1.35)

arr = dict(arrowstyle="-|>", lw=2.5, color="0.25", mutation_scale=20)
fig.canvas.draw()                                            # final axes positions, for the arrows
pa, pb, pc = a.get_position(), b.get_position(), c.get_position()
y = 0.5 * (pb.y0 + pb.y1)
fig.add_artist(FancyArrowPatch((pa.x1 + 0.065, y), (pb.x0 - 0.045, y), transform=fig.transFigure, **arr))
fig.add_artist(FancyArrowPatch((pb.x1 + 0.005, y), (pc.x0 - 0.045, y), transform=fig.transFigure, **arr))
fig.add_artist(FancyArrowPatch((pc.x1 - 0.02, 0.47), (0.66, 0.40), transform=fig.transFigure, connectionstyle="arc3,rad=-0.2", **arr))
fig.text(0.80, 0.475, "use the model", fontsize=9.5, color="0.25", style="italic", ha="right")
out = ROOT / "_static" / "cannon_principle.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}; test star {te.apogee_id[j]}: inferred {lab_j.round(2)} +- {err_j.round(3)}, APOGEE {te[NAMES].iloc[j].to_numpy().round(2)}")
