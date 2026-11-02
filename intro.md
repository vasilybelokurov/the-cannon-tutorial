# Stellar labels from spectra with The Cannon

This tutorial teaches how to measure the parameters and chemical abundances of stars — their "labels":
effective temperature, surface gravity, metallicity, element abundances — from their spectra with
**The Cannon**, the data-driven spectral model of Ness et al. (2015). It starts from the spectra of the APOGEE
survey as they come out of the reduction pipeline and builds the method step by step: you write The Cannon
yourself in a few dozen lines of numpy, train it, validate it in several independent ways, push it outside its
comfort zone, and finally transfer APOGEE's labels to the spectra of another instrument, Gaia's RVS.

Each chapter is a Jupyter notebook that runs on its own (in Google Colab, or locally), downloads the small data
files it needs, and backs every claim with a figure and an explicit numerical check (`assert`).

## What The Cannon does, in brief

**The problem.** A stellar spectrum depends on a handful of numbers — effective temperature $T_{\rm eff}$, surface
gravity $\log g$, metallicity [Fe/H], abundances [X/Fe] — that astronomers call the star's *labels*. The classical
way to measure them is to compute synthetic spectra from model atmospheres and a list of atomic and molecular
lines, and to find the model that fits. The models are imperfect (one-dimensional atmospheres, local
thermodynamic equilibrium, incomplete line lists), so they never fit every pixel of a high-quality spectrum.

**The idea.** The Cannon (Ness et al. 2015) needs no stellar models. It assumes only that stars with the same
labels have the same spectrum, and that the spectrum changes smoothly with the labels. Then it *learns* how the
spectrum depends on the labels from **reference stars** whose labels are already known, and uses what it learned
to label other stars observed with the same instrument.

**How, in two steps** (see the figure):

1. **Training step — labels known, model unknown.** Take a few hundred to a few thousand reference stars with
   continuum-normalised spectra $f_{n\lambda}$ (star $n$, pixel $\lambda$) and labels $\boldsymbol\ell_n$. At
   *every pixel separately*, fit the flux as a quadratic polynomial of the labels,
   $f_{n\lambda} = \boldsymbol\theta_\lambda \cdot \mathbf v(\boldsymbol\ell_n) + {\rm noise}$, where
   $\mathbf v = (1, \ell_1, \ell_2, \dots, \ell_1^2, \ell_1\ell_2, \dots)$. The noise has the measured variance
   $\sigma_{n\lambda}^2$ plus a fitted "scatter" $s_\lambda^2$ that absorbs what the labels do not describe. Because
   the model is linear in the coefficients $\boldsymbol\theta_\lambda$, this is weighted linear least squares, done
   for all pixels at once in seconds. The coefficients are a *spectral model*: a baseline spectrum and its
   derivatives with respect to each label.
2. **Test step — model known, labels unknown.** For a new star, find the labels whose model spectrum best fits the
   observed spectrum, using all pixels at once (a non-linear least-squares fit of a few numbers). Because the
   noise model is explicit, a model trained on high-S/N spectra can label much noisier ones.

**What you get, and what you must provide.** Labels on the *scale of the reference set* (if the reference labels
are biased, so are the results), precise because every pixel contributes, and fast enough for millions of stars.
In return, the reference stars must be observed with the same instrument, all spectra must be normalised the same
way, and the reference set must cover the labels of the stars you want to measure: The Cannon interpolates, it
does not extrapolate. In this tutorial a three-label model trained on 1700 APOGEE giants reproduces the APOGEE
labels of stars it has not seen to 72 K, 0.19 dex and 0.06 dex (Chapter 5) and transfers them to Gaia's
lower-resolution RVS spectra (Chapter 9); the chapters also show where it fails.

```{figure} _static/cannon_principle.png
:name: fig-principle
:width: 100%

The Cannon in one picture, made from the data of this tutorial (`scripts/make_principle_figure.py`).
**Training step** (top): the continuum-normalised spectra of reference stars with known labels (left, nine APOGEE
giants of similar [Fe/H] coloured by $T_{\rm eff}$, in a 50 Å window); at every pixel, the flux of all 1703 reference
stars is fitted as a quadratic function of their labels (middle, for the pixel at the core of the Mg I line at
15 753 Å, marked by the dotted line: the flux falls with [Fe/H] and rises with $T_{\rm eff}$; the curves are the
fitted model at three temperatures); the result is a spectral model — a baseline spectrum with its scatter, and its
derivatives with respect to $T_{\rm eff}$, $\log g$ and [Fe/H] (right). **Test step** (bottom): for a new star with
S/N = 62 that was not used in training, the labels are those whose model spectrum (red) best fits the observed
spectrum (grey points with error bars); the dashed line is the model at the median training labels, where the fit
starts. The inferred labels agree with APOGEE's within 2 K, 0.09 dex and 0.04 dex.
```

**The code.** The tutorial does not use any of the public implementations of The Cannon: the model is written
from scratch in Chapter 2, from the equations of Ness et al. (2015) and Casey et al. (2016), with numpy and scipy
only, and reused in the later chapters. For research use, see the published packages:
[TheCannon](https://github.com/annayqho/TheCannon) (Ho et al. 2016, [ascl:1602.010](https://ui.adsabs.harvard.edu/abs/2016ascl.soft02010H))
and [AnniesLasso](https://github.com/andycasey/AnniesLasso) (*The Cannon 2*, Casey et al. 2016). The appendix checks our
code against AnniesLasso 0.2.96: run the same way, the two agree to numerical precision. For research use, the
[pollux](https://github.com/adrn/pollux) package (A. Price-Whelan) implements both The Cannon and Lux (Chapter 11)
in JAX; Appendix B shows how to use it and checks our code against it.

**Chapters**

0. APOGEE spectra and their labels
1. How spectra depend on labels: lines, and one pixel at a time
2. The Cannon on synthetic spectra: training step, test step, scatter, pulls
3. Continuum normalisation, and why it must not depend on S/N
4. Training The Cannon on APOGEE spectra: what the model learns
5. Validation: cross-validation, linear vs quadratic, formal errors, extrapolation
6. Low S/N: repeat visits of the same stars
7. Star clusters as a test
8. More labels: element abundances, and whether they are measured or predicted
9. Label transfer to Gaia RVS spectra
10. Many labels: flexibility, L1 regularisation, censoring
11. Beyond The Cannon: a latent-variable model (Lux)

Appendix A: our Cannon against a published implementation (AnniesLasso)

Appendix B: Lux and The Cannon with *pollux*

Appendix C: stability of the training-size and Lux comparisons — repeated splits, missing and noisy labels,
training-only continuum selection, and paired uncertainty intervals

Later chapters (masses and ages of red giants, chemical doppelgängers) are planned in `STRUCTURE.md` in the repository.

**Prerequisites.** Basic Python (numpy, matplotlib), basic statistics (least squares, Gaussian likelihood) and
elementary stellar astrophysics (what $T_{\rm eff}$, $\log g$ and [Fe/H] mean).

**Main references.** Ness et al. 2015, *The Cannon: a data-driven approach to stellar label determination*,
[ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N); Casey et al. 2016, *The Cannon 2*,
[arXiv:1603.03040](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C); Ness et al. 2016,
[ApJ 823, 114](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..114N); Ho et al. 2017,
[ApJ 836, 5](https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H); Ness et al. 2018,
[ApJ 853, 198](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N).

**Data.** APOGEE DR17 spectra and labels (SDSS Science Archive Server and the WSDB database in Cambridge) and
Gaia DR3 RVS spectra, published for this tutorial on Zenodo,
[doi:10.5281/zenodo.23813051](https://doi.org/10.5281/zenodo.23813051). See `data/README.md` for sources and licences.
