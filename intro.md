# Stellar labels from spectra with The Cannon

This tutorial teaches how to measure the parameters and chemical abundances of stars — their "labels":
effective temperature, surface gravity, metallicity, element abundances — from their spectra with
**The Cannon**, the data-driven spectral model of Ness et al. (2015). It starts from the spectra of the APOGEE
survey as they come out of the reduction pipeline and builds the method step by step: you write The Cannon
yourself in a few dozen lines of numpy, train it, validate it in several independent ways, push it outside its
comfort zone, and finally transfer APOGEE's labels to the spectra of another instrument, Gaia's RVS.

Each chapter is a Jupyter notebook that runs on its own (in Google Colab, or locally), downloads the small data
files it needs, and backs every claim with a figure and an explicit numerical check (`assert`).

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

Later chapters (masses and ages of red giants, a regularised model with many abundances, chemical
doppelgängers) are planned in `STRUCTURE.md` in the repository.

**Prerequisites.** Basic Python (numpy, matplotlib), basic statistics (least squares, Gaussian likelihood) and
elementary stellar astrophysics (what $T_{\rm eff}$, $\log g$ and [Fe/H] mean).

**Main references.** Ness et al. 2015, *The Cannon: a data-driven approach to stellar label determination*,
[ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N); Casey et al. 2016, *The Cannon 2*,
[arXiv:1603.03040](https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C); Ness et al. 2016,
[ApJ 823, 114](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..114N); Ho et al. 2017,
[ApJ 836, 5](https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H); Ness et al. 2018,
[ApJ 853, 198](https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N).

**Data.** APOGEE DR17 spectra and labels (SDSS Science Archive Server and the WSDB database in Cambridge) and
Gaia DR3 RVS spectra. See `data/README.md` for sources and licences.
