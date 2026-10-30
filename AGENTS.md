# AGENTS.md

## Project

A Python tutorial, in Jupyter notebooks published from a public git repo, on measuring stellar labels
(T_eff, log g, [Fe/H], element abundances) from spectra with **The Cannon**, the data-driven spectral model
of Ness et al. 2015 ([ApJ 808, 16](https://doi.org/10.1088/0004-637X/808/1/16)). It starts from raw APOGEE
DR17 spectra and builds the method step by step, following the papers of Ness and collaborators.
Students write The Cannon themselves in numpy. **Public data only** (APOGEE DR17, Gaia DR3).

## Rules

1. **Write the least code that works.**
2. **Reproduce every bug with a failing test** before fixing it.
3. **Show a plot before claiming a result.**

## Workflow

- Chapters are `notebooks/NN_<slug>.ipynb`, committed without outputs; each runs standalone in Colab.
- Students have no WSDB access: `scripts/build_data.py` queries WSDB (sqlutilpy) and the SDSS Science
  Archive Server and writes the files that notebooks download (with a SHA-256 check against `data/MANIFEST.json`).
- Each notebook backs every claim with a plot and an `assert` with a stated tolerance against a known result
  (a synthetic truth, a published number, or a held-out measurement).
- Every figure has a numbered caption (**Figure N.k.**) that states what is plotted, the sample, the axes
  and units, and what the reader should see; the text refers to figures by number.
- Comment generously for beginners: the physics or statistics a line implements, numpy idioms (shapes,
  broadcasting, `axis=`), and the arguments of scipy calls.
- Before pushing: `pytest tests/ --nbmake notebooks/` and `jupyter-book build . -W -n --keep-going`.
- Every literature reference carries a working ADS/arXiv/DOI link, checked before it is committed.
