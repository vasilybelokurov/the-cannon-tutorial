# Stellar labels from spectra with The Cannon

A step-by-step Python tutorial on measuring stellar parameters and abundances from spectra with The Cannon
(Ness et al. 2015, [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)), on APOGEE DR17
spectra, following the papers of Ness and collaborators. Students write the model themselves in numpy.

**Read online:** https://vasilybelokurov.github.io/the-cannon-tutorial/

Chapters 0–11 and Appendices A–C are implemented. Appendix C checks the stability of the training-size and Lux
comparisons with repeated splits and perturbations. The plan, including later chapters, is in
[STRUCTURE.md](STRUCTURE.md).

## Run in the browser

Open any notebook in Google Colab, e.g.
[Chapter 0](https://colab.research.google.com/github/vasilybelokurov/the-cannon-tutorial/blob/main/notebooks/00_apogee_spectra.ipynb).
Each notebook downloads the data files it needs from Zenodo,
[doi:10.5281/zenodo.23813051](https://doi.org/10.5281/zenodo.23813051), and checks their SHA-256 checksums. A
changed selection gets a new file name in a new version of the record, so a published checksum never changes.

## Run locally

```bash
pip install -r requirements.txt jupyter pytest nbmake
pytest tests/ --nbmake notebooks/ --nbmake-timeout=3600
jupyter lab notebooks/
```

To build the website, also install `jupyter-book==1.0.4.post1`. Execute the notebooks with
`pytest tests/ --nbmake notebooks/ --nbmake-timeout=3600 --overwrite`, then run
`jupyter-book build . -W -n --keep-going`. Regenerate notebooks from `src/` before committing so committed
notebooks contain no outputs. Appendix B installs the pinned pollux/JAX stack; Appendix C runs several dozen fits and takes longer
than the introductory chapters.

## Data

The data files (https://doi.org/10.5281/zenodo.23813051, CC BY 4.0) are extracts of APOGEE DR17 and Gaia DR3 made
with `scripts/build_data.py`, which records the exact query of each file in `data/MANIFEST.json`;
`scripts/zenodo_upload.py` publishes them. Sources, licences and acknowledgements:
[data/README.md](data/README.md).

## Licence

Tutorial code and text: MIT (see [LICENSE](LICENSE)). Data files: see [data/README.md](data/README.md).
