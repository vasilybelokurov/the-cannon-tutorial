# Stellar labels from spectra with The Cannon

A step-by-step Python tutorial on measuring stellar parameters and abundances from spectra with The Cannon
(Ness et al. 2015, [ApJ 808, 16](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N)), on APOGEE DR17
spectra, following the papers of Ness and collaborators. Students write the model themselves in numpy.

**Read online:** https://vasilybelokurov.github.io/the-cannon-tutorial/

Status: in development. Chapters 0–9 are built (release v0.1); the plan, including later chapters, is in
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
pytest tests/ --nbmake notebooks/
jupyter lab notebooks/
```

## Data

The data files (https://doi.org/10.5281/zenodo.23813051, CC BY 4.0) are extracts of APOGEE DR17 and Gaia DR3 made
with `scripts/build_data.py`, which records the exact query of each file in `data/MANIFEST.json`;
`scripts/zenodo_upload.py` publishes them. Sources, licences and acknowledgements:
[data/README.md](data/README.md).

## Licence

Tutorial code and text: MIT (see [LICENSE](LICENSE)). Data files: see [data/README.md](data/README.md).
