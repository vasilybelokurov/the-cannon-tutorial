import hashlib, json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Data access (identical in every chapter) -------------------------------------------------------------
# The data files are extracts of APOGEE DR17 and Gaia DR3, published on Zenodo (https://doi.org/10.5281/zenodo.23813051).
# MANIFEST.json lists, for each file, the query that made it and its SHA-256 checksum.
DATA_URL = 'https://zenodo.org/records/23813051/files/{}?download=1'
_local = Path('../data/MANIFEST.json')            # present when you run inside a clone of the repository
MANIFEST = json.loads(_local.read_text() if _local.exists() else urllib.request.urlopen(DATA_URL.format('MANIFEST.json')).read())


def load(name, cache=Path('data')):
    """Download <name> (.parquet table or .npz arrays) from Zenodo if needed, check its SHA-256, read it.

    A .parquet file is returned as a pandas DataFrame, an .npz file as a dict of numpy arrays.
    """
    ext = '.npz' if name.startswith(('spectra_', 'rvs_')) else '.parquet'
    path = cache / f'{name}{ext}'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if not path.exists() or sha(path) != MANIFEST[name]['sha256']:   # first use, or a stale/damaged copy
        cache.mkdir(exist_ok=True)
        print(f'downloading {path.name} ...')
        part = path.with_suffix('.part')
        urllib.request.urlretrieve(DATA_URL.format(path.name), part)   # download to a temporary file first
        if sha(part) != MANIFEST[name]['sha256']:
            raise IOError(f'{path.name}: checksum does not match the manifest (incomplete download?)')
        part.replace(path)                                         # only a verified file takes the final name
    # the checksum guarantees that everyone runs this chapter on exactly the same data
    if ext == '.parquet':
        return pd.read_parquet(path)
    with np.load(path, allow_pickle=False) as f:   # allow_pickle=False: arrays only, no code can be loaded
        return {k: f[k] for k in f.files}
