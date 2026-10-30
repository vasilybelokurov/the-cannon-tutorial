"""Build the data files of the tutorial; they are attached to a GitHub release, not stored in git.

Usage:  python scripts/build_data.py [name ...|all]   (writes data_build/<name>.{parquet,npz} and data/MANIFEST.json)
        python scripts/build_data.py continuum           (afterwards: runs Chapter 3 to make continuum_pixels)

Sources
- Labels: the APOGEE DR17 allStar catalogue (`apogee_dr17.allstar` in WSDB, Cambridge; the same table is
  allStar-dr17-synspec_rev1.fits on the SDSS Science Archive Server).
- Spectra: DR17 apStar files from the SDSS SAS, https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/
  We keep row 0 (the combined spectrum) of HDU 1-3 (flux, error, APOGEE_PIXMASK bitmask), and for
  `spectra_visits` also the single-visit rows (rows 2.. when NVISITS > 1).
- Gaia DR3 RVS mean spectra: `gaia_dr3.rvs_mean_spectrum` in WSDB (= Gaia archive table of the same name).
- Line list: the DR17 Turbospectrum atomic line list (Smith et al. 2021), SAS speclib/linelists/turbospec.

Spectra are stored in .npz files: flux and error divided by the median flux of each spectrum, the flux as
float32 and the error as float16 (relative rounding <= 2^-11 = 5e-4 of the error itself), the bitmask as int32, and
the median as float32. Pixels with no data are NaN in the flux, as in the apStar files. (A first version stored the
flux as float16 too: at the per-pixel S/N of up to ~2000 of the brightest training stars that rounds the flux by up
to ~1 sigma; Chapter 0 tests this.)
"""
import hashlib
import io
import re
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sqlutilpy as sqlutil
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_build"
MANIFEST = ROOT / "data" / "MANIFEST.json"
SAS = "https://data.sdss.org/sas/dr17/apogee/spectro"
NPIX = 8575                                    # apStar grid: log10(lambda/A) = 4.179 + 6e-6 * i, vacuum

# allStar columns kept in `labels` (calibrated labels, their _spec versions, errors, flags, S/N, Gaia).
ELEM = ["c", "n", "o", "mg", "al", "si", "ca", "mn", "ni"]
COLS = ("apogee_id, telescope, field, ra, dec, glon, glat, j, h, k, ak_targ, snr, nvisits, vhelio_avg, vscatter, "
        "starflag, aspcapflag, extratarg, memberflag, member, meanfib, "
        "teff, teff_err, logg, logg_err, m_h, m_h_err, alpha_m, alpha_m_err, fe_h, fe_h_err, vmicro, vmacro, vsini, "
        "teff_spec, logg_spec, aspcap_chi2, "
        + ", ".join(f"{e}_fe, {e}_fe_err, {e}_fe_flag" for e in ELEM)
        + ", gaiaedr3_source_id, gaiaedr3_parallax, gaiaedr3_parallax_error, gaiaedr3_phot_g_mean_mag, "
          "gaiaedr3_phot_bp_mean_mag, gaiaedr3_phot_rp_mean_mag")

# Common quality cuts.
# - telescope = 'apo25m': one instrument, hence one line-spread function for training and test stars.
# - aspcapflag bit 23 = STAR_BAD (ASPCAP fit unreliable).
# - starflag: reject a named subset of APOGEE_STARFLAG bits (bit: name) - 0 BAD_PIXELS, 3 VERY_BRIGHT_NEIGHBOR,
#   4 LOW_SNR, 9 PERSIST_HIGH, 16 SUSPECT_RV_COMBINATION, 17 SUSPECT_BROAD_LINES, 21 MULTIPLE_SUSPECT, 22 RV_FAIL,
#   23 SUSPECT_ROTATION; sum of 2^bit = 14877209. Low and medium persistence (bits 10, 11, set for ~20% of
#   bright stars) are kept, as in Casey et al. 2016.
# - vscatter < 1 km/s rejects most spectroscopic binaries.
# - extratarg = 0: main-survey row of each star (no duplicates, no commissioning or telluric-standard rows).
# - memberflag = 0 outside `clusters`: cluster members are kept for the cluster test, never trained on.
STARBAD = 14877209
CLEAN = (f"telescope = 'apo25m' AND (aspcapflag & (1::bigint << 23)) = 0 AND (starflag & {STARBAD}) = 0 "
         "AND vscatter < 1 AND teff > 0 AND logg > -1 AND fe_h > -9 AND fe_h <> 'NaN'::real")
FIELD = f"{CLEAN} AND extratarg = 0 AND memberflag = 0"
GIANT = "teff BETWEEN 3800 AND 5300 AND logg BETWEEN 0.8 AND 3.6"
ABUND = " AND ".join(f"{e}_fe <> 'NaN'::real AND {e}_fe_flag = 0" for e in ["c", "n", "mg"])
# A deterministic pseudo-random order (the same rows on every rebuild).
RAND = "ORDER BY md5(apogee_id || field)"

SAMPLES = {
    # Stratified in label space: at most 4 stars per cell of 100 K x 0.25 dex (log g) x 0.25 dex ([Fe/H]), so that
    # rare labels (metal-poor, cool, luminous) are covered; S/N > 200, or > 100 at [Fe/H] < -1 where bright
    # stars are rare.
    "train": f"""WITH g AS (SELECT {COLS}, row_number() OVER (PARTITION BY floor(teff / 100), floor(logg / 0.25),
                                   floor(fe_h / 0.25) {RAND}) AS rk
                            FROM apogee_dr17.allstar WHERE {FIELD} AND {GIANT} AND fe_h BETWEEN -1.5 AND 0.4
                            AND (snr > 200 OR (fe_h < -1 AND snr > 100)) AND {ABUND})
                 SELECT * FROM g WHERE rk <= 4""",
    "test": f"""SELECT {COLS} FROM apogee_dr17.allstar WHERE {FIELD} AND {GIANT}
                AND fe_h BETWEEN -1.5 AND 0.4 AND snr BETWEEN 50 AND 100 {RAND} LIMIT 1500""",
    "metalpoor": f"""SELECT {COLS} FROM apogee_dr17.allstar WHERE {FIELD} AND {GIANT}
                     AND fe_h BETWEEN -2.5 AND -1.5 AND snr > 100 {RAND} LIMIT 300""",
    "dwarfs": f"""SELECT {COLS} FROM apogee_dr17.allstar WHERE {FIELD}
                  AND teff BETWEEN 4000 AND 6000 AND logg > 4.0 AND snr > 100 {RAND} LIMIT 200""",
    # S/N 100-200 at [Fe/H] > -1, so never in `train`; 4-12 visits: the single visits are real low-S/N spectra
    "visits": f"""SELECT {COLS} FROM apogee_dr17.allstar WHERE {FIELD} AND {GIANT}
                  AND fe_h BETWEEN -1.0 AND 0.4 AND snr BETWEEN 100 AND 200 AND nvisits BETWEEN 4 AND 12
                  {RAND} LIMIT 200""",
    # cluster members flagged by APOGEE (`member`); duplicate rows allowed, the highest-S/N row of each star kept
    "clusters": f"""SELECT DISTINCT ON (apogee_id) {COLS} FROM apogee_dr17.allstar WHERE {CLEAN} AND {GIANT}
                    AND member IN ('M67','N6819','N188','N7789','N2420','N6791','N2158',
                                   'M3','M5','M13','M71','M107','M92','M15','M53','M2','M12')
                    AND snr > 70 ORDER BY apogee_id, snr DESC""",
}


def get(sql):
    """Run a query on WSDB and return a DataFrame (strings as str, not bytes)."""
    df = pd.DataFrame(sqlutil.get(sql, asDict=True, db="wsdb"))
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def fetch_apstar(row, visits=False, session=requests.Session(), tries=4):
    """Download one apStar file and return (flux, err, mask, visit_snr) rows; the file is not kept."""
    url = f"{SAS}/redux/dr17/stars/{row.telescope}/{row.field}/apStar-dr17-{row.apogee_id}.fits"
    for k in range(tries):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    with fits.open(io.BytesIO(r.content)) as h:
        nv = h[0].header["NVISITS"]
        rows = [0] if not visits else ([0] + list(range(2, 2 + nv)) if nv > 1 else [0])
        flux, err, mask = (np.atleast_2d(h[i].data)[rows] for i in (1, 2, 3))
        snr = np.array([h[0].header["SNR"]] + [h[0].header.get(f"SNRVIS{v + 1}", np.nan) for v in range(len(rows) - 1)])
    return flux, err, mask.astype(np.int32), snr


def pack(flux, err):
    """Divide by the median flux of each spectrum; flux as float32, error as float16 (see module docstring)."""
    med = np.nanmedian(np.where(flux > 0, flux, np.nan), axis=1).astype(np.float32)
    with np.errstate(over="ignore"):                        # errors > 65504 x median become inf: no information
        return (flux / med[:, None]).astype(np.float32), (err / med[:, None]).astype(np.float16), med


def build_spectra(name, workers=12):
    lab = get(SAMPLES[name])
    visits = name == "visits"
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda r: fetch_apstar(r, visits), lab.itertuples()))
    star = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    flux, err, med = pack(np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res]))
    out = dict(apogee_id=lab.apogee_id.to_numpy().astype("U18"), star=star, flux=flux, err=err,
               mask=np.vstack([r[2] for r in res]), median=med, snr=np.concatenate([r[3] for r in res]).astype(np.float32),
               visit=np.concatenate([np.arange(len(r[0])) for r in res]))   # visit 0 = combined spectrum
    path = OUT / f"spectra_{name}.npz"
    np.savez_compressed(path, **out)
    lab.insert(0, "sample", name)
    return path, lab, [SAMPLES[name]]


def build_rvs(n=4000):
    """Gaia DR3 RVS mean spectra (846-870 nm, 2401 pixels of 0.01 nm) of APOGEE giants with clean labels."""
    sql = f"""WITH a AS MATERIALIZED (SELECT {COLS} FROM apogee_dr17.allstar WHERE {FIELD} AND {GIANT}
                 AND fe_h BETWEEN -2.0 AND 0.4 AND snr > 100 AND gaiaedr3_source_id > 0),
              r AS MATERIALIZED (SELECT a.*, r.flux AS rvs_flux, r.flux_error AS rvs_flux_error, r.combined_transits
                 FROM a JOIN gaia_dr3.rvs_mean_spectrum r ON r.source_id = a.gaiaedr3_source_id {RAND} LIMIT {n})
              SELECT r.*, g.rvs_spec_sig_to_noise, p.teff_gspspec, p.logg_gspspec, p.mh_gspspec, p.alphafe_gspspec,
                     p.flags_gspspec
              FROM r JOIN gaia_dr3.gaia_source g ON g.source_id = r.gaiaedr3_source_id
              LEFT JOIN gaia_dr3.astrophysical_parameters p ON p.source_id = r.gaiaedr3_source_id"""
    lab = get(sql)
    flux = np.vstack(lab.pop("rvs_flux")).astype(np.float32)
    err = np.vstack(lab.pop("rvs_flux_error")).astype(np.float32)
    path = OUT / "rvs_apogee.npz"
    np.savez_compressed(path, apogee_id=lab.apogee_id.to_numpy().astype("U18"), flux=flux, err=err,
                        wavelength_nm=np.round(846 + 0.01 * np.arange(flux.shape[1]), 2))
    lab.insert(0, "sample", "rvs")
    return path, lab, [sql]


# Solar photospheric abundances log eps (Asplund et al. 2009, ARA&A 47, 481, Table 1).
SOLAR = dict(C=8.43, N=7.83, O=8.69, Na=6.24, Mg=7.60, Al=6.45, Si=7.51, P=5.41, S=7.12, K=5.03, Ca=6.34, Ti=4.95,
             V=3.93, Cr=5.64, Mn=5.43, Fe=7.50, Co=4.99, Ni=6.22, Cu=4.19, Ce=1.58, Nd=1.42)


def read_turbospec(url, molecules=None):
    """Parse a Turbospectrum line list: species header lines in quotes, then wavelength [A, air], chi [eV], log gf."""
    rows, species = [], None
    for line in requests.get(url, timeout=300).text.splitlines():
        if line.startswith("'"):
            tag = line.split("'")[1].strip()
            if molecules is None and len(tag) <= 6 and tag[:1].isalpha():     # atomic species, e.g. 'FE I'
                species = tag
            elif molecules is not None and re.fullmatch(r"\d{4}\.\d{6}", tag):  # isotopologue code, e.g. 0607.012014
                species = molecules.get(tag)                               # None: an isotopologue we skip
        elif line.startswith(" ") and species:
            w, chi, loggf = map(float, line.split()[:3])
            rows.append((species, w, chi, loggf))
    return pd.DataFrame(rows, columns=["species", "wave_air", "chi_ev", "loggf"])


def build_linelist():
    """Lines in the APOGEE window from the DR17 Turbospectrum lists (Smith et al. 2021), with vacuum wavelengths.
    Strength proxy for a 4500 K giant: log eps_sun(X) + log gf - theta * chi, theta = 5040 / T [eV^-1]
    (abundance x oscillator strength x Boltzmann factor; ionisation and molecular equilibrium are ignored, so
    compare lines of one species only; for molecules log eps is omitted). The 40 strongest lines of each atomic
    species (neutral and singly ionised) and the 60 strongest of 12C14N, 12C16O and OH are kept."""
    base = f"{SAS}/speclib/linelists/turbospec/turbospec.20180901t20"
    at = read_turbospec(base + ".atoms")
    at["element"] = at.species.str.split().str[0].str.capitalize()
    at = at[at.element.isin(SOLAR) & at.species.str.split().str[1].isin(["I", "II"])].copy()
    at["strength"] = at.element.map(SOLAR) + at.loggf - 5040 / 4500 * at.chi_ev
    mol = read_turbospec(base + ".molec", {"0607.012014": "CN", "0608.012016": "CO", "0108.001016": "OH"})
    mol["element"] = mol.species
    mol["strength"] = mol.loggf - 5040 / 4500 * mol.chi_ev
    df = pd.concat([at, mol])
    s2 = (1e4 / df.wave_air) ** 2                       # vacuum = air x n(air), n from Morton (2000), eq. 8
    df["wave_vac"] = df.wave_air * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
    df = df[(df.wave_vac > 15140) & (df.wave_vac < 16960)]
    df = df.sort_values("strength", ascending=False)
    rank = df.groupby("species").cumcount().to_numpy()                 # 0 for the strongest line of a species
    df = df[rank < np.where(df.species.isin(["CN", "CO", "OH"]), 60, 40)]
    path = OUT / "linelist.parquet"
    df.sort_values("wave_vac").reset_index(drop=True).to_parquet(path, compression="zstd")
    return path, df, [base + ".atoms and .molec (strength = log eps + loggf - 1.12 chi; 40 strongest per atomic "
                      "species, 60 per molecule; vacuum wavelengths by Morton 2000)"]


def build_continuum():
    """The continuum pixels of Chapter 3, computed by running that notebook's own code cells up to the cell that
    loads the shipped list (so the definition lives in one place). Needs spectra_train, labels and linelist in
    notebooks/data (run the other targets first and copy data_build/* there)."""
    import os
    nb = json.loads((ROOT / "notebooks" / "03_continuum.ipynb").read_text())
    env, cwd = {}, os.getcwd()
    os.chdir(ROOT / "notebooks")
    try:
        import matplotlib
        matplotlib.use("Agg")
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            if "load('continuum_pixels')" in src:
                break
            exec(src, env)
    finally:
        os.chdir(cwd)
    df = pd.DataFrame({"pixel": np.arange(NPIX), "wave_vac": env["wave"], "continuum": env["cont"]})
    path = OUT / "continuum_pixels.parquet"
    df.to_parquet(path, compression="zstd")
    return path, df, ["notebooks/03_continuum.ipynb, sect. 3.3 (N15 criteria on a model trained on "
                      "quantile-normalised spectra_train)"]


def save_manifest(name, path, nrows, queries):
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man[name] = {"rows": int(nrows), "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "queries": [" ".join(q.split()) for q in queries]}
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(sorted(man.items())), indent=1))
    print(f"{name}: {nrows} rows, {path.stat().st_size / 1e6:.1f} MB")


def main(names):
    OUT.mkdir(exist_ok=True)
    labels = []
    for name in names:
        t = time.time()
        if name in SAMPLES:
            path, lab, q = build_spectra(name)
            labels.append((lab, q))
        elif name == "rvs":
            path, lab, q = build_rvs()
            name = "rvs_apogee"
        elif name == "linelist":
            path, lab, q = build_linelist()
        elif name == "continuum":
            path, lab, q = build_continuum()
            name = "continuum_pixels"
        else:
            raise SystemExit(f"unknown file {name}")
        save_manifest(name if name in ("rvs_apogee", "linelist", "continuum_pixels") else f"spectra_{name}", path, len(lab), q)
        if name == "rvs_apogee":
            lab.to_parquet(OUT / "labels_rvs.parquet", compression="zstd")
            save_manifest("labels_rvs", OUT / "labels_rvs.parquet", len(lab), q)
        print(f"  {time.time() - t:.0f} s")
    if labels:                       # labels of all APOGEE spectra samples built in this run, one table
        path = OUT / "labels.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        new = pd.concat([lab for lab, _ in labels])
        if old is not None:
            new = pd.concat([old[~old["sample"].isin(new["sample"].unique())], new])
        new.reset_index(drop=True).to_parquet(path, compression="zstd")
        save_manifest("labels", path, len(new), [q for s in SAMPLES for q in [SAMPLES[s]]])


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    main(list(SAMPLES) + ["rvs", "linelist"] if args == ["all"] else args)   # then: python scripts/build_data.py continuum
