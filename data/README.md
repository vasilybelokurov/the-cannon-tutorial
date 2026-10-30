# Data for The Cannon tutorial

Extracts of APOGEE DR17 and Gaia DR3, prepared for the notebooks of
https://github.com/vasilybelokurov/the-cannon-tutorial and published on Zenodo,
https://doi.org/10.5281/zenodo.23813051 (record https://zenodo.org/records/23813051). The notebooks download the
files from there and check each one against the SHA-256 checksum in `MANIFEST.json`. A changed selection is
published as a new file name in a new version of the record, so a checksum in `MANIFEST.json` never changes.

Every file is rebuilt by `scripts/build_data.py`. `MANIFEST.json` lists, for each file, the exact query (SQL on
WSDB, Cambridge, or a URL on the SDSS Science Archive Server), the row count, the size, the SHA-256 checksum and the
creation time.

## Files

| File | Format | Content |
|---|---|---|
| `labels` | parquet | APOGEE DR17 allStar rows of every star in the `spectra_*` files, with a `sample` column; calibrated labels (`teff`, `logg`, `m_h`, `alpha_m`, `fe_h`, `<x>_fe` for C, N, O, Mg, Al, Si, Ca, Mn, Ni), their errors and flags, `teff_spec`, `logg_spec`, `snr`, `nvisits`, `vscatter`, `starflag`, `aspcapflag`, `memberflag`, `member`, `meanfib`, Gaia EDR3 identifiers and photometry |
| `spectra_<sample>` | npz | DR17 `apStar` spectra (APO 2.5 m) of the stars of one sample, in the order of `labels`: `flux`, `err` (float16, divided by `median`), `mask` (`APOGEE_PIXMASK`, int32), `median` (float32, units of 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹), `snr`, `apogee_id`, `star` (row in `labels` of that sample), `visit` (0 = combined spectrum; 1, 2, ... = single visits, only in `spectra_visits`) |
| `continuum_pixels` | parquet | `pixel`, `wave_vac`, `continuum` (bool): the continuum pixels of Chapter 3 |
| `linelist` | parquet | strongest lines in the APOGEE window from the DR17 Turbospectrum line lists (Smith et al. 2021): `species`, `wave_air`, `wave_vac` (Morton 2000), `chi_ev`, `loggf`, `strength` |
| `rvs_apogee` | npz | Gaia DR3 RVS mean spectra of APOGEE giants: `flux`, `err` (float32, pseudo-continuum normalised by the Gaia pipeline), `wavelength_nm` (846–870 nm, 2401 pixels), `apogee_id` |
| `labels_rvs` | parquet | the APOGEE labels of the stars in `rvs_apogee` (same columns as `labels`), `rvs_spec_sig_to_noise`, and Gaia GSP-Spec `teff_gspspec`, `logg_gspspec`, `mh_gspspec`, `alphafe_gspspec`, `flags_gspspec` |

Spectra: the `apStar` wavelength grid is $\log_{10}(\lambda/{\rm Å}) = 4.179 + 6\times10^{-6}\,i$, $i = 0..8574$
(vacuum, rest frame). Pixels without data are NaN. Float16 storage: relative rounding ≤ 2⁻¹¹ (Chapter 0 checks
that this is < 0.25σ for 99.9% of the pixels).

Sample definitions (full SQL in `MANIFEST.json`): all samples are APO 2.5 m, without ASPCAP `STAR_BAD`, without the
STARFLAG bits BAD_PIXELS, VERY_BRIGHT_NEIGHBOR, LOW_SNR, PERSIST_HIGH, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES,
MULTIPLE_SUSPECT, RV_FAIL, SUSPECT_ROTATION, and with `vscatter < 1` km/s. Field samples have `extratarg = 0` and
`memberflag = 0`.

| sample | selection |
|---|---|
| `train` | giants (3800 < T_eff < 5300 K, 0.8 < log g < 3.6), −1.5 < [Fe/H] < 0.4, S/N > 200 (> 100 at [Fe/H] < −1), unflagged [C/Fe], [N/Fe], [Mg/Fe]; at most 4 stars per cell of 100 K × 0.25 dex × 0.25 dex |
| `test` | giants, −1.5 < [Fe/H] < 0.4, 50 < S/N < 100 (1500 stars) |
| `metalpoor` | giants, −2.5 < [Fe/H] < −1.5, S/N > 100 (300) |
| `dwarfs` | 4000 < T_eff < 6000 K, log g > 4, S/N > 100 (200) |
| `visits` | giants, −1 < [Fe/H] < 0.4, 100 < S/N < 200, 4–12 visits (200); combined and single-visit spectra |
| `clusters` | giants flagged by APOGEE (`member`) as members of M67, NGC 6819, 188, 7789, 2420, 6791, 2158, M3, M5, M13, M71, M107, M92, M15, M53, M2, M12, S/N > 70; the highest-S/N row of each star |

## Licences and terms

| Product | Terms | Please cite |
|---|---|---|
| SDSS-IV APOGEE DR17 | public data; acknowledge SDSS | Abdurro'uf et al. 2022, https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A ; Majewski et al. 2017, https://ui.adsabs.harvard.edu/abs/2017AJ....154...94M |
| APOGEE DR17 line list | public | Smith et al. 2021, https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S |
| Gaia DR3 | open, with ESA/Gaia/DPAC credit | Gaia Collaboration 2023, https://ui.adsabs.harvard.edu/abs/2023A&A...674A...1G ; GSP-Spec: Recio-Blanco et al. 2023, https://ui.adsabs.harvard.edu/abs/2023A&A...674A..29R |

The files themselves are distributed under CC BY 4.0. Please cite the Zenodo record (https://doi.org/10.5281/zenodo.23813051) and the original data releases above.

## Acknowledgements

Funding for the Sloan Digital Sky Survey IV has been provided by the Alfred P. Sloan Foundation, the U.S.
Department of Energy Office of Science, and the Participating Institutions (https://www.sdss4.org). This work has
made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by
the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). The
catalogue queries were run on WSDB, the Whole Sky Database in Cambridge (Koposov & Bartunov 2006,
https://ui.adsabs.harvard.edu/abs/2006ASPC..351..735K).
