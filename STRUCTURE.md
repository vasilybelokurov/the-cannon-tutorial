# STRUCTURE.md

Plan of the tutorial **Stellar labels from spectra with The Cannon**. Audience: students (beginner upwards)
who know basic Python/numpy and what a stellar spectrum is. Exit level: writing, training, validating and
stress-testing a data-driven spectral model, The Cannon (Ness et al. 2015), on real APOGEE spectra, and
transferring its labels to another survey.

This plan was reviewed independently by Claude Fable and by Codex (OpenAI) on 2026-10-30. Their main
recommendations are adopted: cluster members are never trained on; the training set is stratified in label
space; flagged pixels get inflated errors instead of being dropped; the toy model comes before continuum
normalisation; the "same LSF" claim is replaced by a fibre diagnostic; repeat-visit precision is measured from
visit-to-visit scatter (a visit and the combination containing it are correlated); asserts test behaviour with
tolerances frozen after a development run, not hoped-for outcomes; the abundance chapter tests whether a label
is *measured* or *predicted from correlations* by censoring its lines; the Gaia chapter is an adaptation of the
label-transfer idea of Ho et al. 2017 (who used LAMOST), compared with Gaia's own GSP-Spec labels.

## Principles

1. **From the pixels up.** Students start from the APOGEE `apStar` spectra (reduced, resampled, rest-frame
   flux with errors and a pixel bitmask), not from pipeline-normalised spectra, and write The Cannon
   themselves in numpy. No `thecannon` package as a black box.
2. **Toy before real.** The model is first fitted to synthetic spectra drawn from a known quadratic model, where
   the right answer is known.
3. **Follow the papers.** Each chapter rebuilds a figure or test of Ness et al. 2015 (N15), Casey et al. 2016
   (C16), Ness et al. 2016 (N16), Ho et al. 2017 (H17) or Ness et al. 2018 (N18) with DR17 data; differences are
   stated with their size and cause.
4. **Label scale is inherited.** The reference labels are the DR17 ASPCAP calibrated `TEFF`, `LOGG`, `FE_H`
   (and `MG_FE`, `C_FE`, `N_FE`). Agreement with ASPCAP measures *precision on the ASPCAP scale*, not accuracy.
   Independent checks are repeat visits, star clusters, and another survey (Gaia RVS).
5. **Validate honestly.** k-fold cross-validation, real single-visit spectra, clusters, extrapolation outside
   the training set, and a censoring test, not only a one-to-one plot of the training set.
6. **Public data as pre-extracted snippets**, as in the sibling tutorials: `scripts/build_data.py` queries WSDB
   (`apogee_dr17.allstar`, `gaia_dr3.rvs_mean_spectrum`, `gaia_dr3.astrophysical_parameters`) and the SDSS SAS
   (`apStar` files, line lists) and writes small files plus `data/MANIFEST.json` (query, rows, SHA-256);
   notebooks download them from Zenodo (https://doi.org/10.5281/zenodo.23813051, made with
   `scripts/zenodo_upload.py`) and check the checksum.
7. **Plot + assertion.** Every claim comes with a figure and an `assert` with a tolerance. Tolerances are
   starting budgets, frozen after a development run.
8. **Least code that works.** Each notebook is self-contained (runs alone in Colab); the small helpers (loader,
   normalisation, train, test) are repeated rather than hidden in a package.

## Chapters (v0.1) — built 2026-10-30

Asserts as implemented; measured values in brackets. Tolerances were frozen after a development run; where a
first guess failed, the test was changed to what the data showed and the text says so.

| Ch | Notebook | Learning objective | Data | Assert (tolerance) [measured] |
|---|---|---|---|---|
| 00 | `00_apogee_spectra` | APOGEE `apStar` spectra: log-linear vacuum grid, three detectors, flux, error, `APOGEE_PIXMASK`; S/N; flagged pixels get zero weight; storage precision; samples and Kiel diagrams | `labels`, `spectra_train` | grid 15 100.80–16 999.8 Å, 3 detectors; catalogue SNR = median flux/err (< 0.1%) [exact]; flux stored as float32 (rounding < 1e-3 σ) and float16 would not be safe (99.9th pct > 0.5σ) [0.75σ] |
| 01 | `01_spectra_and_labels` | Line list (atoms, CN, CO, OH; air → vacuum); spectra ordered by one label; flux of one pixel vs labels, linear vs quadratic; coefficients pixel by pixel | `spectra_train`, `linelist` | vacuum − air 4.0–4.6 Å [4.37]; Fe I core ∂f/∂[Fe/H] < 0 at > 5σ [−0.159 ± 0.005]; quadratic rms < linear [0.015 vs 0.021] |
| 02 | `02_cannon_toy` | The Cannon on synthetic spectra: scaling, training step (WLS + ML scatter, why ln V matters), test step, formal errors, pulls, regression dilution | synthetic | model pulls std 1 ± 0.1, ≥ 99% within 3σ [1.005]; s recovered ±10% where s = σ, < 0.3σ where s = 0; label pulls 1 ± 0.1 [0.99–1.03]; dilution slope = 1 + (σ_noise/σ_label)² within 0.03 |
| 03 | `03_continuum` | Running-quantile pseudo-continuum and its S/N bias; continuum pixels from a first model (N15 §5.3); Chebyshev per detector; S/N test | `spectra_train`, `linelist`, `continuum_pixels` | continuum pixels avoid lines (< ½ the line proximity of average pixels) [0.08 vs 0.21]; shipped list identical; quantile bias at S/N 30 > 3× continuum-pixel bias [2.3% vs < 1e-4] |
| 04 | `04_cannon_apogee` | Training on 1363 giants; cleaning the training set (χ²/ν ≥ 3 removed); coefficients vs line list (N15 Fig. 5); held-out χ²; flagged pixels; fibre (LSF) residuals | `spectra_train`, `linelist` | Fe I cores ∂f/∂[Fe/H] < 0 for ≥ 90% [100%]; held-out median χ²/ν in 0.7–1.1 [0.82] and noise-only > 1.5× [8.0]; sky-line outliers > 3× clean [5×]; fibre–residual Spearman ρ > 0.2, p < 1e-6 [0.41] |
| 05 | `05_validation` | 10-fold CV with bias, std, RMSE, robust std, tails; formal vs empirical errors; precision vs [Fe/H]; linear vs quadratic (paired bootstrap); literature comparison (N15, N18, C16, Leung & Bovy 19); continuum-selection leak measured; extrapolation (χ², N15's EFLAG) | `spectra_train`, `spectra_metalpoor`, `spectra_dwarfs` | all fits converge; CV std below N15 Fig. 4 (96 K, 0.24, 0.08) [72 K, 0.19, 0.061; robust 51 K, 0.115, 0.034]; formal/std < 0.5 [0.05–0.07]; quadratic better (95% CI > 0) for T_eff, log g; fold-local continuum pixels change the scatter < 2%; metal-poor bias > 3σ [+0.60 dex below −2]; dwarf log g off > 0.3 dex; EFLAG dwarfs > 5× giants [0.60 vs 0.05]; flags miss many [either: 72% dwarfs, 36% metal-poor] |
| 06 | `06_repeat_visits` | Single-visit precision from matched-S/N pairs of visits (bootstrap over stars); formal errors; bias of single visits vs combined; quantile vs continuum pixels | `spectra_visits` | pair precision > formal error in every S/N bin, beyond the bootstrap range at the highest S/N [≈ 20 K, 0.03, 0.014 dex at S/N 120; 2–4× formal]; precision improves with S/N; quantile [Fe/H] bias at S/N < 60 larger than continuum-pixel bias [+0.058 vs +0.002] |
| 07 | `07_clusters` | 17 open and globular clusters (never trained on): Kiel diagrams, cluster means across clusters, spreads cluster by cluster, T_eff trends centred within clusters | `spectra_clusters` | slope of cluster means 0.95–1.05 inside the training range [0.98]; open-cluster median std < 0.05 and < 1.5× ASPCAP [0.036 vs 0.034; Cannon wider in 5 of 7]; M92/M15 tight but +0.7 dex off (shown, not asserted) |
| 08 | `08_abundances` | Six labels; literature comparison (C16, N18, Leung & Bovy 19); derivatives at Mg I, CO, CN lines; α sequences at S/N 50–100; measured or predicted: Mg lines hidden, random-pixel control, prediction from the other ASPCAP labels and from Cannon labels without Mg lines | `spectra_train`, `spectra_test`, `linelist` | rms/spread < 0.6, 0.7, 0.9 for Mg, C, N [0.49, 0.57, 0.80]; six labels improve T_eff, log g over three [53 vs 72 K]; own-line derivative > 2× average; same α sequence as ASPCAP > 80% [88%]; Mg lines hidden > +10% [+16%], random < +3% [0%]; predictions from the other labels within 10% of the six-label scatter [0.063, 0.067 vs 0.063] |
| 09 | `09_gaia_rvs` | Label transfer APOGEE → Gaia DR3 RVS; Gaia normalisation unstable at low S/N; renormalisation on continuum pixels chosen inside each fold; comparison with GSP-Spec (best-quality flags, RB23 calibration, robust std) and with Das+25 | `rvs_apogee`, `labels_rvs` | continuum-offset scatter at S/N 20–30 > 5× that at S/N > 70 and ρ(offset, Δ[Fe/H]) < −0.5 [−0.66]; renormalisation improves log g, [Fe/H] at S/N < 30 by > 30%, removes > 70% of the [Fe/H] outliers; on best-quality stars Cannon robust std < calibrated GSP-Spec for all three labels [48 vs 70 K, 0.13 vs 0.17, 0.066 vs 0.081; GSP-Spec better in log g, [Fe/H] at S/N > 100] |
| 10 | `10_many_labels` | Twelve labels (T_eff, log g, [Fe/H], 9 [X/Fe]); more labels vs more flexibility (cubic); L1 regularisation (C16) by ADMM, Λ scan and training-set size; censoring with line-list windows (Holtzman+18); several starting points; convex-hull test; code in `src/more.py` | `spectra_train`, `spectra_test`, `linelist`, `labels` | cubic and 12 labels each improve validation T_eff > 20% [50 → 33, 34 K]; some Λ > 0 lowers held-out χ² [1.36 → 1.31 at Λ = 300], Λ = 10⁴ raises it and worsens labels; at N = 120 regularisation improves log g > 20% and [X/Fe] [0.22 → 0.14; 0.053 → 0.038], at full N it does not [0.046 → 0.050]; censoring improves Mg, Al, Ca, Ni at S/N 50–100 by > 5% [17–21%], degrades O and T_eff > 10%; multi-start never worse, lowers χ² for some [12 of 741; labels closer to ASPCAP for 6]; hull: 3-D test > 95% inside, metal-poor < 20%, 12-D < 50% [99%, 0%, 18%] |
| 11 | `11_lux` | Lux (Horta+25): labels and spectra linear in a latent vector; training by weighted least squares (`src/lux.py`); P and the null space of under-determined steps; vs 12-label Cannon and ridge regression; Mg lines hidden; missing [N/Fe]; noisy training labels | `spectra_train`, `spectra_test`, `labels`, `linelist` | minimum-norm P = 49 ≡ P = 13 (rank 13) [exact]; Lux abundances better than Cannon on validation [0.035 vs 0.046], T_eff at S/N 50–100 > 1.3× worse [67 vs 34 K]; ridge abundances better than Cannon [0.039]; hiding Mg I lines costs > 10% for all three [20–43%] but Lux without them beats Cannon with them [0.033 vs 0.040]; missing [N/Fe] for half: Lux all ≈ Lux half (< 10%) [0.063 both]; noisy labels (5σ): Cannon abundances > 1.5× worse [2.0×, 2.1×], Lux < 1.15× [1.08×, 0.94×], ridge worse than Lux |
| A | `A_annieslasso` | Our Cannon against AnniesLasso 0.2.96 (Casey+16 code) on the same 1703 training / 1500 test stars: first run the same way (bug check), then each as designed (method difference); four workarounds needed (numpy 2, scipy 1.18) | `spectra_train`, `spectra_test` | same model: spectral models agree < 1e-3 σ [2e-6 σ]; labels 99th pct < 0.05 formal σ [0.021]; where they differ AnniesLasso has the higher χ² [2 stars, Δχ² 0 and 17.8]; as designed: AL scatter ≤ ours at every pixel [median ratio 0.44]; codes differ < 0.3× scatter about ASPCAP [10 K, 0.032, 0.008 vs 51 K, 0.121, 0.033]; ours closer to ASPCAP in all labels [AL 56 K, 0.126, 0.036]; θ weights matter more than s |
| B | `B_pollux` | Lux and The Cannon with pollux 0.1.0 (pinned versions): data containers, training, test step; exact check of our least-squares blocks; label precision of both implementations | as Ch 11 | block solvers agree < 1e-6 of errors [~1e-9]; Cannon precisions within 8%, Lux within 20% |

Findings that changed the plan during the build: float16 flux storage was unsafe (Ch 0 test failed; files rebuilt
with float32 flux); one scatter per pixel is inflated by ~3% outlier training stars (Ch 4 cleaning step); a six-label
[Mg/Fe] is at most 5–10% better than a prediction from the other labels (Ch 8); Gaia's RVS normalisation is S/N
dependent and produces a spurious metal-poor branch among faint stars (Ch 9); χ² alone flags only a minority of
extrapolated stars (Ch 5).

Second review (Claude Fable and Codex, 2026-10-30, on the executed tutorial, focused on performance against the
literature). Adopted: robust statistics next to every scatter and a literature table in Ch 5 and 8 (we are ~25%
better than N15 but 2–3× worse than N18, C16 and astroNN); a fair GSP-Spec comparison (quality flags, calibration,
robust std) in Ch 9, which shrank the Cannon's advantage from 2× to 1.2–1.5×; matched-pair repeat precision in Ch 6
(the earlier "noise floor" was an artefact of a biased estimator); clean correlation baselines for [Mg/Fe] in Ch 8;
continuum pixels chosen inside each fold in Ch 9 and the APOGEE leak measured (< 2%) in Ch 5; cluster spreads
reported cluster by cluster in Ch 7; optimiser convergence checked; atomic downloads; a reference list at the end of
every chapter.

## Later chapters (v0.2)

| Ch | Title | Data | Check |
|---|---|---|---|
| 12 | Masses and ages of red giants (N16: 0.07 dex in mass, 0.21 dex in age from 1639 APOKASC stars; CN features) | APOKASC-3 (Pinsonneault+25) × DR17 | CV rms log M |
| 13 | Chemical doppelgängers (N18: ~0.3% of field pairs, ~1% at [Fe/H] = 0 ± 0.02, are indistinguishable) | DR17 clusters, field | pair χ² distribution of cluster vs field pairs |
| 14 | Bootstrap model uncertainty and coverage (N15 Fig. 11) | DR17 | bootstrap vs formal errors in sparse regions |

## Data (v0.1)

Spectra: DR17 `apStar` files, APO 2.5 m only (one instrument; the line-spread function still varies from fibre to
fibre, and `meanfib` is kept for that diagnostic). Row 0 of HDUs 1–3 (combined flux, error, `APOGEE_PIXMASK`);
for `spectra_visits` also the single-visit rows. Stored as `.npz`: flux and error divided by the median flux of
each spectrum (flux float32, error float16; Chapter 0 shows why the flux cannot be float16), the bitmask as
int32, the median as float32. Pixels without data are NaN.

Quality cuts (all samples): `telescope = apo25m`; ASPCAP `STAR_BAD` not set; `STARFLAG` without BAD_PIXELS,
VERY_BRIGHT_NEIGHBOR, LOW_SNR, PERSIST_HIGH, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES, MULTIPLE_SUSPECT,
RV_FAIL, SUSPECT_ROTATION; `vscatter < 1 km/s`. Field samples: `extratarg = 0` (one row per star) and
`memberflag = 0`. Giants: 3800 < T_eff < 5300 K, 0.8 < log g < 3.6.

| File | Content |
|---|---|
| `labels` | allStar rows (labels, errors, flags, SNR, `meanfib`, `member`, Gaia IDs) of every star in the spectra files, with its `sample` |
| `spectra_train` | 1703 giants, −1.5 < [Fe/H] < 0.4, SNR > 200 (> 100 at [Fe/H] < −1), [C/Fe], [N/Fe], [Mg/Fe] unflagged; at most 4 stars per (100 K, 0.25 dex, 0.25 dex) cell |
| `spectra_test` | 1500 giants, same label range, 50 < SNR < 100 |
| `spectra_metalpoor` | 300 giants, −2.5 < [Fe/H] < −1.5, SNR > 100 |
| `spectra_dwarfs` | 200 dwarfs, 4000 < T_eff < 6000 K, log g > 4, SNR > 100 |
| `spectra_visits` | 200 giants (1285 spectra), −1 < [Fe/H] < 0.4, 100 < SNR < 200, 4–12 visits: combined and single-visit spectra, per-visit S/N |
| `spectra_clusters` | 735 giant members (`member`) of 17 open and globular clusters, SNR > 70, best row per star |
| `rvs_apogee`, `labels_rvs` | Gaia DR3 RVS mean spectra (flux, error) of 4000 APOGEE giants, their labels and Gaia GSP-Spec parameters |
| `continuum_pixels` | the 704 continuum pixels of Chapter 3 (made by running that chapter's cells) |
| `linelist` | strongest lines in the APOGEE window from the DR17 Turbospectrum lists (Smith et al. 2021): 40 per atomic species (I, II), 60 each of ¹²C¹⁴N, ¹²C¹⁶O, OH; vacuum wavelengths |

## Repository layout

```
src/            notebook sources (percent format) and the shared blocks loader.py, norm.py, cannon.py, more.py,
                pasted into notebooks at lines #@@LOADER etc.
notebooks/      NN_<slug>.ipynb generated from src/ by scripts/py2nb.py, committed without outputs
scripts/        build_data.py (WSDB + SDSS SAS → data files + manifest), zenodo_upload.py (instructor only),
                py2nb.py (src → notebook)
data/           README.md (licences, columns), MANIFEST.json, zenodo.json; data files are on Zenodo
tests/          pytest tests
_config.yml, _toc.yml, intro.md   Jupyter Book
_static/custom.css                page layout as in the sibling tutorials
.github/workflows/book.yml        tests, executes notebooks, publishes to GitHub Pages
```

## Testing

- `tests/test_cannon.py`: unit tests of the shared Cannon code on synthetic data (orders 1–3, L1 optimality
  conditions, censoring, multi-start, hull). `tests/test_sources.py`: every notebook equals its source in src/.
- `pytest tests/ --nbmake notebooks/` executes every notebook in a fresh kernel; notebooks contain explicit
  `assert` cells. `jupyter-book build . -W -n --keep-going` builds the site. Both run in CI; fixed random seeds.

## Literature (checked in ADS)

The Cannon and its applications: Ness+15 https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N ·
Casey+16 https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C · Ness+16 https://ui.adsabs.harvard.edu/abs/2016ApJ...823..114N ·
Ho+17a (label transfer to LAMOST) https://ui.adsabs.harvard.edu/abs/2017ApJ...836....5H ·
Ho+17b (LAMOST masses) https://ui.adsabs.harvard.edu/abs/2017ApJ...841...40H ·
Hogg+16 (chemical tagging) https://ui.adsabs.harvard.edu/abs/2016ApJ...833..262H ·
Ness+18 (doppelgängers) https://ui.adsabs.harvard.edu/abs/2018ApJ...853..198N ·
Ness+19 https://ui.adsabs.harvard.edu/abs/2019ApJ...883..177N · Wheeler+20 https://ui.adsabs.harvard.edu/abs/2020ApJ...898...58W ·
Ness+22 https://ui.adsabs.harvard.edu/abs/2022ApJ...926..144N · Rix+16 (polynomial spectral models) https://ui.adsabs.harvard.edu/abs/2016ApJ...826L..25R ·
Martig+16 ([C/N] masses) https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3655M ·
Das+25 (The Cannon on Gaia RVS) https://ui.adsabs.harvard.edu/abs/2025MNRAS.538..605D

Other data-driven and model-driven methods: Ting+19 (The Payne) https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T ·
Leung & Bovy 19 (astroNN) https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L

APOGEE: Majewski+17 https://ui.adsabs.harvard.edu/abs/2017AJ....154...94M ·
Nidever+15 (reduction) https://ui.adsabs.harvard.edu/abs/2015AJ....150..173N ·
García Pérez+16 (ASPCAP) https://ui.adsabs.harvard.edu/abs/2016AJ....151..144G ·
Jönsson+20 (DR16) https://ui.adsabs.harvard.edu/abs/2020AJ....160..120J ·
Abdurro'uf+22 (DR17) https://ui.adsabs.harvard.edu/abs/2022ApJS..259...35A ·
Zasowski+13 (targeting) https://ui.adsabs.harvard.edu/abs/2013AJ....146...81Z ·
Pinsonneault+18 (APOKASC-2) https://ui.adsabs.harvard.edu/abs/2018ApJS..239...32P ·
Pinsonneault+25 (APOKASC-3) https://ui.adsabs.harvard.edu/abs/2025ApJS..276...69P

## Open items

- APOKASC-3 (VizieR J/ApJS/276/69) was not reachable on 2026-10-30 (CDS down); Chapter 12 waits for it.
- Smith et al. 2021 (DR16/17 line list) https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S · Asplund et al. 2009 https://ui.adsabs.harvard.edu/abs/2009ARA&A..47..481A · Morton 2000 https://ui.adsabs.harvard.edu/abs/2000ApJS..130..403M · Recio-Blanco et al. 2023 (GSP-Spec) https://ui.adsabs.harvard.edu/abs/2023A&A...674A..29R: checked in ADS.
- The second, positive-offset group of faint-star RVS spectra (Ch 9): origin in the Gaia processing not traced.
- Ch 10: validation-star abundances are better from 300 training stars than from 1308 (both subsets; not seen on the
  S/N 50–100 test stars). Not traced. Tests to run (partly from the Codex review of 2026-10-31): swap theta and s
  between the 300- and 1308-star models; apply ASPCAP quality flags for all nine elements (build_data cuts only C, N,
  Mg); remove influential training stars (`train_clean`); repeat with nested subsets, several validation splits and
  per-element bias/MAD/RMSE with intervals; check multi-start effects.
