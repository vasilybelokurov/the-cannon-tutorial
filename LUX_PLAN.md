# Plan: Chapter 11 — Beyond The Cannon: a latent-variable model (Lux)

Status: v2 (2026-10-31), revised after a Codex review of v1. **Implemented 2026-11-01** as Chapter 11
(`src/11_lux.py`, `src/lux.py`, `tests/test_lux.py`) and Appendix B (`src/B_pollux.py`). Changes from this plan:
- E4 uses noise of 5× (not 2×) the ASPCAP errors: 2× is too small next to the label precision to show an effect.
- E2 dropped the τ scan (inert, §3) and P = 48; E5 (PCA, PLS) not done.
- The reference comparison (§5) was done against *pollux* 0.1.0 (A. Price-Whelan), which implements Lux with exact
  block solves, instead of against H25's 2025 scripts: block solvers agree to ~1e-9 of the errors (Appendix B).
- H25's code gained an MIT licence and the stage-2 fix (db9d2ae) on 2026-11-01, after we reported it.
- Not done (open, from the Codex review of 2026-11-01): repeated random draws with paired bootstrap intervals for
  E3/E4; per-label bias/RMSE and [Fe/H]-bin tables; spectral prediction and label-response checks of E1; ridge
  baseline for E3; ridge re-tuned on noisy labels and Lux told underestimated errors in E4; continuum pixels chosen
  within the training split only.

## 1. Goal

Implement the linear Lux model of Horta et al. 2025 (H25; AJ 169, 314; arXiv:2502.01745;
https://ui.adsabs.harvard.edu/abs/2025AJ....169..314H) in numpy, in the style of the tutorial, and compare it with our
Cannon and with simple linear baselines on the data of Chapter 10. Describe it as **a numpy implementation of the
method of H25**, not a reproduction of their code: v1 review found that the published code
(https://github.com/DHortaD/Lux, 2025-02-04, no licence) departs from the paper in its last training stage (§3).

## 2. The model (H25 §3)

For star n: labels ℓ_n (M), label errors σℓ_n (ASPCAP), fluxes f_n (Λ pixels), flux errors σ_n (∞ where masked);
latent vector z_n (P). Parameters: A (M×P), B (Λ×P), pixel scatter s (Λ).

    ℓ_n = A z_n + noise,  noise ~ N(0, σℓ_n²);      f_n = B z_n + noise,  noise ~ N(0, σ_n² + s²)

Objective to minimise (−2 ln L plus the code's Gaussian prior on z; masked entries — σ = ∞ or σℓ = ∞ — are excluded
from both the residual and the log-variance terms):

    Q = Σ_nm (ℓ_nm − a_m·z_n)²/σℓ_nm² + Σ_nλ [ (f_nλ − b_λ·z_n)²/(σ_nλ² + s_λ²) + ln(σ_nλ² + s_λ²) ] + Σ_n |z_n|²/τ²

τ = 1000 as in the code (`scatters.py:259`, notebook `l2_reg_strength = 1000`). The paper's eq. 7–8 ("ln L + Ω Σz²",
Ω = 10³) cannot be used literally: added to a maximised log-likelihood it rewards large z. Note: the prior on z alone
does not fix the scale degeneracy (z → c z, A, B → A/c, B/c lowers the penalty for c → 0); τ is meaningful only
together with the fixed schedule below.

## 3. Training schedule — what we implement, and how the published code differs

Initialisation (H25 Table 1, eq. 9; `init_latents.py`): z_n = [1, (ℓ_n − median)/((p97.5 − p2.5)/4), 0, …, 0];
A, B ~ U(0, 1) (seeded); s = e⁻⁸. P = 49 (the notebook's value; the paper says 4M = 48, the CV script uses
[13, 24, 48, 96]; we test both 48 and 49 — §6, E2).

Stage 1, 5 iterations, s = 0, no prior: A-step (labels, z fixed), B-step (fluxes, z fixed), z-step (labels and
fluxes, A and B fixed). Each is a weighted linear least-squares problem.

**Null space (v1 review, finding 2).** With the zero-padded initial z, Z has rank ≤ M + 1 = 13. A minimum-norm solve
of the first A- and B-steps sets the coefficients of the 36 unused latent dimensions to zero, and the following z-steps
keep them at zero: the model would stay at rank 13 whatever P is. H25 instead take Gauss–Newton steps from the random
initial A, B, which keep the null-space components of the starting values and so can activate the extra dimensions.
We implement each block update as one incremental least-squares step from the current values,

    c_new = c_old + D⁺ (y − D c_old)        (D the whitened design, D⁺ via SVD/lstsq),

which minimises the same conditional residual as the minimum-norm solution but keeps the null-space part of c_old.
Unit test: rank of Z after training > 13 for P = 49 with incremental steps, = 13 with minimum-norm steps. The chapter
shows both, because it changes what P means.

Stage 2, once, as written in the paper (§4.2): (i) s-step at fixed B, z; (ii) B-step with weights 1/(σ² + s²) using
the new s; (iii) z-step with the new B, the new s and the prior. **The published code differs** (`scatters.py:323–331`,
verified): its z-step optimises z *jointly with s*, and is passed the *old* B and the *initial* s, not the ones just
fitted; it returns the new B with the z and s fitted against the old B. We follow the paper and say so in the text.

Consequence (v1 review): in this schedule τ affects only the final training z, after A, B and s are fixed, so it
cannot change test predictions. We therefore do not scan τ (v1's E2 scan would have been inert); a unit test documents
this. If we later iterate stage 2, τ becomes active and must be tuned.

Scatter: `fit_scatter` (Chapter 2, grid + parabola) is an approximation to the ML solve. Stage 2 accepts the new s at a
pixel only if it does not increase that pixel's Q (compare with the previous value).

Test step (H25 eq. 10; `scatters.py:361–384`, no prior): z_n = argmin of the flux term at fixed B, s — weighted least
squares; labels ℓ_n = A z_n. Formal covariance A (BᵀW_nB)⁻¹ Aᵀ, conditional on A, B, s being exact and BᵀW_nB being
well conditioned (report its condition number; no pseudoinverse that hides unconstrained directions).

Code: new shared block `src/lux.py` (~60 lines). Masks: the tutorial's convention (finite placeholder flux, σ = ∞).
The reference code mixes σ = 9999 and ivar = 0.01 for the same bad pixels (`load_data.py:103–118`); we do not.

## 4. Unit tests (`tests/test_lux.py`, synthetic data from a known linear latent model)

1. Each block step equals an independent weighted least-squares solve (incremental form: same residual; null-space
   part of the starting value kept).
2. Rank: zero-padded initial z → rank > M + 1 after training with incremental steps; = M + 1 with minimum-norm steps.
3. Stage-1 objective (no s, no prior) does not increase over iterations; stage-2 Q does not increase over its steps;
   the reported Q equals Q recomputed from the returned parameters (catches stale-parameter wiring of the kind in the
   reference code).
4. τ is inert for test predictions under the paper's schedule (documents §3).
5. Masks: NaN/∞ entries, whole missing labels and pixels, stars without usable flux drop out; no ∞ from ln(σ² + s²).
6. Leakage: changing hidden labels (σℓ = ∞) or test-set labels does not change the fit or the flux-only predictions;
   inputs are not modified.
7. Test step: recovers labels of synthetic stars generated with fixed A, B, s; Monte Carlo scatter of the recovered
   labels matches the formal covariance; finite-difference check of ∂ℓ̂/∂f = A(BᵀWB)⁻¹BᵀW.
8. Scatter step never increases Q, including boundary cases (s = 0, s at the grid edge).

## 5. Reference comparison (mandatory, small)

In a separate venv with JAX and the jaxopt version of early 2025, run the reference stage-1 functions (`optimise.py`)
on small synthetic arrays (not their data loader, which unpickles with dill and rewrites caches) and compare our
stage-1 predictions after each block. Not redistributed (no licence). Stage 2 is compared only to document the
difference in §3.

## 6. Experiments for the chapter (data and splits of Chapter 10)

12 labels; training 1308 (stratified in label space, S/N > 200 mostly), validation 328, test 741 (S/N 50–100). This is
a comparison on our data, not a reproduction of H25's 5000-star results (different selection and S/N). Hyperparameters
are chosen on the validation stars using labels inferred from their **spectra only**; the test stars are used once.
All label comparisons are against ASPCAP; differences between methods come with paired bootstrap intervals.

E1. Lux vs our 12-label Cannon (Chapter 10) and vs ridge regression of labels on fluxes (mandatory baseline; λ chosen
    on validation): robust std, bias, RMSE of each label, on validation and test stars, by [Fe/H] bin.
    Spectra: predicted from the ASPCAP labels (Lux: z from labels, then B z; Cannon: model at the labels) with a common
    noise model (σ² + the Cannon's s²), as in Chapter 10 — a prediction, unlike reconstructing a spectrum from z fitted
    to the same pixels. Label response to flux: ∂ℓ̂/∂f = A(BᵀWB)⁻¹BᵀW for Lux and (JᵀWJ)⁻¹JᵀW for the Cannon, near
    Mg I lines (the same direction for both).
E2. P ∈ {13, 25, 48, 49} with incremental steps, and P = 49 with minimum-norm steps (rank 13): validation precision.
E3. Missing labels: hide [N/Fe] (σℓ = ∞) for a random half of the training stars *before* computing pivots, scales
    and initial z. Compare: Lux on all stars; Lux on the complete half; Cannon on the complete half; ridge per label on
    all stars with that label. No pass/fail on which wins.
E4. Noisy labels: add noise of 2σ to the training labels (three draws); Lux told the true total error √5 σ, and told
    the original σ (underestimated); Cannon (labels treated as exact). Corrupt before initialisation and scaling.
E5. (If time allows) PCA + regression, partial least squares.

Acceptance criteria (frozen before running): unit tests pass; reference stage-1 agreement to numerical precision on
synthetic data; notebook < 15 min locally (to be measured on real data before writing the chapter). All E1–E5
outcomes are reported as measured, with no criterion on which method wins.

## 7. Claims we will not repeat without our own evidence

H25 interpret tighter label distributions as better precision, write of "deconvolution", and suggest linear models
extrapolate safely; shrinkage and inherited abundance correlations can also tighten distributions. Their flux-
perturbation errors use σ² only while the model includes s² (`horta25_paper.txt:1053–1056`). The supplied CV scripts
do not match the paper's grid (`run_kfold.py:36–39`). We quote H25's numbers (Fig. 7) only after reading them from
the PDF.
