"""Unit tests of the Lux block (src/lux.py) on synthetic data drawn from a known linear latent model.

Run: pytest tests/test_lux.py
"""
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
ns = {"np": np}
for block in ["cannon.py", "lux.py"]:
    exec((SRC / block).read_text(), ns)
globals().update({k: v for k, v in ns.items() if not k.startswith(("__", "test"))})


def latent_data(N=300, M=4, L=60, P_true=6, label_err=0.02, seed=0, masked=0.0):
    """Stars with latents z = [1, ...] (P_true), labels A z + noise, fluxes B z + noise (errors 0.005-0.02)."""
    rng = np.random.default_rng(seed)
    Z = np.c_[np.ones(N), rng.normal(0, 1, (N, P_true - 1))]
    A, B = rng.normal(0, 1, (M, P_true)), rng.normal(0, 0.05, (L, P_true))
    B[:, 0] += 1                                               # fluxes near 1
    lab_err = np.full((N, M), label_err)
    err = rng.uniform(0.005, 0.02, (N, L))
    lab = Z @ A.T + rng.normal(0, 1, (N, M)) * lab_err
    flux = Z @ B.T + rng.normal(0, 1, (N, L)) * err
    if masked:
        err[rng.random((N, L)) < masked] = np.inf
        flux[~np.isfinite(err)] = np.nan
    return dict(Z=Z, A=A, B=B, lab=lab, lab_err=lab_err, flux=flux, err=err)


def test_block_steps_solve_their_least_squares_problems():
    d = latent_data()
    wl, l0 = lux_weights(d["lab"], d["lab_err"])
    Z = d["Z"]
    A0 = np.random.default_rng(1).uniform(0, 1, d["A"].shape)
    for incremental in [False, True]:
        A = lux_a_step(Z, l0, wl, A0, incremental)
        for m in range(A.shape[0]):                             # Z has full rank: unique solution
            sw = np.sqrt(wl[:, m])
            ref = np.linalg.lstsq(Z * sw[:, None], l0[:, m] * sw, rcond=None)[0]
            assert np.allclose(A[m], ref, atol=1e-8)


def test_incremental_step_keeps_the_unconstrained_part_of_the_start():
    """Zero-padded latents (rank M + 1): the incremental step reaches the same residual as the minimum-norm solve but
    leaves the coefficients of the unused latent dimensions at their starting values; minimum-norm sets them to 0."""
    d = latent_data()
    wl, l0 = lux_weights(d["lab"], d["lab_err"])
    A0, _, Z0 = lux_init(d["lab"], 9, 60, seed=2)                # columns 5..8 of Z0 are zero
    inc, mn = lux_a_step(Z0, l0, wl, A0, True), lux_a_step(Z0, l0, wl, A0, False)
    assert np.allclose(Z0 @ inc.T, Z0 @ mn.T, atol=1e-8)
    assert np.allclose(inc[:, 5:], A0[:, 5:]) and np.allclose(mn[:, 5:], 0)


def test_prior_shrinks_weakly_constrained_latents_instead_of_keeping_the_start():
    """Review finding: with a prior, a direction the data barely constrain must shrink towards 0, not keep its start
    (normal matrix diag(1e12 + 1, 1) with tau = 1, start 7: the ridge solution is 0)."""
    A = np.zeros((1, 2)); B = np.array([[1e6, 0.0]])
    z = lux_z_step(A, B, np.zeros((1, 1)), np.zeros((1, 1)), np.array([[1e6]]), np.ones((1, 1)), np.array([[0.0, 7.0]]),
                   tau=1.0)
    assert np.allclose(z, [[1e12 / (1e12 + 1), 0.0]])


def test_nonpositive_errors_are_masked():
    w, v = lux_weights(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.0, -1.0]))
    assert np.allclose(w, [100, 0, 0]) and np.allclose(v, [1, 0, 0])


def test_b_and_z_steps_solve_their_least_squares_problems():
    d = latent_data()
    wl, l0 = lux_weights(d["lab"], d["lab_err"])
    w, f0 = lux_weights(d["flux"], d["err"])
    B = lux_b_step(d["Z"], f0, w, np.zeros_like(d["B"]))
    for j in [0, 17, 59]:
        sw = np.sqrt(w[:, j])
        assert np.allclose(B[j], np.linalg.lstsq(d["Z"] * sw[:, None], f0[:, j] * sw, rcond=None)[0], atol=1e-8)
    Z = lux_z_step(d["A"], d["B"], l0, wl, f0, w, np.zeros_like(d["Z"]))
    for n in [0, 123]:
        D = np.vstack([d["A"] * np.sqrt(wl[n])[:, None], d["B"] * np.sqrt(w[n])[:, None]])
        y = np.r_[l0[n] * np.sqrt(wl[n]), f0[n] * np.sqrt(w[n])]
        assert np.allclose(Z[n], np.linalg.lstsq(D, y, rcond=None)[0], atol=1e-8)


def test_incremental_steps_use_all_latent_dimensions_minimum_norm_steps_do_not():
    d = latent_data()
    ranks = {}
    for incremental in [True, False]:
        _, _, _, Z, _ = lux_train(d["lab"], d["lab_err"], d["flux"], d["err"], P=9, incremental=incremental)
        ranks[incremental] = np.linalg.matrix_rank(Z, tol=1e-8 * np.abs(Z).max())
    assert ranks[True] > 5 and ranks[False] == 5                 # M + 1 = 5


def test_objective_never_increases_and_is_reported_correctly():
    d = latent_data(masked=0.05)
    A, B, s, Z, hist = lux_train(d["lab"], d["lab_err"], d["flux"], d["err"], P=9, n_iter=8, tau=10.0)
    assert np.all(np.diff(hist[:-1]) <= 1e-8 * np.abs(hist[:-1]).max())          # stage 1
    wl, l0 = lux_weights(d["lab"], d["lab_err"])
    w, f0 = lux_weights(d["flux"], d["err"])
    assert np.isclose(hist[-1], lux_objective(A, B, Z, s, l0, wl, f0, w, 10.0))   # reported = recomputed


def test_stage_two_steps_never_increase_the_objective():
    d = latent_data(masked=0.05)
    wl, l0 = lux_weights(d["lab"], d["lab_err"])
    w, f0 = lux_weights(d["flux"], d["err"])
    A, B, s, Z, _ = lux_train(d["lab"], d["lab_err"], d["flux"], d["err"], P=9, tau=10.0)
    q = [lux_objective(A, B, Z, s, l0, wl, f0, w, 10.0)]
    s = lux_scatter_step(B, Z, f0, w, s)
    q.append(lux_objective(A, B, Z, s, l0, wl, f0, w, 10.0))
    wv = w / (1 + w * s**2)
    B = lux_b_step(Z, f0, wv, B)
    q.append(lux_objective(A, B, Z, s, l0, wl, f0, w, 10.0))
    Z = lux_z_step(A, B, l0, wl, f0, wv, Z, 10.0)
    q.append(lux_objective(A, B, Z, s, l0, wl, f0, w, 10.0))
    assert np.all(np.diff(q) <= 1e-8 * abs(q[0])), q


def test_tau_does_not_change_test_predictions_in_this_schedule():
    d = latent_data()
    out = [lux_train(d["lab"], d["lab_err"], d["flux"], d["err"], P=9, tau=t) for t in [1000.0, 1.0]]
    assert all(np.allclose(a, b) for a, b in zip(out[0][:3], out[1][:3]))         # A, B, s identical
    assert not np.allclose(out[0][3], out[1][3])                                    # only the training Z changes


def test_masks_and_missing_labels_drop_out():
    d = latent_data(masked=0.1)
    lab = d["lab"].copy()
    lab[:50, 1] = np.nan                                         # missing labels
    A, B, s, Z, hist = lux_train(lab, d["lab_err"], d["flux"], d["err"], P=9)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(hist))
    lab2 = lab.copy()
    lab2[:50, 1] = 1e6                                           # any value with infinite error is ignored
    err2 = d["lab_err"].copy()
    err2[:50, 1] = np.inf
    out2 = lux_train(lab2, err2, d["flux"], d["err"], P=9)
    assert np.allclose(A, out2[0]) and np.allclose(B, out2[1])


def test_inputs_are_not_modified():
    d = latent_data(masked=0.1)
    copies = {k: d[k].copy() for k in ["lab", "lab_err", "flux", "err"]}
    lux_train(d["lab"], d["lab_err"], d["flux"], d["err"], P=9)
    assert all(np.array_equal(d[k], copies[k], equal_nan=True) for k in copies)


def test_noiseless_training_reproduces_labels_and_spectra():
    d = latent_data(label_err=1e-6)
    flux = d["Z"] @ d["B"].T                                    # noiseless
    lab = d["Z"] @ d["A"].T
    A, B, s, Z, _ = lux_train(lab, np.full_like(lab, 1e-3), flux, np.full_like(flux, 1e-3), P=6, n_iter=30)
    assert np.allclose(Z @ A.T, lab, atol=1e-6) and np.allclose(Z @ B.T, flux, atol=1e-6)


def test_test_step_recovers_labels_with_correct_formal_errors():
    """Known A, B, s: labels of new stars from their spectra; pulls of unit width; linear response matches finite
    differences."""
    d = latent_data(N=2000, seed=3)
    s = np.full(60, 0.003)
    rng = np.random.default_rng(4)
    flux = d["Z"] @ d["B"].T + rng.normal(0, 1, d["err"].shape) * np.sqrt(d["err"]**2 + s**2)
    lab, cov, cond, leak = lux_test(flux, d["err"], d["A"], d["B"], s)
    assert np.all(leak == 0)
    pull = (lab - d["Z"] @ d["A"].T) / np.sqrt(np.einsum('nmm->nm', cov))
    assert np.allclose(pull.std(axis=0), 1, atol=0.05) and np.all(np.abs(pull.mean(axis=0)) < 0.1)
    f1 = flux[:1].copy()
    f1[0, 7] += 1e-3
    lab1 = lux_test(f1, d["err"][:1], d["A"], d["B"], s)[0]
    W = 1 / (d["err"][0]**2 + s**2)
    K = d["A"] @ np.linalg.inv((d["B"].T * W) @ d["B"]) @ (d["B"].T * W)       # d labels / d flux
    assert np.allclose((lab1 - lab[:1])[0] / 1e-3, K[:, 7], rtol=1e-6)


def test_scatter_step_never_increases_the_pixel_objective():
    d = latent_data(masked=0.05)
    w, f0 = lux_weights(d["flux"], d["err"])
    for s0 in [np.zeros(60), np.full(60, 0.5), np.full(60, 1e-4)]:
        s = lux_scatter_step(d["B"], d["Z"], f0, w, s0)
        q = lambda s_: lux_objective(d["A"], d["B"], d["Z"], s_, np.zeros_like(d["lab"]), np.zeros_like(d["lab"]),
                                     f0, w)
        assert q(s) <= q(s0) + 1e-9
