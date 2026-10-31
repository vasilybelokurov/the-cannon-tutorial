"""Unit tests of the shared Cannon code (src/cannon.py, src/more.py) on synthetic data with known answers.

Run: pytest tests/test_cannon.py
"""
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
ns = {"np": np}
for block in ["cannon.py", "more.py"]:                     # the blocks are pasted into notebooks; run them the same way
    exec((SRC / block).read_text(), ns)
globals().update({k: v for k, v in ns.items() if not k.startswith(("__", "test"))})
run_test_more = ns["test_more"]   # renamed: pytest would collect a module-level test_* function


def synthetic(N, K, P, order=2, sigma=0.01, seed=0, theta=None):
    """Scaled labels x (N, K), coefficients theta (P, D) and noisy fluxes with errors sigma."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.5, (N, K))
    V = vectorize_n(x, order)
    if theta is None:
        theta = np.c_[np.ones(P), rng.normal(0, 0.05, (P, V.shape[1] - 1))]
    flux = V @ theta.T + rng.normal(0, sigma, (N, P))
    return x, theta, flux, np.full((N, P), sigma)


def normal_equations(V, flux, var):
    w = 1 / var
    return np.einsum("np,nd,ne->pde", w, V, V), np.einsum("np,nd->pd", w * flux, V)


def test_vectorize_n_order2_is_vectorize():
    x = np.random.default_rng(1).normal(size=(20, 4))
    assert np.array_equal(vectorize_n(x, 2), vectorize(x))
    assert np.allclose(dvectorize_n(x[0], 2), dvectorize(x[0]))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_dvectorize_n_matches_finite_differences(order):
    x = np.random.default_rng(2).normal(size=4)
    h = 1e-6
    num = np.array([(vectorize_n(x + h * e, order)[0] - vectorize_n(x - h * e, order)[0]) / (2 * h)
                    for e in np.eye(4)]).T
    assert np.allclose(num, dvectorize_n(x, order), atol=1e-7)
    assert vectorize_n(x, order).shape[1] == {1: 5, 2: 15, 3: 35}[order]      # (K + order)! / (K! order!)


def test_test_more_recovers_labels_of_a_cubic_model():
    x, theta, _, _ = synthetic(1, 3, 400, order=3)
    flux = vectorize_n(x, 3)[0] @ theta.T                  # noiseless
    xf, cov, chi2, n, ok = run_test_more(flux, np.full(400, 0.01), theta, np.zeros(400), order=3)
    assert ok and np.allclose(xf, x[0], atol=1e-6)


def test_several_starting_points_escape_a_local_minimum():
    """One label, flux = 1 + a x^2 + c x with small c: chi^2 has minima near x = +1 (true) and x = -1 (worse)."""
    rng = np.random.default_rng(3)
    P = 200
    theta = np.c_[np.ones(P), 0.005 * rng.normal(1, 0.1, P), rng.normal(0.05, 0.01, P)]   # [1, c, a]
    flux = vectorize_n(np.array([[1.0]]), 2)[0] @ theta.T
    err, s = np.full(P, 0.01), np.zeros(P)
    one = run_test_more(flux, err, theta, s, starts=[[-0.8]])
    many = run_test_more(flux, err, theta, s, starts=[[-0.8], [0.8]])
    assert one[0][0] < 0 and one[2] > 1                    # stuck in the wrong minimum
    assert abs(many[0][0] - 1) < 1e-6 and many[2] < 1e-12


def test_l1_with_lambda_zero_is_weighted_least_squares():
    x, _, flux, err = synthetic(300, 3, 50)
    V = vectorize_n(x)
    assert np.allclose(fit_theta_l1(V, flux, err**2)[0], fit_theta(V, flux, err**2)[0], atol=1e-10)


@pytest.mark.parametrize("lam", [1e2, 1e4])
def test_l1_solution_satisfies_optimality_conditions(lam):
    """Minimum of chi^2 + lam |theta_{1:}|_1: gradient g = 2(A theta - b); g_d = -lam sign(theta_d) where theta_d != 0,
    |g_d| <= lam where theta_d = 0, and g_0 = 0."""
    x, _, flux, err = synthetic(300, 3, 50)
    V = vectorize_n(x)
    th = fit_theta_l1(V, flux, err**2, lam, tol=1e-10)[0]
    A, b = normal_equations(V, flux, err**2)
    g = 2 * (np.einsum("pde,pe->pd", A, th) - b)
    nz = th[:, 1:] != 0
    assert np.all(np.abs(g[:, 0]) < 1e-3 * lam)
    assert np.all(np.abs(g[:, 1:] + lam * np.sign(th[:, 1:]))[nz] < 1e-3 * lam)
    assert np.all(np.abs(g[:, 1:])[~nz] <= lam * (1 + 1e-6))
    assert 0 < nz.mean() < 1                               # some, not all, coefficients are zero


def test_l1_with_huge_lambda_leaves_the_weighted_mean():
    x, _, flux, err = synthetic(300, 3, 50)
    th = fit_theta_l1(vectorize_n(x), flux, err**2, 1e12)[0]
    assert np.all(th[:, 1:] == 0) and np.allclose(th[:, 0], flux.mean(axis=0))


@pytest.mark.parametrize("lam", [0.0, 1e3])
def test_censored_coefficients_are_zero_and_the_rest_solve_the_reduced_problem(lam):
    x, _, flux, err = synthetic(300, 3, 40)
    V = vectorize_n(x)
    allowed = np.ones((40, 3), bool)
    allowed[:20, 2] = False                                # label 2 may not act on the first 20 pixels
    mask = censor_mask(allowed)
    th = fit_theta_l1(V, flux, err**2, lam, mask, tol=1e-10)[0]
    assert np.all(th[~mask] == 0)
    keep = mask[0]                                         # same mask for the first 20 pixels
    ref = fit_theta_l1(V[:, keep], flux[:, :20], err[:, :20]**2, lam, tol=1e-10)[0]
    assert np.allclose(th[:20][:, keep], ref, atol=1e-8)
    assert keep.sum() == 1 + 2 + 3                         # 1, x0, x1, x0^2, x0 x1, x1^2


def test_l1_beats_least_squares_when_the_truth_is_sparse_and_the_data_few():
    """8 labels, 45 coefficients, 120 training stars; each pixel depends on one label only."""
    K, P, rng = 8, 100, np.random.default_rng(4)
    D = 1 + len(label_terms(K))
    theta = np.zeros((P, D))
    theta[:, 0] = 1
    theta[np.arange(P), 1 + rng.integers(0, K, P)] = rng.normal(0, 0.05, P)
    x, _, flux, err = synthetic(120, K, P, theta=theta, seed=5)
    xt, _, ft, et = synthetic(2000, K, P, theta=theta, seed=6)
    V, Vt = vectorize_n(x), vectorize_n(xt)
    rms = lambda th: np.sqrt(np.mean((Vt @ th.T - (Vt @ theta.T))**2))      # error of the model spectra
    r0, r1 = rms(fit_theta_l1(V, flux, err**2)[0]), rms(fit_theta_l1(V, flux, err**2, 3e2)[0])
    assert r1 < 0.6 * r0, (r0, r1)


def test_train_more_without_extras_is_train():
    x, _, flux, err = synthetic(300, 3, 50)
    th0, _, s0 = train(x, flux, err)
    th1, s1 = train_more(x, flux, err)
    assert np.allclose(th0, th1, atol=1e-10) and np.allclose(s0, s1)


def test_in_hull_sees_what_the_label_ranges_miss():
    """Labels along a line: a point inside both 1-D ranges but off the line is outside the hull."""
    t = np.random.default_rng(7).uniform(-1, 1, 200)
    X = np.c_[t, t + np.random.default_rng(8).normal(0, 0.05, 200)]
    assert list(in_hull([[0, 0], [0.8, -0.8], [3, 3]], X)) == [True, False, False]
