# --- More of The Cannon (Chapter 10): any polynomial order, censoring, L1 regularisation, several starting points,
# --- and a test of whether labels lie inside the training set -------------------------------------------------
from itertools import combinations_with_replacement
import warnings
from scipy.optimize import linprog


def label_terms(K, order=2):
    """The terms of the label vector after the leading 1: tuples of label indices, e.g. (0,), (0, 1), (2, 2, 2)."""
    return [t for p in range(1, order + 1) for t in combinations_with_replacement(range(K), p)]


def vectorize_n(x, order=2):
    """Label vector of polynomial order `order` for scaled labels x (N, K): (N, D). For order 2 it equals vectorize."""
    x = np.atleast_2d(x)
    return np.hstack([np.ones((len(x), 1))] + [np.prod(x[:, list(t)], axis=1, keepdims=True)
                                                for t in label_terms(x.shape[1], order)])


def dvectorize_n(x, order=2):
    """Derivative dv/dx of the label vector of one star (x of shape (K,)): (D, K)."""
    T = label_terms(len(x), order)
    J = np.zeros((1 + len(T), len(x)))
    for i, t in enumerate(T):
        for j in range(len(t)):                              # d/dx of x_a x_b x_c ...: drop one factor at a time
            J[1 + i, t[j]] += np.prod(x[list(t[:j] + t[j + 1:])])
    return J


def censor_mask(allowed, order=2):
    """Coefficients allowed at each pixel, (P, D), from the labels allowed at each pixel, (P, K): a term may be
    non-zero only if all the labels in it are allowed. The leading coefficient (the mean spectrum) always is."""
    T = label_terms(allowed.shape[1], order)
    return np.hstack([np.ones((len(allowed), 1), bool)] + [allowed[:, list(t)].all(axis=1, keepdims=True) for t in T])


def fit_theta_l1(V, flux, var, lam=0.0, mask=None, theta0=None, chunk=2048, tol=1e-8, max_iter=20000, kappa=0.1):
    """At every pixel minimise chi^2 + lam * sum_{d >= 1} |theta_d| (Casey et al. 2016), with theta_d = 0 where
    mask (P, D) is False. lam = 0: weighted least squares, solved directly. lam > 0: ADMM (Boyd et al. 2011, §6.4) on rescaled coefficients,
    all pixels of a chunk at once, started from theta0 (P, D) if given. Weights 1/var (N, P); zero-weight entries drop
    out. Returns theta (P, D) and the number of ADMM iterations used (0 for lam = 0)."""
    N, D = V.shape
    P = flux.shape[1]
    mask = np.ones((P, D), bool) if mask is None else mask
    w = 1 / var
    flux = np.where(w > 0, flux, 0.0)
    VV = (V[:, :, None] * V[:, None, :]).reshape(N, D * D).T
    pen = np.r_[0.0, np.ones(D - 1)]                             # theta_0 (the mean spectrum) is not penalised
    theta, n_used = np.zeros((P, D)), 0
    for p0 in range(0, P, chunk):
        sl = slice(p0, p0 + chunk)
        A = (VV @ w[:, sl]).T.reshape(-1, D, D)                  # V^T W V at each pixel
        b = (V.T @ (w[:, sl] * flux[:, sl])).T                   # V^T W f
        m = mask[sl]
        if lam == 0:                                             # censored rows/columns replaced by the identity
            Am = A * (m[:, :, None] & m[:, None, :]) + np.eye(D) * ~m[:, None, :]
            theta[sl] = np.einsum('pde,pe->pd', np.linalg.pinv(Am), b * m) * m
            continue
        # Rescale each coefficient so that the curvature of chi^2 along it is 1: theta = d * y. Without this, linear and
        # quadratic terms have curvatures that differ by orders of magnitude and ADMM crawls. |theta_k| = d_k |y_k|.
        dg = np.einsum('pdd->pd', A)
        d = 1 / np.sqrt(np.where(dg > 0, 2 * dg, 1.0))
        As, bs = A * d[:, :, None] * d[:, None, :], b * d
        # ADMM: split y (quadratic part) from z (L1 part and mask), y = z; u is the scaled dual variable. The step
        # size rho scales with the soft threshold lam * d, which in these units sets how far the solution moves.
        rho = np.minimum(kappa * lam * np.median(d, axis=1), 1.0)[:, None]   # capped at the (unit) curvature
        M = np.linalg.inv(2 * As + rho[:, :, None] * np.eye(D))   # factorised once, used at every iteration
        z = np.zeros_like(b) if theta0 is None else theta0[sl] * m / d
        u = np.zeros_like(b)
        thr = pen * lam * d / rho                                # soft threshold for each scaled coefficient
        act = np.arange(len(b))                                  # pixels not yet converged
        for it in range(max_iter):
            y = np.einsum('pde,pe->pd', M[act], 2 * bs[act] + rho[act] * (z[act] - u[act]))
            v = y + u[act]
            z_new = np.where(m[act], np.sign(v) * np.maximum(np.abs(v) - thr[act], 0), 0.0)
            da = d[act]                                          # convergence judged on theta, not on y
            done = (np.abs(da * (y - z_new)).max(axis=1) < tol) & (np.abs(da * (z_new - z[act])).max(axis=1) < tol)
            z[act], u[act] = z_new, v - z_new
            act = act[~done]
            if len(act) == 0:
                break
        if len(act):
            warnings.warn(f'fit_theta_l1: {len(act)} pixels did not converge in {max_iter} iterations')
        n_used = max(n_used, it + 1)
        theta[sl] = z * d                                        # z has the exact zeros
    return theta, n_used


def train_more(x, flux, err, lam=0.0, mask=None, order=2, n_iter=3):
    """Training step with polynomial order, censoring mask and L1 regularisation; scatter by maximum likelihood as in
    `train` (Chapter 2). Returns theta (P, D) and s (P,)."""
    V, err2 = vectorize_n(x, order), err**2
    s, theta = np.zeros(flux.shape[1]), None
    for _ in range(n_iter):
        theta = fit_theta_l1(V, flux, err2 + s**2, lam, mask, theta)[0]     # each fit starts from the previous one
        s = fit_scatter(flux - V @ theta.T, err2)
    return fit_theta_l1(V, flux, err2 + s**2, lam, mask, theta)[0], s


def test_more(flux, err, theta, s, order=2, starts=None):
    """Test step for one spectrum, as `test` (Chapter 2) but for any polynomial order and from several starting
    points (rows of `starts`, scaled labels; default: the median labels, 0); of the fits that converged, the one with
    the lowest chi^2 is kept.
    Returns scaled labels, covariance, chi^2, number of pixels used, converged?"""
    K = next(k for k in range(1, 100) if 1 + len(label_terms(k, order)) == theta.shape[1])
    ok = np.isfinite(flux) & np.isfinite(err) & (theta[:, 0] != 0)
    w = 1 / np.sqrt(err[ok]**2 + s[ok]**2)
    th, f = theta[ok], flux[ok]
    res = lambda x: (f - th @ vectorize_n(x, order)[0]) * w
    jac = lambda x: -(th @ dvectorize_n(x, order)) * w[:, None]
    fits = [least_squares(res, x0, jac=jac, method='lm') for x0 in (np.zeros((1, K)) if starts is None else
                                                                     np.atleast_2d(starts))]
    r = min(fits, key=lambda r: (not r.success, np.sum(r.fun**2)))   # converged fits first, then lowest chi^2
    return r.x, np.linalg.pinv(r.jac.T @ r.jac), np.sum(r.fun**2), ok.sum(), r.success


def in_hull(x, X):
    """Is each row of x (M, K) inside the convex hull of the training labels X (N, K)? True if x is a weighted mean
    of training points with non-negative weights summing to 1: a linear-programming feasibility problem, which works
    in any number of dimensions (a Delaunay triangulation does not, beyond a few). Labels are scaled by the standard
    deviation of X so that kelvin and dex are treated alike."""
    sd = X.std(axis=0)
    A = np.vstack([(X / sd).T, np.ones(len(X))])
    status = np.array([linprog(np.zeros(len(X)), A_eq=A, b_eq=np.r_[p / sd, 1], bounds=(0, None), method='highs').status
                       for p in np.atleast_2d(x)])
    if not np.all((status == 0) | (status == 2)):              # 0: feasible (inside), 2: infeasible (outside)
        raise RuntimeError(f'linprog failed with status {set(status) - {0, 2}}')
    return status == 0
