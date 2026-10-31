# --- The Cannon (as written in Chapter 2) -----------------------------------------------------------------
from scipy.optimize import least_squares


def vectorize(x):
    """Quadratic label vector v(x) = [1, x_k, x_k x_l (k <= l)] for scaled labels x of shape (N, K): (N, D)."""
    x = np.atleast_2d(x)
    k, l = np.triu_indices(x.shape[1])
    return np.hstack([np.ones((len(x), 1)), x, x[:, k] * x[:, l]])


def dvectorize(x):
    """Derivative dv/dx of the label vector of one star (x of shape (K,)): (D, K)."""
    K = len(x)
    k, l = np.triu_indices(K)
    J = np.zeros((1 + K + len(k), K))
    J[1:1 + K] = np.eye(K)
    for i, (a, b) in enumerate(zip(k, l)):
        J[1 + K + i, a] += x[b]
        J[1 + K + i, b] += x[a]
    return J


def fit_theta(V, flux, var):
    """Weighted least squares at every pixel: theta (P, D) and its covariance (P, D, D); weights 1/var (N, P).
    Zero-weight entries (var = inf) drop out of the sums."""
    w = 1 / var
    flux = np.where(w > 0, flux, 0.0)                       # NaN flux at zero weight must not poison the sums
    N, Dv = V.shape
    A = ((V[:, :, None] * V[:, None, :]).reshape(N, Dv * Dv).T @ w).T.reshape(-1, Dv, Dv)
    b = (V.T @ (w * flux)).T
    cov = np.linalg.pinv(A)                                 # pinv: pixels without data give theta = 0, not an error
    return np.einsum('pde,pe->pd', cov, b), cov


S_GRID = np.r_[0, np.logspace(-4, 0, 41)]                  # 0, then 1e-4 ... 1 in steps of 0.1 dex


def fit_scatter(resid, err2):
    """Maximum-likelihood scatter s per pixel for fixed residuals: best value on the grid S_GRID, refined by a
    parabola in log s (Chapter 2). Zero-weight entries (err2 = inf) are ignored."""
    use = np.isfinite(err2) & np.isfinite(resid)
    r2 = np.where(use, resid**2, 0.0)
    e2 = np.where(use, err2, 1.0)
    nll = np.array([np.sum(use * (r2 / (e2 + s**2) + np.log(e2 + s**2)), axis=0) for s in S_GRID])
    best = np.argmin(nll, axis=0)
    j = np.clip(best, 2, len(S_GRID) - 2)
    p = np.arange(nll.shape[1])
    a, b, c = nll[j - 1, p], nll[j, p], nll[j + 1, p]
    shift = np.clip(0.5 * (a - c) / np.where(a - 2 * b + c > 0, a - 2 * b + c, np.inf), -1, 1)   # stay near the grid optimum
    return np.where(best == 0, 0.0, S_GRID[j] * np.exp(shift * np.log(S_GRID[2] / S_GRID[1])))


def train(x, flux, err, n_iter=3):
    """Training step. x: scaled labels (N, K); flux, err: (N, P) (err = inf for pixels to ignore).
    Returns theta (P, D), its covariance (P, D, D) and the scatter s (P,)."""
    V, err2 = vectorize(x), err**2
    s = np.zeros(flux.shape[1])
    for _ in range(n_iter):
        theta, cov = fit_theta(V, flux, err2 + s**2)
        s = fit_scatter(flux - V @ theta.T, err2)
    theta, cov = fit_theta(V, flux, err2 + s**2)
    return theta, cov, s


def test(flux, err, theta, s, x0=None):
    """Test step for one spectrum: scaled labels (K,), covariance (K, K), chi^2, number of pixels used, converged?"""
    K = int(round((np.sqrt(8 * theta.shape[1] + 1) - 3) / 2))   # invert D = 1 + K + K(K+1)/2
    ok = np.isfinite(flux) & np.isfinite(err) & (theta[:, 0] != 0)
    w = 1 / np.sqrt(err[ok]**2 + s[ok]**2)
    th, f = theta[ok], flux[ok]
    res = lambda x: (f - th @ vectorize(x)[0]) * w
    jac = lambda x: -(th @ dvectorize(x)) * w[:, None]
    r = least_squares(res, np.zeros(K) if x0 is None else x0, jac=jac, method='lm')
    # covariance from the Jacobian at the optimum (pinv: no crash if a label is unconstrained); r.success: converged
    return r.x, np.linalg.pinv(r.jac.T @ r.jac), np.sum(r.fun**2), ok.sum(), r.success


def train_clean(x, flux, err, cut=3.0):
    """Train, drop the training stars whose mean squared residual per pixel (in units of sigma^2 + s^2) exceeds
    `cut`, and train again (Chapter 4). Returns theta, its covariance, s, and the boolean mask of stars kept."""
    theta, cov, s = train(x, flux, err)
    z2 = (flux - vectorize(x) @ theta.T)**2 / (err**2 + s**2)
    z2 = np.where(np.isfinite(err) & (theta[:, 0] != 0), z2, np.nan)
    keep = np.nanmean(z2, axis=1) < cut
    theta, cov, s = train(x[keep], flux[keep], err[keep])
    return theta, cov, s, keep


def scatter_stats(d):
    """Summary of differences d (e.g. Cannon - reference): mean (bias), standard deviation, root-mean-square error,
    robust standard deviation 1.4826 x median absolute deviation (= std for a Gaussian, insensitive to outliers),
    and the fraction of points more than 3 robust std from the median (the tails)."""
    d = np.asarray(d)[np.isfinite(d)]
    med = np.median(d)
    rob = 1.4826 * np.median(np.abs(d - med))
    return {'bias': d.mean(), 'std': d.std(), 'rmse': np.sqrt(np.mean(d**2)), 'robust': rob,
            'tail': np.mean(np.abs(d - med) > 3 * rob)}
