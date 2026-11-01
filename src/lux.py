# --- Lux (Horta et al. 2025): labels and spectra both generated from a latent vector per star (Chapter 11) ----
# labels l_n = A z_n + noise (errors sl_n), fluxes f_n = B z_n + noise (errors s_n and pixel scatter s); every step of
# the training is a weighted linear least-squares problem, solved here in closed form.


def lux_weights(values, errors):
    """Inverse variances, 0 where the error is not finite and positive or the value is not finite (masked pixels,
    missing labels), and the values with those entries set to 0 so that they cannot produce NaN."""
    ok = np.isfinite(errors) & (errors > 0) & np.isfinite(values)
    w = np.where(ok, 1 / np.where(ok, errors, 1.0)**2, 0.0)
    return w, np.where(w > 0, values, 0.0)


def lsq_update(G, h, c0, incremental=True, rcond=1e-10):
    """Solve the normal equations G c = h for a stack of problems (G: (K, P, P), h and c0: (K, P)).
    incremental=True: c0 + G^+ (h - G c0), which has the same residual as the minimum-norm solution but keeps the part
    of c0 that the data do not constrain (as a Gauss-Newton step from c0 does); incremental=False: G^+ h.
    Directions with eigenvalues of G below rcond times the largest count as unconstrained (rcond = 0: none do, for G
    that is positive definite, e.g. with a prior)."""
    vals, vecs = np.linalg.eigh(G)
    inv = np.where(vals > rcond * np.max(vals, axis=1, keepdims=True), 1 / np.where(vals > 0, vals, 1.0), 0.0)
    r = h - np.einsum('kpq,kq->kp', G, c0) if incremental else h
    step = np.einsum('kpq,kq,kq->kp', vecs, inv, np.einsum('kqp,kq->kp', vecs, r))
    return c0 + step if incremental else step


def lux_a_step(Z, lab, wl, A, incremental=True):
    """Label coefficients A (M, P) at fixed latents Z (N, P): one weighted least-squares problem per label."""
    G = np.einsum('nm,np,nq->mpq', wl, Z, Z)
    return lsq_update(G, np.einsum('nm,np->mp', wl * lab, Z), A, incremental)


def lux_b_step(Z, flux, w, B, incremental=True):
    """Flux coefficients B (Lambda, P) at fixed latents Z: one weighted least-squares problem per pixel."""
    N, P = Z.shape
    G = ((Z[:, :, None] * Z[:, None, :]).reshape(N, P * P).T @ w).T.reshape(-1, P, P)
    return lsq_update(G, (Z.T @ (w * flux)).T, B, incremental)


def lux_z_step(A, B, lab, wl, flux, w, Z, tau=np.inf, incremental=True):
    """Latents Z (N, P) at fixed A, B: one weighted least-squares problem per star, using its labels (weights wl) and
    fluxes (weights w), with a Gaussian prior of width tau on every latent (ridge term)."""
    P = A.shape[1]
    G = (w @ (B[:, :, None] * B[:, None, :]).reshape(-1, P * P)).reshape(-1, P, P)
    G += np.einsum('nm,mp,mq->npq', wl, A, A) + np.eye(P) / tau**2
    rcond = 0.0 if np.isfinite(tau) else 1e-10                # with a prior every direction is constrained by it
    return lsq_update(G, (w * flux) @ B + (wl * lab) @ A, Z, incremental, rcond)


def lux_objective(A, B, Z, s, lab, wl, flux, w, tau=np.inf):
    """Q = -2 ln L (labels and fluxes, up to a constant) + sum |z|^2 / tau^2; masked entries (weight 0) excluded."""
    var = np.where(w > 0, 1 / np.where(w > 0, w, 1.0) + s**2, 1.0)
    q_flux = np.where(w > 0, (flux - Z @ B.T)**2 / var + np.log(var), 0.0)
    return np.sum(wl * (lab - Z @ A.T)**2) + np.sum(q_flux) + np.sum(Z**2) / tau**2


def lux_scatter_step(B, Z, flux, w, s):
    """Pixel scatter by maximum likelihood at fixed B, Z (`fit_scatter` of Chapter 2), kept at each pixel only if it
    does not increase that pixel's part of Q."""
    err2 = np.where(w > 0, 1 / np.where(w > 0, w, 1.0), np.inf)
    resid = flux - Z @ B.T
    q = lambda s_: np.sum(np.where(w > 0, resid**2 / (err2 + s_**2) + np.log(np.where(w > 0, err2, 1.0) + s_**2), 0.0),
                          axis=0)
    s_new = fit_scatter(resid, err2)
    return np.where(q(s_new) <= q(s), s_new, s)


def lux_init(lab, P, n_flux, seed=0):
    """H25 Table 1, eq. 9: z_n = [1, (l_n - median) / ((p97.5 - p2.5) / 4), 0, ...], A and B uniform in [0, 1].
    Pivot and scale come from the labels that are present (NaN = missing); missing entries start at 0."""
    rng = np.random.default_rng(seed)
    q = np.nanpercentile(lab, [2.5, 50, 97.5], axis=0)
    x = np.nan_to_num((lab - q[1]) / ((q[2] - q[0]) / 4))
    Z = np.hstack([np.ones((len(lab), 1)), x, np.zeros((len(lab), P - lab.shape[1] - 1))])
    return rng.uniform(0, 1, (lab.shape[1], P)), rng.uniform(0, 1, (n_flux, P)), Z


def lux_train(lab, lab_err, flux, err, P=49, n_iter=5, tau=1000.0, incremental=True, seed=0):
    """Training as described in H25 §4.2. Stage 1: n_iter rounds of A-, B- and z-steps without scatter or prior.
    Stage 2 (once): scatter, then B with the new scatter, then z with the new B, scatter and the prior.
    Labels missing as NaN or with infinite error are ignored. Returns A, B, s, Z and Q after every step."""
    wl, l0 = lux_weights(lab, lab_err)
    w, f0 = lux_weights(flux, err)
    A, B, Z = lux_init(np.where(wl > 0, lab, np.nan), P, flux.shape[1], seed)
    s, history = np.zeros(flux.shape[1]), []
    for _ in range(n_iter):
        A = lux_a_step(Z, l0, wl, A, incremental)
        B = lux_b_step(Z, f0, w, B, incremental)
        Z = lux_z_step(A, B, l0, wl, f0, w, Z, np.inf, incremental)
        history.append(lux_objective(A, B, Z, s, l0, wl, f0, w))
    s = lux_scatter_step(B, Z, f0, w, s)
    wv = w / (1 + w * s**2)                                   # 1 / (sigma^2 + s^2), 0 where masked
    B = lux_b_step(Z, f0, wv, B, incremental)
    Z = lux_z_step(A, B, l0, wl, f0, wv, Z, tau, incremental)
    history.append(lux_objective(A, B, Z, s, l0, wl, f0, w, tau))
    return A, B, s, Z, np.array(history)


def lux_test(flux, err, A, B, s, rcond=1e-12):
    """Test step (H25 eq. 10): latents of each star from its spectrum alone (no prior), labels = A z.
    Latent directions the spectrum does not constrain (eigenvalues of B^T W B below rcond times the largest) are left at
    zero; the labels do not depend on them only if A has no component along them, which is checked and returned.
    Returns labels (N, M), their formal covariance (N, M, M), and for each star the condition number of the retained
    part of B^T W B and the largest |A v| over the discarded directions v (0 if none)."""
    w, f0 = lux_weights(flux, err)
    wv = w / (1 + w * s**2)
    P = B.shape[1]
    G = (wv @ (B[:, :, None] * B[:, None, :]).reshape(-1, P * P)).reshape(-1, P, P)
    vals, vecs = np.linalg.eigh(G)
    keep = vals > rcond * vals[:, -1:]
    Ginv = np.einsum('npq,nq,nrq->npr', vecs, np.where(keep, 1 / np.where(keep, vals, 1.0), 0.0), vecs)
    z = np.einsum('npq,nq->np', Ginv, (wv * f0) @ B)
    cond = vals[:, -1] / np.min(np.where(keep, vals, np.inf), axis=1)
    leak = np.max(np.where(~keep[:, None, :], np.abs(np.einsum('mp,npq->nmq', A, vecs)), 0.0), axis=(1, 2))
    return z @ A.T, np.einsum('mp,npq,lq->nml', A, Ginv, A), cond, leak
