# --- Spectra and continuum normalisation (as in Chapters 0 and 3) ----------------------------------------
NPIX = 8575
wave = 10**(4.179 + 6e-6 * np.arange(NPIX))               # apStar vacuum wavelength grid [Angstrom]
CHIPS = [(15150, 15800), (15865, 16425), (16480, 16950)]  # the three detectors [A], inside the edges of Chapter 0
BAD_BITS = sum(1 << b for b in [0, 1, 2, 3, 4, 5, 6, 7, 12])   # pixels given zero weight (Chapter 0)


def prepare(sp, rows=slice(None)):
    """flux and err (float64) of the spectra file sp; err = inf where there are no data, a BAD_BITS flag, or
    outside CHIPS (the detector edges, where only some spectra have data)."""
    flux = sp['flux'][rows].astype(float)
    err = sp['err'][rows].astype(float)
    on_chip = np.any([(wave > lo) & (wave < hi) for lo, hi in CHIPS], axis=0)   # pixels at the detector edges and
    bad = ((sp['mask'][rows] & BAD_BITS) > 0) | ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0) | ~on_chip  # gaps: no weight
    return np.where(bad, 0.0, flux), np.where(bad, np.inf, err)


def cheb_normalise(flux, err, cont, order=2):
    """Divide by a Chebyshev polynomial of the given order fitted, per detector, to the continuum pixels `cont`
    (boolean, NPIX) with weights 1/err^2 (N15 sect. 2.3 and 5.3). Returns normalised flux and error;
    pixels outside the detectors get err = inf."""
    fn, en = np.zeros_like(flux), np.full_like(err, np.inf)
    for lo, hi in CHIPS:
        c = (wave > lo) & (wave < hi)
        t = (wave[c] - (lo + hi) / 2) / ((hi - lo) / 2)          # wavelength mapped to [-1, 1]
        B = np.polynomial.chebyshev.chebvander(t, order)         # (pixels, order+1) basis
        use = cont[c]
        w = 1 / err[:, c][:, use]**2                             # (N, continuum pixels); 0 for flagged pixels
        # one small weighted least-squares problem per star: (B^T W B) a = B^T W f
        BtWB = np.einsum('pi,np,pj->nij', B[use], w, B[use])
        BtWf = np.einsum('pi,np->ni', B[use], w * flux[:, c][:, use])
        a = np.linalg.solve(BtWB, BtWf[:, :, None])[:, :, 0]     # (N, order+1); b as a stack of column vectors
        cont_fit = a @ B.T                                       # (N, pixels of this detector)
        fn[:, c] = flux[:, c] / cont_fit
        en[:, c] = err[:, c] / cont_fit
    return fn, en
