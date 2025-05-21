import numpy as np


def compute_covariogram(unita_hist, unitb_hist, unita_psth, unitb_psth, lags):
    n_trials, n_bins = unita_hist.shape
    sigma_terms = np.zeros((3, lags.size))
    SIGMATERMS = (("var", "var"), ("mean", "var"), ("var", "mean"))
    cross_corr = np.zeros(lags.size)
    shuffle_corrector = np.zeros(lags.size)
    for lagidx, lag in enumerate(lags):
        start, finish = max(-lag, 0), min(n_bins, n_bins - lag)
        left = np.arange(start, finish)
        right = left + lag
        cross_corr[lagidx] = np.sum(unita_hist[:, left] * unitb_hist[:, right])
        shuffle_corrector[lagidx] = np.sum(
            unita_psth["mean"][left] * unitb_psth["mean"][right]
        )
        for term_idx, (left_term, right_term) in enumerate(SIGMATERMS):
            sigma_terms[term_idx, lagidx] = np.sum(
                (unita_psth[left_term][left] * unitb_psth[right_term][right]) ** 2
            )
    covariogram = cross_corr - shuffle_corrector
    sigma = (sigma_terms.sum(axis=0) / n_trials) ** 0.5
    crit = 2 * sigma
    return covariogram, crit
