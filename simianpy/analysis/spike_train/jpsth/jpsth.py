import numpy as np

from .compute_coincidence_histogram import compute_coincidence_histogram
from .compute_covariogram import compute_covariogram
from .compute_crosscorr import compute_crosscorr
from .compute_psth_stats import compute_psth_stats
from .find_longest_significant_span import find_longest_significant_span
from .normalize_jpsth import normalize_jpsth


def jpsth(t, unita_hist, unitb_hist, coincidence_width=10, lag=50):
    """Compute JPSTH for two units

    Parameters
    ----------
    unita_hist, unitb_hist : np.ndarray
        Histograms of unita and unitb
    coincidence_width : int
    lag : int
    """
    coincidence_widths = np.arange(-coincidence_width, coincidence_width + 1, dtype=int)
    lags = np.arange(-lag, lag + 1, dtype=int)

    unita_stats = compute_psth_stats(unita_hist)
    unitb_stats = compute_psth_stats(unitb_hist)

    raw_jpsth = np.mean(
        np.expand_dims(unita_hist, 2) * np.expand_dims(unitb_hist, 1), axis=0
    )
    normalized_jpsth = normalize_jpsth(raw_jpsth, unita_stats, unitb_stats)
    xcorr_hist = compute_crosscorr(normalized_jpsth, lags)
    pstch = compute_coincidence_histogram(normalized_jpsth, coincidence_widths)
    covariogram, crit = compute_covariogram(
        unita_hist, unitb_hist, unita_stats, unitb_stats, lags
    )
    peak_span = find_longest_significant_span(covariogram, crit)
    trough_span = find_longest_significant_span(-covariogram, -crit)

    return {
        "t": t,
        "unita": unita_stats,
        "unitb": unitb_stats,
        "raw_jpsth": raw_jpsth,
        "normalized_jpsth": normalized_jpsth,
        "xcorr_hist": xcorr_hist,
        "coincidence_histogram": pstch,
        "coincidence_widths": coincidence_widths,
        "lags": lags,
        "covariogram": covariogram,
        "crit": crit,
        "peak_span": peak_span,
        "trough_span": trough_span,
    }
