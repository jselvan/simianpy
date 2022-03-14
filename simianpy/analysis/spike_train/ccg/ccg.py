import cupy as cp
import numpy as np
import pandas as pd
import scipy.stats

from simianpy.misc.cupy.view_as_windows import view_as_windows


def compute_ccg(raster_a, raster_b, lag=50):
    n_trials, n_bins = raster_a.shape
    pad = cp.zeros((n_trials, lag))
    raster_b = cp.concatenate([pad, raster_b, pad], axis=1)
    raster_b = view_as_windows(raster_b, (1, n_bins))[:, :, 0, :]
    raster_a = cp.expand_dims(raster_a, axis=1)
    ccg = cp.sum(raster_a * raster_b, (0, 2)) / cp.sqrt(
        cp.sum(raster_a, (0, 2)) * cp.sum(raster_b, (0, 2))
    )
    return ccg.get()


def jitter(raster, window):
    ntrials, nbins = raster.shape
    # if needed, pad to be divisible by window
    if nbins % window:
        pad = cp.zeros((ntrials, -nbins % window))
        raster = cp.concatenate([raster, pad], axis=1)
    nbins_rounded = raster.shape[1]
    n_jitter_bins = nbins_rounded // window

    # get psth
    psth = raster.mean(axis=0)

    # bin over window and sum
    raster_binned = cp.reshape(raster, (ntrials, window, n_jitter_bins)).sum(axis=1)
    psth_binned = cp.reshape(psth, (window, n_jitter_bins)).sum(axis=0)

    # determine correction
    correction = raster_binned / cp.expand_dims(psth_binned, 0)
    correction = cp.tile(cp.expand_dims(correction, 1), [1, window, 1])
    correction = cp.reshape(correction, (ntrials, nbins_rounded))

    # apply correction
    raster_jittered = cp.expand_dims(psth, 0) * correction

    # trim off padding
    raster_jittered = raster_jittered[:, :nbins]
    raster_jittered[cp.isnan(raster_jittered)] = 0
    return raster_jittered


def jitter_corrected_ccg(raster_a, raster_b, lag, window, peak_params=None):
    jittered_raster_a = jitter(raster_a, window)
    jittered_raster_b = jitter(raster_b, window)

    # compute corrected ccg
    lags = np.arange(-lag, lag + 1)
    uncorrected_ccg = compute_ccg(raster_a, raster_b, lag)
    jittered_ccg = compute_ccg(jittered_raster_a, jittered_raster_b, lag)
    corrected_ccg = uncorrected_ccg - jittered_ccg

    output = dict(ccg=pd.Series(corrected_ccg, index=lags), lags=lags)

    if peak_params is not None:
        abs_lag = np.abs(lag)
        peak_range_mask = abs_lag < peak_params["peak_range"]
        peak_range_lags = lags[peak_range_mask]
        null_range_lower, null_range_upper = peak_params["null_range"]
        null_range_mask = (null_range_lower < abs_lag) & (abs_lag < null_range_upper)
        zscores = scipy.stats.zmap(
            corrected_ccg[peak_range_mask], corrected_ccg[null_range_mask], axis=0
        )
        significant_lags = peak_range_lags[zscores > peak_params["threshold"]]
        output["significant_lags"] = significant_lags
    return output
