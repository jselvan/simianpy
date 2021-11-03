from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats

def ISI(spike_train):
    isi = np.diff(spike_train)
    return isi

def CV(isi):
    isi = np.asarray(isi)
    cv = isi.std()/isi.mean()
    return cv

def LV(isi):
    isi = np.asarray(isi)
    lv = (3 * (isi.size-2)) * (((isi[:-1] - isi[1:])/(isi[:-1]+isi[1:]))**2).sum()
    return lv

def gamma(isi):
    fit = namedtuple('gamma_mle_fit',['alpha','loc','beta'])(*scipy.stats.gamma.fit(isi))
    return fit

def ccg(spike_train_a, spike_train_b, bin_size=1e3):
    """Compute cross-correlogram for two spike trains

    Parameters
    ----------
    spike_train_a, spike_train_b : array-like
        spike times in ms
    bin_size : numeric, optional
        bin width in ms, by default 1e3

    Returns
    -------
    pd.Series
        cross correlation histogram
    """
    import quantities as pq
    from elephant.conversion import BinnedSpikeTrain
    from elephant.spike_train_correlation import cross_correlation_histogram
    bin_size = bin_size*pq.ms
    xcorr, lags = cross_correlation_histogram(
        BinnedSpikeTrain(spike_train_a,bin_size=bin_size),
        BinnedSpikeTrain(spike_train_b,bin_size=bin_size)
    )
    return pd.Series(xcorr.as_array(), index=lags)

def acg(spike_train, bin_size=1e3):
    """Compute auto-correlogram for a spike train

    Parameters
    ----------
    spike_train : array-like
        spike times in ms
    bin_size : numeric, optional
        bin width in ms, by default 1e3

    Returns
    -------
    pd.Series
        auto correlation histogram
    """
    #TODO: Remove center peak
    acg_data = ccg(spike_train, spike_train, bin_size=1e3)
    return acg_data

