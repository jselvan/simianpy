from collections import namedtuple

import numpy as np
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

def ccg(spike_train_a, spike_train_b):
    ccg_data = np.correlate(spike_train_a, spike_train_b, 'full')
    return ccg_data

def acg(spike_train):
    #TODO: Remove center peak
    acg_data = ccg(spike_train, spike_train)
    return acg_data