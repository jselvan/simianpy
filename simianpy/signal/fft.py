from simianpy.plotting.util import get_ax

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq

#TODO: add cupy support
class FFT:
    """Helper class for performing a FFT on time series data 

    Parameters
    ----------
    data: numpy array-like, pandas DataFrame or Series
        data to be transformed
    sampling_rate: int or float
        sampling rate in Hz of the input data
    
    Attributes
    ----------
    length
    power
    freqs

    Methods
    -------
    plot(ax=None,logx=False,logy=False)
        plots the FFT transform (only the real component)
    """
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate
    @property
    def nyquist(self):
        return self.sampling_rate / 2
    @property
    def length(self):
        return len(self.data)
    @property
    def power(self):
        return pd.Series(fft(self.data, self.length), index=self.freqs).sort_index()
    @property
    def freqs(self):
        return fftfreq(self.length, 1/self.sampling_rate)
    def plot(self, ax=None, logx=False, logy=False):
        ax = get_ax(ax)
        self.power.abs().plot(ax=ax)
        ax.set_xlim([0, self.nyquist])
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        return ax

def FFTpower(data, sampling_rate, real=True):
    power = FFT(data, sampling_rate).power
    if real:
        return power.loc[slice(0,None)]
    return power
