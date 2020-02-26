import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq

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
        #TODO: rewrite so that this function is applied to a 1D vector only? (use pd.DataFrame.apply for multiple variables)
        columns = None
        if isinstance(self.data, pd.DataFrame):
            data = self.data.values.T
            columns = np.asarray(self.data.columns)
        elif isinstance(self.data, pd.Series):
            data = self.data.values
            columns = np.asarray(self.data.name)
        else:
            data = np.asarray(self.data)

        if columns is None:
            if data.ndim == 1:
                columns = [0]
            elif data.ndim == 2:
                columns = np.arange(data.shape[0])
            else:
                raise ValueError(f'data must have 1 or 2 dimensions not {data.ndim}')
                
        return pd.DataFrame(fft(data, self.length).T, index = self.freqs, columns = columns).sort_index()
    @property
    def freqs(self):
        return fftfreq(self.length, 1/self.sampling_rate)
    def plot(self, ax = None, logx = False, logy = False):
        if ax is None:
            _, ax = plt.subplots()
        self.power.abs().plot(ax=ax)
        ax.set_xlim([0, self.nyquist])
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

def FFTpower(data, sampling_rate, real = True):
    power = FFT(data, sampling_rate).power
    if real:
        return power.loc[slice(0,None)]
    return power
