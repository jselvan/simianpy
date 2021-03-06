import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
#TODO: add cupy support
class Filter:
    """Class for designing digital frequency filters. 
    
    This has only been tested with scipy.signal.butter & scipy.signal.filtfilt.
    Instances of this class are callable on data that is array-like. Use plot method to visualize the filter

    Parameters
    ----------
    filter_type: {'low','high','bandpass'}
    filter_order: int
    freq_bounds: int or array-like of int
        the cutoff frequenc(y/ies) for the filter
    sampling_frequency: int or float
        sampling frequency in Hz
    filter_fun: fun or None, optional, default: None
        filter function - if None: scipy.signal.butter
    apply_fun: fun or None, optional, default: None
        function used to apply filter - if None: scipy.signal.filtfilt

    Attributes
    ----------
    nyquist_frequency: float; read only
        returns nyquist frequency
    freq_bounds_norm: float; read only
        returns normalized frequency boundaries

    Methods
    -------
    __call__(input_data,axis=1)
        filter input data
    plot(ax=None,semilogx=False)
        plots the frequency response of filter
    check(input_data,axis=1,ax=None)
        plots the raw and filtered data
    """
    def __str__(self):
        return f"Filter - Type: '{self.filter_type}'; Order: {self.filter_order}; Cutoff(s): {self.freq_bounds}; Fs: {self.sampling_frequency}"

    def __init__(self, filter_type, filter_order, freq_bounds, sampling_frequency, filter_fun = None, apply_fun = None):
        if filter_fun is None:
            self.filter_fun = scipy.signal.butter
        else:
            self.filter_fun = filter_fun
        
        if apply_fun is None:
            self.apply_fun = scipy.signal.filtfilt
        else:
            self.apply_fun = apply_fun

        self.filter_type = filter_type
        self.filter_order = filter_order
        self.sampling_frequency = sampling_frequency
        self.freq_bounds = np.asarray(freq_bounds)
    
    @property
    def _filter(self):
        """Returns (b,a) corresponding to numerator and denominator polynomials respectively"""
        return self.filter_fun(N = self.filter_order, Wn = self.freq_bounds_norm, btype = self.filter_type)

    @property
    def nyquist_frequency(self):
        return self.sampling_frequency/2
    
    @property
    def freq_bounds_norm(self):
        return self.freq_bounds/self.nyquist_frequency
    
    def __call__(self, input_data):
        """Filter input data"""
        return self.apply_fun(*self._filter, input_data)

    def check(self, input_data, ax = None):
        if ax is None:
            _, ax = plt.subplots()

        filtered_data = self.__call__(input_data)
        ax.plot(input_data, label = 'original', c = 'k')
        ax.plot(filtered_data, label = 'filtered', c = 'r')
        ax.legend()
    
    def plot(self, ax = None, semilogx = False):
        if ax is None:
            _, ax = plt.subplots()
        b, a = self._filter
        w, h = scipy.signal.freqz(b, a, fs = self.sampling_frequency)

        if semilogx:
            ax.semilogx(w, 20 * np.log10(abs(h)))
        else:
            ax.plot(w, 20 * np.log10(abs(h)))

        ax.set_title('filter frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')