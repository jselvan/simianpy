from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


# TODO: add cupy support
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
    sampling_rate
    _filter

    Methods
    -------
    __call__(input_data,axis=-1)
        filter input data
    plot(ax=None,semilogx=False)
        plots the frequency response of filter
    check(input_data,axis=1,ax=None)
        plots the raw and filtered data
    """

    def __str__(self):
        return f"Filter - Type: '{self.filter_type}'; Order: {self.filter_order}; Cutoff(s): {self.freq_bounds}; Fs: {self.sampling_frequency}"

    def __init__(
        self,
        filter_type,
        filter_order,
        freq_bounds,
        sampling_frequency,
        filter_fun=None,
        apply_fun=None,
    ):
        if apply_fun is None:
            self.apply_fun = scipy.signal.filtfilt
        else:
            self.apply_fun = apply_fun

        freq_bounds = np.asarray(freq_bounds)
        freq_bounds_norm = freq_bounds / sampling_frequency / 2
        self.freq_bounds = freq_bounds

        if filter_fun is None:
            filter_fun = scipy.signal.butter
        self.filter_order = filter_order
        self.filter_type = filter_type
        self._filter = filter_fun(
            N=filter_order, Wn=freq_bounds, btype=filter_type, fs=sampling_frequency
        )
        self.sampling_frequency = sampling_frequency

    def __call__(self, input_data, axis=-1):
        """Filter input data"""
        return self.apply_fun(*self._filter, input_data, axis=axis)

    def check(self, input_data, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        filtered_data = self.__call__(input_data)
        ax.plot(input_data, label="original", c="k")
        ax.plot(filtered_data, label="filtered", c="r")
        ax.legend()

    def plot(self, ax=None, semilogx=False):
        if ax is None:
            _, ax = plt.subplots()
        b, a = self._filter
        w, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)

        if semilogx:
            ax.semilogx(w, 20 * np.log10(abs(h)))
        else:
            ax.plot(w, 20 * np.log10(abs(h)))

        ax.set_title("filter frequency response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [dB]")
