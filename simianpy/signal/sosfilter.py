import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from copy import copy

from simianpy.signal.filter import Filter


class sosFilter(Filter):
    def __init__(
        self,
        filter_type,
        filter_order,
        freq_bounds,
        sampling_frequency,
    ):
        self.apply_fun = scipy.signal.sosfiltfilt
        self._filter = scipy.signal.butter(
            N=filter_order,
            Wn=freq_bounds,
            btype=filter_type,
            fs=sampling_frequency,
            output="sos",
        )
        self.filter_type = filter_type
        self.filter_order = filter_order
        self.freq_bounds = freq_bounds
        self.sampling_frequency = sampling_frequency

    def __call__(self, x, axis=-1):
        return self.apply_fun(self._filter, x, axis=axis)

    def plot(self, ax=None, semilogx=False, n_points=512):
        if ax is None:
            _, ax = plt.subplots()
        w, h = scipy.signal.sosfreqz(self._filter, fs=self.sampling_frequency, worN=n_points)

        if semilogx:
            ax.semilogx(w, 20 * np.log10(abs(h)))
        else:
            ax.plot(w, 20 * np.log10(abs(h)))

        ax.set_title("filter frequency response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [dB]")

    def __str__(self):
        if self.filter_type == 'compound':
            return f"Compound filter with sampling frequency {self.sampling_frequency} Hz"
        else:
            return super().__str__()

    def __add__(self, otherFilter):
        if self.sampling_frequency != otherFilter.sampling_frequency:
            raise ValueError("Sampling frequencies must be the same")
        new_filter = copy(self)
        new_filter.filter_type = 'compound'
        self.filter_order = None
        self.freq_bounds = None
        new_filter._filter = np.concatenate(
            [self._filter, otherFilter._filter], axis=0
        )
        return new_filter