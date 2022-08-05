import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

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

    def __call__(self, x, axis):
        return self.apply_fun(self._filter, x, axis=axis)

    def plot(self, ax=None, semilogx=False):
        if ax is None:
            _, ax = plt.subplots()
        w, h = scipy.signal.sosfreqz(self._filter, fs=self.sampling_frequency)

        if semilogx:
            ax.semilogx(w, 20 * np.log10(abs(h)))
        else:
            ax.plot(w, 20 * np.log10(abs(h)))

        ax.set_title("filter frequency response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [dB]")
