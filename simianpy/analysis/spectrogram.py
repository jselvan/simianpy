from pathlib import Path

import numpy as np
import pandas as pd
import wavio


class Spectrogram:
    DEFAULT_UNITS = {"frequency": "kHz", "time": "ms", "power": "dB"}

    def __init__(self, data, sampling_rate, method="scipy.signal", method_kwargs={}):
        self.data = data
        self.sampling_rate = sampling_rate
        self.N = len(data)
        self._spectrogram = None
        self._compute(method, method_kwargs)

    @classmethod
    def from_wav(cls, wav_file, **kwargs):
        wav_file = Path(wav_file)
        wave = wavio.read(wav_file.as_posix())
        data = wave.data.squeeze()
        sampling_rate = wave.rate
        return cls(data, sampling_rate, **kwargs)

    def _compute(self, method, method_kwargs):
        if method == "scipy.signal":
            from scipy.signal import spectrogram

            f, t, s = spectrogram(self.data, fs=self.sampling_rate, scaling="spectrum", **method_kwargs)
        elif method == "scipy.signal.ShortTimeFFT":
            from scipy.signal import ShortTimeFFT
            from scipy.signal.windows import gaussian
            g_std = method_kwargs.get("std", 12)  # standard deviation for Gaussian window in samples
            g_mean = method_kwargs.get("mean", 50)  # mean for Gaussian window in samples
            hop = method_kwargs.get("hop", 2)  # hop size in samples
            mfft = method_kwargs.get("mfft", 800)  # number of points for FFT
            win = gaussian(M=g_mean, std=g_std, sym=True)  # symmetric Gaussian wind.
            SFT = ShortTimeFFT(
                win, hop=hop, fs=self.sampling_rate, mfft=mfft, scale_to="psd"
            )
            Sx2 = SFT.spectrogram(self.data)
            s = 10 * np.log10(Sx2)
            f = SFT.f
            t = SFT.t(self.data.size)
        f /= 1e3
        t *= 1e3
        self.frequency = f
        self.time = t
        self.power = s

    @property
    def spectrogram(self):
        if self._spectrogram is None:
            spectrogram = pd.DataFrame(
                self.power,
                index=pd.Index(self.frequency, name="frequency"),
                columns=pd.Index(self.time, name="time"),
            )
            self._spectrogram = spectrogram.stack().rename("power")
        return self._spectrogram

    def plot(self, ax=None, **kwargs):
        from simianpy.plotting import Image

        kwargs["im_kwargs"] = im_kwargs = kwargs.get("im_kwargs", {})
        if "aspect" not in im_kwargs:
            kwargs["im_kwargs"]["aspect"] = "auto"

        im = Image(
            x="time", y="frequency", z="power", data=self.spectrogram, ax=ax, **kwargs
        )

        return im
