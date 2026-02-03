import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
from typing import Tuple

# =========================
# MORLET WAVELETS (FFT DOMAIN)
# =========================
def morlet_fft(freqs: ArrayLike, n_timepoints: int, sampling_rate: float, n_cycles: ArrayLike) -> np.ndarray:
    freqs = np.asarray(freqs)
    n_cycles = np.asarray(n_cycles)
    t = np.arange(-n_timepoints // 2, n_timepoints // 2) / sampling_rate
    n_freqs = freqs.size
    fft_w = np.zeros((n_freqs, n_timepoints), dtype=np.complex64)
    freqs_fft = np.fft.fftfreq(n_timepoints, d=1/sampling_rate)

    for i, (f, nc) in enumerate(zip(freqs, n_cycles)):
        sigma = nc / (2 * np.pi * f)
        w = np.exp(2j * np.pi * f * t) * np.exp(-t**2 / (2 * sigma**2))
        # normalize
        w /= np.sqrt(0.5) * np.linalg.norm(w.ravel())
        w = np.fft.ifftshift(w)
        fft_w[i] = np.fft.fft(w)
        fft_w[i, freqs_fft < 0] = 0
    return fft_w

# =========================
# FFT CONVOLUTION KERNEL
# =========================
def morlet_conv_fft(signal: np.ndarray, fft_wavelets: np.ndarray) -> np.ndarray:
    sig_fft = np.fft.fft(signal)
    return np.fft.ifft(sig_fft[None, :] * fft_wavelets, axis=-1)

def xr_cwt(
        data: xr.DataArray, 
        freq_range: Tuple[float, float], 
        n_freqs: int = 40,
        time_dim: str = "time",
        sampling_rate: float = 1000.0,
    ) -> xr.Dataset:
    freq_min, freq_max = freq_range
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_timepoints = data.sizes[time_dim]
    n_cycles = np.clip(freqs / 2, 3, 10)
    fft_wavelets = morlet_fft(freqs, n_timepoints, sampling_rate, n_cycles)
    fft_wavelets_da = xr.DataArray(
        fft_wavelets,
        dims=("freq", time_dim),
        coords={"freq": freqs},
    )
    tf = xr.apply_ufunc(
        morlet_conv_fft,
        data,
        fft_wavelets_da,
        input_core_dims=[[time_dim], ["freq", time_dim]],
        output_core_dims=[["freq", time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.complex64],
        dask_gufunc_kwargs={'output_sizes': {'freq': n_freqs}},
    )
    
    
    power = (tf.real**2 + tf.imag**2).astype(np.float32)

    # PSD normalization: convert power to power per Hz using factor n_cycles / freq
    # avoid division by zero for any zero-frequency entries
    scale = np.where(freqs == 0, 0.0, (n_cycles / freqs))
    scale_da = xr.DataArray(scale, dims=("freq",), coords={"freq": freqs})
    power = (power * scale_da).astype(np.float32)

    phase = xr.apply_ufunc(
        np.angle,
        tf,
        input_core_dims=[["freq", time_dim]],
        output_core_dims=[["freq", time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    return xr.Dataset(
        data_vars=dict(
            power=power,
            phase=phase,
        ),
        coords=tf.coords,
    )
