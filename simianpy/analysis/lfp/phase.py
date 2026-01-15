from ssqueezepy import Wavelet, cwt
from ssqueezepy.experimental import scale_to_freq
import numpy as np
import xarray as xr
from typing import Tuple

def compute_phase(data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the instantaneous phase of the input signal using Continuous Wavelet Transform (CWT).

    Parameters:
    - data: 1D numpy array of the input signal.
    - sampling_rate: Sampling rate of the input signal in Hz.

    Returns:
    - phase: 1D numpy array of the instantaneous phase in radians.
    - freqs: 1D numpy array of the corresponding frequencies in Hz.
    """

    # Perform Continuous Wavelet Transform
    N = data.size
    wavelet = Wavelet('morlet')
    coef, scales = cwt(data, wavelet) #type: ignore
    freqs = scale_to_freq(scales, wavelet, N, int(sampling_rate))
    # Compute instantaneous phase
    phase = np.angle(coef) #type: ignore
    return phase, freqs

    # return phase, freqs

def xr_compute_phase(
    data: xr.DataArray,
    time_dim: str,
    sampling_rate: float
) -> xr.DataArray:
    """
    Compute the instantaneous phase of the input xarray DataArray along the specified time dimension.

    Parameters:
    - data: xarray DataArray containing the input signal.
    - time_dim: Name of the time dimension in the DataArray.
    - sampling_rate: Sampling rate of the input signal in Hz.

    Returns:
    - phase_da: xarray DataArray of the instantaneous phase in radians.
    """
    def phase_func(signal: np.ndarray) -> np.ndarray:
        global freqs
        phase, freqs = compute_phase(signal, sampling_rate)
        return phase
    phase_da = xr.apply_ufunc(
        phase_func,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[['freq', time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
    )
    phase_da = phase_da.assign_coords(freq=freqs)

    return phase_da