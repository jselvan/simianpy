from simianpy.analysis.lfp.phase import xr_compute_phase
import numpy as np
import xarray as xr

def test_xr_compute_phase_basic():
    """
    Test xr_compute_phase on a simple sinusoidal signal.
    """

    sampling_rate = 1000.0  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    freq = 5.0  # Hz
    signal = np.sin(2 * np.pi * freq * t)

    data = xr.DataArray(
        signal,
        dims=("time",),
        coords={"time": t},
    )

    phase_da = xr_compute_phase(
        data,
        time_dim="time",
        sampling_rate=sampling_rate,
    )

    # ---- assertions ----

    # Correct dims
    assert phase_da.dims == ("freq", "time")

    # Check that the frequency axis contains the expected frequency
    freqs = phase_da.freq.values
    assert np.any(np.isclose(freqs, freq, atol=0.5)), "Expected frequency not found in freq axis"