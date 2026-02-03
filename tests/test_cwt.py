from simianpy.analysis.lfp.decomposition import xr_cwt

def test_xr_cwt():
    import xarray as xr
    import numpy as np

    n_channels = 2
    n_timepoints = 1000
    sampling_rate = 1000.0

    times = np.arange(n_timepoints) / sampling_rate
    data = xr.DataArray(
        np.random.randn(n_channels, n_timepoints).astype(np.float32),
        dims=("channel", "time"),
        coords={"channel": np.arange(n_channels), "time": times},
    )

    freq_range = (5.0, 100.0)
    n_freqs = 10

    cwt_result = xr_cwt(
        data,
        freq_range=freq_range,
        n_freqs=n_freqs,
        time_dim="time",
        sampling_rate=sampling_rate,
    )

    assert "power" in cwt_result.data_vars
    assert "phase" in cwt_result.data_vars
    assert cwt_result.power.shape == (n_channels, n_freqs, n_timepoints)
    assert cwt_result.phase.shape == (n_channels, n_freqs, n_timepoints)
    assert cwt_result.power.dtype == np.float32
    assert cwt_result.phase.dtype == np.float32