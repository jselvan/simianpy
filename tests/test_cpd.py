import numpy as np
import xarray as xr

from simianpy.analysis.stats.xarray.cpd import xr_compute_cpd

def test_compute_cpd_single_target_basic():
    """
    CPD should be ~1 for the true predictor and ~0 for a noise predictor.
    """

    rng = np.random.default_rng(0)

    n_obs = 200

    # True signal
    x_true = rng.standard_normal(n_obs)
    x_noise = rng.standard_normal(n_obs)

    # Response depends ONLY on x_true
    y = 3.0 * x_true + 0.01 * rng.standard_normal(n_obs)

    X = xr.DataArray(
        np.stack([x_true, x_noise], axis=1),
        dims=("obs", "predictor"),
        coords={"predictor": ["true", "noise"]},
    )

    Y = xr.DataArray(
        y,
        dims=("obs",),
    )

    CPD = xr_compute_cpd(
        X,
        Y,
        obs_dim="obs",
        predictor_dim="predictor",
    )

    # ---- assertions ----

    # Correct dims
    assert CPD.dims == ("predictor",)

    # True predictor explains almost all variance
    assert CPD.sel(predictor="true") > 0.95

    # Noise predictor explains almost nothing
    assert CPD.sel(predictor="noise") < 0.05

def test_compute_cpd_multiple_targets_vectorized():
    rng = np.random.default_rng(1)

    n_obs = 300
    n_targets = 4

    x1 = rng.standard_normal(n_obs)
    x2 = rng.standard_normal(n_obs)

    Y = np.stack(
        [
            2.0 * x1 + 0.01 * rng.standard_normal(n_obs),
            -1.5 * x2 + 0.01 * rng.standard_normal(n_obs),
            0.01 * rng.standard_normal(n_obs),
            3.0 * x1 + 0.5 * x2 + 0.01 * rng.standard_normal(n_obs),
        ],
        axis=1,
    )

    X = xr.DataArray(
        np.stack([x1, x2], axis=1),
        dims=("obs", "predictor"),
        coords={"predictor": ["x1", "x2"]},
    )

    Y = xr.DataArray(
        Y,
        dims=("obs", "target"),
        coords={"target": ["t1", "t2", "t3", "t4"]},
    )

    CPD = xr_compute_cpd(
        X,
        Y,
        obs_dim="obs",
        predictor_dim="predictor",
    )

    # CPD should preserve the target dimension
    assert CPD.dims == ("target", "predictor")

    # Predictor relevance checks
    assert CPD.sel(target="t1", predictor="x1") > 0.9
    assert CPD.sel(target="t1", predictor="x2") < 0.1

    assert CPD.sel(target="t2", predictor="x2") > 0.9
    assert CPD.sel(target="t2", predictor="x1") < 0.1
