import numpy as np
import xarray as xr


def compute_cpd(X, y):
    """
    X: (obs, p)
    y: (obs,)
    """
    obs, p = X.shape

    # Full model
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat_full = X @ beta_full
    sse_full = np.sum((y - y_hat_full) ** 2)

    cpd = np.zeros(p)

    for j in range(p):
        X_red = np.delete(X, j, axis=1)
        beta_red, *_ = np.linalg.lstsq(X_red, y, rcond=None)
        y_hat_red = X_red @ beta_red
        sse_red = np.sum((y - y_hat_red) ** 2)

        cpd[j] = 1.0 - (sse_full / sse_red)

    return cpd

def xr_compute_cpd(
    X: xr.DataArray,
    Y: xr.DataArray,
    obs_dim: str,
    predictor_dim: str = "predictor",
    add_intercept: bool = False
):
    """
    Compute Coefficient of Partial Determination (CPD) using xarray + apply_ufunc.

    Parameters
    ----------
    X : xr.DataArray
        Design matrix with dims (..., obs_dim, predictor_dim)
    Y : xr.DataArray
        Response matrix with dims (..., obs_dim, target)
    obs_dim : str
        Name of observation/core dimension
    predictor_dim : str
        Name of predictor dimension

    Returns
    -------
    CPD : xr.DataArray
        CPD values with dims (..., predictor_dim, target)
    """
    if add_intercept:
        intercept = xr.DataArray(
            np.ones(X.sizes[obs_dim]),
            dims=[obs_dim],
            coords={obs_dim: X[obs_dim]},
            name="intercept"
        )
        X = xr.concat([intercept, X], dim=predictor_dim)
    CPD = xr.apply_ufunc(
        compute_cpd,
        X,
        Y,
        input_core_dims=[[obs_dim, predictor_dim], [obs_dim]],
        output_core_dims=[[predictor_dim]],
        vectorize=True,
        dask="parallelized",
    )
    CPD.name = "CPD"
    return CPD
