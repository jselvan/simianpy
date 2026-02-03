
import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import List
import xarray as xr

def xr_zscore(data, dim):
    mean = data.mean(dim)
    std = data.std(dim)
    z = (data - mean) / std
    z = z.fillna(0)
    return z

def get_design_matrix(
    data: pd.DataFrame | xr.DataArray, 
    regressors: List[str], 
    add_intercept=True
) -> xr.DataArray:
    X = xr.DataArray(
        [zscore(data[r].values) for r in regressors], 
        dims=['trialid', 'regressor'], 
        coords={'regressor': regressors, 'trialid': data.trialid}
    )
    if add_intercept:
        intercept = xr.DataArray(
            np.ones(X.sizes['trialid']),
            dims=['trialid'],
            coords={'trialid': X.trialid}
        )
        X = xr.concat([intercept, X], dim='regressor')
        X = X.assign_coords({'regressor': ['intercept'] + regressors})
    return X


def compute_cpd(X, Y):
    """
    X: (obs, p)
    Y: (obs, ...)
    """
    obs, p = X.shape
    y = Y.reshape(obs, -1)  # (obs, independent tests)

    # Full model
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat_full = X @ beta_full
    sse_full = np.sum((y - y_hat_full) ** 2)

    cpd = np.zeros((p, y.shape[1]), dtype=np.float32)

    for j in range(p):
        X_red = np.delete(X, j, axis=-1)
        beta_red, *_ = np.linalg.lstsq(X_red, y, rcond=None)
        y_hat_red = X_red @ beta_red
        sse_red = np.sum((y - y_hat_red) ** 2, axis=0)

        cpd[j, :] = 1.0 - (sse_full / sse_red)

    return cpd.reshape(p, *Y.shape[1:])

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

def xr_compute_cpd_formula(
    formula: str,
    data: xr.DataArray,
    obs_dim: str,
    predictor_dim: str = "predictor",
    n_iterations: int = 1000,
    n_workers: int = 8,
    show_dashboard: bool = False,
    permutation_chunk_size: int = 1,
) -> xr.DataArray | xr.Dataset:
    pred = data.coords[obs_dim].to_dataframe()
    from formulaic import model_matrix
    X = model_matrix(formula, data=pred)
    assert isinstance(X, pd.DataFrame)
    X.columns.name = predictor_dim
    X_da = X.stack().to_xarray()
    assert isinstance(X_da, xr.DataArray)

    CPD = xr_compute_cpd(
        X_da, 
        data, 
        obs_dim, 
        predictor_dim, 
        add_intercept=False
    )

    # add null permutation dimensions
    if n_iterations is not None and n_iterations > 1:
        null = []
        # add permutation dimension to design matrix
        X_da = X_da.expand_dims({'permutation': np.arange(n_iterations)})
        for iteridx in range(n_iterations):
            permidx = np.random.permutation(X_da[obs_dim].size)
            permda = X_da.isel({obs_dim: permidx, 'permutation': iteridx}).copy()
            permda.coords[obs_dim] = X_da[obs_dim]
            null.append(permda)
        Xnull_da = xr.concat(null, dim='permutation')
        assert isinstance(Xnull_da, xr.DataArray)

        from dask.distributed import LocalCluster, Client
        client = Client(LocalCluster(n_workers=n_workers, threads_per_worker=1))
        with client:
            if show_dashboard:
                try:
                    import webbrowser
                    webbrowser.open(client.dashboard_link)
                except:
                    pass
            CPD_null = xr_compute_cpd(
                Xnull_da.chunk({'permutation': permutation_chunk_size}), 
                data, 
                obs_dim, 
                predictor_dim, 
                add_intercept=False,
            )
            CPD_score = (CPD_null < CPD.expand_dims({'permutation': np.arange(n_iterations)})).sum(dim='permutation') / n_iterations
            CPD_score = CPD_score.compute()
        return xr.Dataset({'CPD': CPD, 'percentile': CPD_score})
    return CPD