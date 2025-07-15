import numpy as np
import xarray as xr

def entropy(p, axis=None):
    H = - p * np.log2(p)
    return H.sum(axis)

def mutual_information_p(px, py, pxy):
    I = pxy * np.log2(pxy/(px*py))
    return I.sum()

def mutual_information(x, y):
    x_states, x_digitized = np.unique(x, return_inverse=True)
    y_states, y_digitized = np.unique(y, return_inverse=True)

    counts, _ = np.histogramdd(
        np.stack([x_digitized, y_digitized]), 
        bins=(np.arange(x_digitized.size+1), np.arange(y_states.size+1))
    )
    px = counts / counts.sum(axis=0, keepdims=True)
    py = counts / counts.sum(axis=1, keepdims=True)
    pxy = counts / counts.sum()
    Ixy = mutual_information_p(px, py, pxy)
    # Hx = entropy(px)
    # Hxy = entropy(pxy)
    # # $H(Y|X) = H(X,Y) - H(X)$
    # Hy_x = Hxy - Hx
    return Ixy


def mutual_information_x_digitized(y, x_digitized, n_x_states):
    y_states, y_digitized = np.unique(y, return_inverse=True)

    counts, _ = np.histogramdd(
        np.stack([x_digitized, y_digitized]), 
        bins=(np.arange(x_digitized.size+1), np.arange(y_states.size+1))
    )
    px = counts / counts.sum(axis=0, keepdims=True)
    py = counts / counts.sum(axis=1, keepdims=True)
    pxy = counts / counts.sum()
    Ixy = mutual_information_p(px, py, pxy)
    return Ixy

def xarray_mutual_information(data, x):
    """Compute mutual information on xarray object

    Parameters
    ----------
    data : xarray.DataArray
        must contain y counts
    x: str
        coordinate dimension on data corresponding to x
    """
    x_states, x_digitized = np.unique(data[x], return_inverse=True)
    dim, = data[x].dims
    n_x_states = x_states.size
    xr.apply_ufunc(
        mutual_information_x_digitized,
        data, 
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        kwargs={"n_x_states": n_x_states, "x_digitized": x_digitized},
    )

def transfer_entropy(xp, yp, yf):
    pass

# def partial_information(X, y):
