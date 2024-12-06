import numpy as np
from scipy.stats import circmean, norm


def circ_r(x, w=None, axis: int=-1) -> float:
    """Computes the mean resultant length of a circular variable x weighted by w

    Parameters
    ----------
    x : array_like
        Circular variable
    w : array_like, optional
        Weights

    Returns
    -------
    r : float
        Mean resultant length
        0 = uniform distribution
        1 = perfectly aligned
    """
    if w is None:
        w = np.ones_like(x)

    r = (w * np.exp(1j * x)).sum(axis)
    r = np.abs(r) / w.sum(axis)
    return r


def circ_corrcc(x, y, axis: int=-1, correction_uniform: bool=False) -> tuple[float, float]:
    """
    Compute circular correlation coefficient between two circular variables.

    Parameters
    ----------
    x, y : array_like
        Circular variables to correlate.
    axis : int, optional, default = -1
        Axis along which to compute the correlation.
    correction_uniform : bool, optional
        If True, apply correction (not implemented).
        If False (default), no correction is applied.

    Returns
    -------
    r : array-like
        Circular correlation coefficient.

    p : array-like
        2-tailed p-value.
    """
    if correction_uniform:
        raise NotImplementedError("Correction for uniform distribution not implemented.")

    x, y = np.asarray(x), np.asarray(y)
    if x.shape[axis] != y.shape[axis]:
        raise ValueError(
            f"x and y must have the same size along `axis`={axis}. Shapes are {x.shape} and {y.shape}."
        )
    n = x.shape[axis]
    x_sin = np.sin(x - np.expand_dims(circmean(x, high=np.pi, low=-np.pi, axis=axis), axis))
    x_sin_sq = x_sin**2
    y_sin = np.sin(y - np.expand_dims(circmean(y, high=np.pi, low=-np.pi, axis=axis), axis))
    y_sin_sq = y_sin**2

    num = (x_sin * y_sin).sum(axis=axis)
    den = (x_sin_sq.sum(axis=axis) * y_sin_sq.sum(axis=axis)) ** 0.5
    rho = num / den

    tval = (
        np.sqrt(
            n
            * x_sin_sq.mean(axis=axis)
            * y_sin_sq.mean(axis=axis)
            / (x_sin_sq * y_sin_sq).mean(axis=axis)
        )
        * rho
    )

    pval = 2 * (1 - norm.cdf(np.abs(tval)))
    return rho, pval
