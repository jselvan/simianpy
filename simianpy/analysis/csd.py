import numpy as np

def CSD(depth, data, slices=None, sigma=-0.3):
    """Compute a 1-D current source density

    Uses a 5 point approximation of second spatial derivative

    Parameters
    ----------
    depth : array-like numeric
        depth in microns. size (n_channels, 1)
    data : array-like numeric
        LFP data. size (n_channels, n_samples)
    slices : array-like int, optional
        if not None, data is averaged in groups as defined by zip(slices[:-1], slices[1:]), by default None
    sigma : float, optional
        by default -0.3 see [2]

    Returns
    -------
    depth_, data_: array-like numeric
        returns updated depth and computed CSD
        If slices is none, size (n_channels-4, n_samples)
        else size (n_slices-4, n_samples)

    Raises
    ------
    ValueError
        Raises a ValueError if channels/slices are not evenly spaced

    References
    ----------
    [1] Freeman and Nicholson 1975
    [2] Petterson et al 2006
    """
    if slices is not None:
        depth = np.stack([depth[l:r].mean(axis=0) for l,r in zip(slices[:-1],slices[1:])])
        data = np.stack([data[l:r, :].mean(axis=0) for l,r in zip(slices[:-1],slices[1:])])

    diff = np.diff(depth)
    if np.unique(diff).size == 1:
        h, = np.unique(diff)
    else:
        raise ValueError

    data_ = sigma * ( (data[4:]-data[2:-2]+data[:-4]) / 4*(h**2) )
    depth_ = depth[2:-2]
    return depth_, data_