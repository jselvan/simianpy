import matplotlib.pyplot as plt
import numpy as np

from simianpy.plotting.util import get_ax, draw_lines

range_ = range

def Histogram(
    data,
    bins=10,
    range=None,
    density=False,
    proportion=False,
    multiplier=1,
    invert=False,
    ax=None,
    cumulative=False,
    kind="bar",
    prepend=None,
    params={},
    quantiles=None,
    quantile_params={},
):
    """ A convenience function for generating histograms

    Parameters
    ----------
    data: array_like
        Input data (see np.histogram)
    bins: int or sequence of scalars or str, optional; default = 10
        Defines histrogram bins (see np.histogram)
    range: (float, float), optional; default = None
        Lower and upper bounds of the bins (see np.histogram)
    density: bool, optional; default = False
        number of samples if False, density if True (see np.histogram)
        density and proportion cannot both be True
        density is such that the area under the histogram is 1 
        np.sum(frequencies * np.diff(edges)) == 1
    proportion: bool, optional; default = False
        number of samples if False, density if True (see np.histogram)
        density and proportion cannot both be True
        proportion is such that the sum of bar heights is 1 
        np.sum(frequencies) == 1
        if proportion is a numeric value, normalized to that instead
    multiplier: numeric, optional; default = 1
        A scalar to multiple all the counts by
    invert: bool, optional; default = False
        If True, all frequencies are multiplied by -1 to flip upside down
    
    Returns
    -------
    returns ax
    """
    if proportion and density:
        raise ValueError("proportion and density cannot both be enabled")

    data = np.asarray(data)
    ax = get_ax(ax)

    weights = (
        np.ones_like(data) * proportion / data.size
        if proportion
        else np.ones_like(data)
    )
    if invert:
        weights *= -1
    weights = weights * multiplier
    counts, edges = np.histogram(
        data, bins, range, density=density, weights=weights
    )

    if cumulative:
        counts = np.cumsum(counts)

    align = params.pop("align", "mid")
    if align == "left":
        bin_align_points = edges[:-1]
    elif align == "right":
        bin_align_points = edges[1:]
    else:
        bin_align_points = np.mean([edges[:-1], edges[1:]], axis=0)

    orientation = params.pop("orientation", "vertical")
    if orientation == "vertical":
        x, y = bin_align_points, counts
    elif orientation == "horizontal":
        x, y = counts, bin_align_points
    
    # prepend = params.pop("prepend", None)
    if prepend is not None:
        x = np.insert(x, 0, prepend[0])
        y = np.insert(y, 0, prepend[1])
    if kind == "bar":
        ax.bar(x, y, **params)
    elif kind == "line":
        ax.plot(x, y, **params)

    if quantiles is not None:
        if not cumulative:
            raise ValueError("quantiles can only be used with cumulative=True")
        if not proportion:
            quantiles = counts[-1] * quantiles
        if orientation == "vertical":
            quantile_params["hlines"] = quantiles
        elif orientation == "horizontal":
            quantile_params["vlines"] = quantiles

        draw_lines(x, y, ax=ax, **quantile_params)
    return ax