import matplotlib.pyplot as plt
import numpy as np

from simianpy.plotting.util import get_ax

try:
    import holoviews as hv
except ImportError:
    hv = None
else:
    from holoviews import dim, opts

default_params = {"matplotlib": {}, "holoviews": {}}


def _histogram_holoviews(edges, frequencies):
    if hv is None:
        raise ImportError(
            "holoviews module could not be imported. use a different engine"
        )
    hist = hv.Histogram((edges, frequencies))
    return hist


def Histogram(
    data,
    bins=10,
    range=None,
    density=False,
    proportion=False,
    multiplier=1,
    invert=False,
    engine="matplotlib",
    ax=None,
    params={},
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
    engine: str, optional; default = 'holoviews'
        Which plotting engine is to be used. Defaults to holoviews
        Currently supported: ['holoviews']
    
    Returns
    -------
    if 'engine' is 'holoviews', returns hv.Histogram
    if 'engine' is 'matplotlib', returns ax
    """
    data = np.asarray(data)

    if proportion and density:
        raise ValueError("proportion and density cannot both be enabled")

    weights = (
        np.ones_like(data) * proportion / data.size
        if proportion
        else np.ones_like(data)
    )
    if invert:
        weights *= -1
    weights = weights * multiplier
    frequencies, edges = np.histogram(
        data, bins, range, density=density, weights=weights
    )

    if engine == "holoviews":
        if invert:
            frequencies *= -1
        frequencies = frequencies * multiplier
        if proportion or density:
            raise NotImplementedError
        hist = _histogram_holoviews(edges, frequencies)
        return hist
    elif engine == "matplotlib":
        ax = get_ax(ax)
        if params.get("histtype") == "line":
            params.pop("histtype")
            cumulative = params.pop("cumulative", False)
            counts, edges = np.histogram(
                data, bins=bins, range=range, density=density, weights=weights
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
                ax.plot(bin_align_points, counts, **params)
            elif orientation == "horizontal":
                ax.plot(counts, bin_align_points, **params)
        else:
            ax.hist(
                data, bins=bins, range=range, density=density, weights=weights, **params
            )
        return ax
    else:
        raise ValueError(
            f"Engine not implemented: {engine}. Choose one of: ['holoviews','matplotlib']"
        )
