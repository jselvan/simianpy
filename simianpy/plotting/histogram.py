import numpy as np
import matplotlib.pyplot as plt 

try:
    import holoviews as hv
except ImportError:
    hv = None
else:
    from holoviews import opts, dim

def _histogram_holoviews(edges, frequencies):
    if hv is None:
        raise ImportError("holoviews module could not be imported. use a different engine")
    hist = hv.Histogram((edges, frequencies))
    return hist

def Histogram(data, bins=10, range=None, invert=False, engine='holoviews'):
    """ A convenience function for generating histograms

    Parameters
    ----------
    data: array_like
        Input data (see np.histogram)
    bins: int or sequence of scalars or str, optional; default = 10
        Defines histrogram bins (see np.histogram)
    range: (float, float), optional; default = None
        Lower and upper bounds of the bins (see np.histogram)
    invert: bool, optional; default = False
        If True, all frequencies are multiplied by -1 to flip upside down
    engine: str, optional; default = 'holoviews'
        Which plotting engine is to be used. Defaults to holoviews
        Currently supported: ['holoviews']
    """
    frequencies, edges = np.histogram(data, bins, range)
    if invert:
        frequencies *= -1
    if engine == 'holoviews':
        hist = _histogram_holoviews(edges, frequencies)
    else:
        raise ValueError(f"Engine not implemented: {engine}. Choose one of: ['holoviews']")

    return hist