from ..analysis import LinearRegression
from .util import get_ax

import numpy as np
import matplotlib.pyplot as plt 

try:
    import holoviews as hv
except ImportError:
    hv = None
else:
    from holoviews import opts, dim

def _regression_holoviews(regression_output, fitline, scatter):
    if hv is None:
        raise ImportError("holoviews module could not be imported. use a different engine")

    plots = []

    if fitline:
        fitline_plot = hv.Curve(
            (regression_output.x_pred, regression_output.y_pred), 
            regression_output.x_label, 
            regression_output.y_label
        )
        plots.append(fitline_plot)

    if scatter:
        scatter_plot = hv.Scatter(
            regression_output.data,
            regression_output.x_label, 
            regression_output.y_label
        )
        plots.append(scatter_plot)

    return hv.Overlay(plots) if plots else None

def Regression(x, y, data=None, drop_na=True, fitline=True, scatter=True, engine='matplotlib', ax=None, fitline_kwargs={}, scatter_kwargs={}):
    """ A convenience function for generating a regression plot

    Parameters
    ----------
    x, y, data, drop_na
        see simi.analysis.linear_regression.LinearRegression
    fitline: bool; default=True
        determines if a fitline is plotted
    scatter: bool; default=True
        determines if the original data is plotted
    engine: str; default='matplotlib'
    ax: Axes; default=None
    fitline_kwargs: dict-like; default={}
    scatter_kwargs: dict-like; default={}
    """
    regression_output = LinearRegression(x, y, data, drop_na)
    
    if engine == 'holoviews':
        plot = _regression_holoviews(regression_output, fitline, scatter)
        return plot
    elif engine == 'matplotlib':
        ax = get_ax(ax)
        if fitline:
            ax.plot(regression_output.x_pred, regression_output.y_pred, **fitline_kwargs)
        if scatter:
            ax.scatter(
                regression_output.data[regression_output.x_label], 
                regression_output.data[regression_output.y_label], 
                **scatter_kwargs
            )
        return ax
    else:
        raise ValueError(f"Engine not implemented: {engine}. Choose one of: ['holoviews','matplotlib']")